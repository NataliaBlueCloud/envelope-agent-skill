"""
Autonomous Envelope Monitor

Permanent monitoring mode that:
1. Captures packets in 10k sample windows
2. Fits envelope model
3. If error > threshold (15%), runs polynomial fitting
4. Estimates percentiles
5. Continues monitoring

Designed for long-running network monitoring with automatic model adaptation.
"""

import numpy as np
import time
import signal
import sys
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime

from .envelope_agent import EnvelopeAgent
from .tcpdump_capture import TCPDumpCapture
from .traffic_generator import TrafficGenerator


@dataclass
class MonitoringEvent:
    """Records a monitoring cycle event."""
    timestamp: str
    cycle_number: int
    packets_captured: int
    rho_real: float
    rho_env: float
    p99_bound: float
    model_error_pct: float
    action_taken: str  # 'normal', 'polynomial_refit', 'alert'


@dataclass
class AutonomousState:
    """State of the autonomous monitor."""
    cycle_count: int = 0
    current_model: Dict = field(default_factory=dict)
    polynomial_model: Dict = field(default_factory=dict)
    event_history: List[MonitoringEvent] = field(default_factory=list)
    total_packets_processed: int = 0
    last_refit_cycle: int = 0


class AutonomousEnvelopeMonitor:
    """
    Autonomous permanent monitoring system.
    
    Continuously captures packets, fits models, and adapts when errors exceed threshold.
    
    Example:
        monitor = AutonomousEnvelopeMonitor(
            interface='eth0',
            window_size=10000,
            error_threshold_pct=15.0,
            link_rate_gbps=10
        )
        
        # Run forever (or specify max_cycles)
        monitor.run_permanent(max_cycles=None, callback=print_results)
    """
    
    def __init__(self,
                 interface: str = 'any',
                 window_size: int = 10000,
                 error_threshold_pct: float = 15.0,
                 link_rate_gbps: float = 10,
                 filter_expr: Optional[str] = None,
                 use_live_capture: bool = True):
        """
        Initialize autonomous monitor.
        
        Args:
            interface: Network interface for capture
            window_size: Number of packets per window (default 10k)
            error_threshold_pct: Trigger polynomial refit if error > this
            link_rate_gbps: Link capacity
            filter_expr: tcpdump filter expression
            use_live_capture: If False, uses synthetic traffic for testing
        """
        self.interface = interface
        self.window_size = window_size
        self.error_threshold_pct = error_threshold_pct
        self.link_rate_gbps = link_rate_gbps
        self.filter_expr = filter_expr
        self.use_live_capture = use_live_capture
        
        self.agent = EnvelopeAgent(link_rate_gbps)
        self.state = AutonomousState()
        self._running = False
        self._generator = None if use_live_capture else TrafficGenerator(link_rate_gbps)
        self._historical_data: List[Dict] = []  # Store (rho_real, rho_env) pairs
        
    def _capture_window(self) -> Dict:
        """
        Capture one window of packets.
        
        Returns:
            Dict with delays, packet_sizes, timestamps
        """
        if self.use_live_capture:
            # Live capture with tcpdump
            capture = TCPDumpCapture(
                interface=self.interface,
                link_rate_gbps=self.link_rate_gbps,
                filter_expr=self.filter_expr
            )
            
            print(f"  Capturing {self.window_size} packets...")
            result = capture.capture_and_analyze(
                packet_count=self.window_size
            )
            
            return {
                'delays': None,  # Already processed in capture
                'packet_sizes': None,
                'result': result,
                'from_capture': True
            }
        else:
            # Synthetic traffic for testing
            if self._generator is None:
                self._generator = TrafficGenerator(self.link_rate_gbps)
            
            # Generate at realistic load
            rho = 0.6 + np.random.uniform(-0.1, 0.1)  # Vary load slightly
            delays, service_times = self._generator.generate_sfmix_like(
                rho=rho,
                num_packets=self.window_size,
                seed=int(time.time()) % 10000
            )
            
            return {
                'delays': delays,
                'packet_sizes': None,
                'result': None,
                'from_capture': False
            }
    
    def _fit_model(self, data: Dict) -> Dict:
        """
        Fit envelope model to captured data (ground truth).
        
        Args:
            data: Output from _capture_window
            
        Returns:
            Fitted model result
        """
        if data.get('from_capture'):
            # Already fitted in capture
            return data['result']
        else:
            # Fit from synthetic delays
            avg_packet_bytes = 1750  # SFM-IX mean
            return self.agent.fit_envelope(
                delay_samples=data['delays'],
                avg_packet_bytes=avg_packet_bytes,
                link_rate_gbps=self.link_rate_gbps
            )
    
    def _calculate_rho_real(self, delays: np.ndarray, avg_packet_bytes: float) -> float:
        """
        Calculate actual rho_real from delay samples.
        
        Args:
            delays: Delay samples in microseconds
            avg_packet_bytes: Average packet size
            
        Returns:
            Calculated rho_real
        """
        service_time_us = self.agent.compute_service_time(avg_packet_bytes)
        avg_delay_real = np.mean(delays)
        # From E(D) = E(X)/(1-ρ), solve for ρ
        rho_real = 1 - (service_time_us / avg_delay_real)
        return max(0.01, min(0.99, rho_real))
    
    def _predict_with_polynomial(self, rho_real: float) -> Dict:
        """
        Predict envelope parameters using saved polynomial model.
        
        Args:
            rho_real: Real system load
            
        Returns:
            Prediction result
        """
        if not self.state.polynomial_model:
            # No model yet, need to fit first
            return None
        
        a = self.state.polynomial_model['coefficients']['a']
        b = self.state.polynomial_model['coefficients']['b']
        c = self.state.polynomial_model['coefficients']['c']
        
        rho_env_predicted = a + b * rho_real + c * (rho_real ** 2)
        rho_env_predicted = min(0.99, rho_env_predicted)
        
        return {
            'rho_real': rho_real,
            'rho_env': rho_env_predicted,
            'envelope_load_factor': rho_env_predicted / rho_real if rho_real > 0 else 1,
            'prediction': True
        }
    
    def _estimate_error(self, current_result: Dict) -> float:
        """
        Estimate model error by comparing with polynomial prediction.
        
        Args:
            current_result: Current envelope fit
            
        Returns:
            Error percentage
        """
        if not self.state.polynomial_model:
            return 0.0  # No model to compare yet
        
        rho_real = current_result['rho_real']
        rho_env_actual = current_result['rho_env']
        
        # Predict using polynomial model
        a = self.state.polynomial_model['coefficients']['a']
        b = self.state.polynomial_model['coefficients']['b']
        c = self.state.polynomial_model['coefficients']['c']
        
        rho_env_predicted = a + b * rho_real + c * (rho_real ** 2)
        rho_env_predicted = min(0.99, rho_env_predicted)
        
        # Calculate error
        if rho_env_actual > 0:
            error_pct = abs(rho_env_predicted - rho_env_actual) / rho_env_actual * 100
        else:
            error_pct = 0.0
        
        return error_pct
    
    def _run_polynomial_fitting(self, historical_results: List[Dict]) -> Dict:
        """
        Run polynomial fitting on historical data.
        
        Args:
            historical_results: List of past envelope results
            
        Returns:
            Polynomial model
        """
        print(f"  Running polynomial fitting on {len(historical_results)} data points...")
        
        rho_real_vals = [r['rho_real'] for r in historical_results]
        rho_env_vals = [r['rho_env'] for r in historical_results]
        
        # Fit polynomial
        coeffs = self.agent.fit_polynomial_mapping(
            np.array(rho_real_vals),
            np.array(rho_env_vals),
            degree=2
        )
        
        a, b, c = coeffs
        
        # Compute R²
        predicted = a + b * np.array(rho_real_vals) + c * np.array(rho_real_vals)**2
        ss_res = np.sum((np.array(rho_env_vals) - predicted) ** 2)
        ss_tot = np.sum((np.array(rho_env_vals) - np.mean(rho_env_vals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        model = {
            'coefficients': {'a': float(a), 'b': float(b), 'c': float(c)},
            'polynomial': f"ρ_env = {a:.4f} + {b:.4f}·ρ_real + {c:.4f}·ρ_real²",
            'r_squared': float(r_squared),
            'training_points': len(historical_results)
        }
        
        self.state.polynomial_model = model
        self.state.last_refit_cycle = self.state.cycle_count
        
        print(f"  New model: {model['polynomial']}")
        print(f"  R² = {r_squared:.4f}")
        
        return model
    
    def _estimate_percentiles(self, result: Dict) -> Dict:
        """
        Estimate delay percentiles from fitted model.
        
        Args:
            result: Envelope fit result
            
        Returns:
            Percentile estimates
        """
        rho_env = result['rho_env']
        service_time = result['avg_service_time_us']
        
        percentiles = {}
        for p in [50, 90, 95, 99, 99.9]:
            percentiles[f'p{p}'] = self.agent.compute_delay_percentile_mm1(
                rho_env, service_time, p / 100
            )
        
        return percentiles
    
    def run_cycle(self) -> Dict:
        """
        Run one complete monitoring cycle.
        
        Returns:
            Cycle results
        """
        self.state.cycle_count += 1
        cycle_num = self.state.cycle_count
        
        print(f"\n{'='*70}")
        print(f" CYCLE {cycle_num}")
        print(f"{'='*70}")
        
        # Step 1: Capture packets
        print(f"[1/5] Capturing {self.window_size} samples...")
        capture_data = self._capture_window()
        
        # Extract actual delays and calculate rho_real from measurement
        if capture_data.get('from_capture'):
            actual_delays = None  # Need to extract from result
            # For capture, we need to re-simulate or extract from pcap
            # For now, use the envelope fit as ground truth
            ground_truth = self._fit_model(capture_data)
            actual_delays = np.random.normal(ground_truth['avg_delay_real_us'], 
                                            ground_truth['avg_delay_real_us'] * 0.3, 
                                            1000)  # Simulated for error calc
            rho_real = ground_truth['rho_real']
            avg_packet_bytes = 1750
        else:
            actual_delays = capture_data['delays']
            avg_packet_bytes = 1750
            rho_real = self._calculate_rho_real(actual_delays, avg_packet_bytes)
        
        # Step 2: Predict using saved polynomial model
        print(f"[2/5] Predicting with saved polynomial model...")
        prediction = self._predict_with_polynomial(rho_real)
        
        if prediction is None:
            print(f"      No saved model yet. Fitting initial model...")
            # First cycle - fit polynomial from single point (will improve over time)
            ground_truth = self._fit_model(capture_data)
            self.state.polynomial_model = {
                'coefficients': {'a': 0.1, 'b': 0.8, 'c': 0.1},  # Default initial
                'polynomial': 'ρ_env = 0.1000 + 0.8000·ρ_real + 0.1000·ρ_real²',
                'r_squared': 0.0,
                'training_points': 0
            }
            prediction = self._predict_with_polynomial(rho_real)
        
        # Step 3: Calculate error (compare prediction with ground truth envelope)
        print(f"[3/5] Calculating prediction error...")
        
        # Get ground truth by fitting envelope to actual captured data
        ground_truth = self._fit_model(capture_data)
        rho_env_actual = ground_truth['rho_env']
        rho_env_predicted = prediction['rho_env']
        
        # Calculate error percentage
        if rho_env_actual > 0:
            error_pct = abs(rho_env_predicted - rho_env_actual) / rho_env_actual * 100
        else:
            error_pct = 0.0
        
        print(f"      Predicted ρ_env: {rho_env_predicted:.4f}")
        print(f"      Actual ρ_env:    {rho_env_actual:.4f}")
        print(f"      Error:           {error_pct:.1f}%")
        
        # Step 4: Check if polynomial refit needed
        action = 'normal'
        if error_pct > self.error_threshold_pct:
            print(f"      ⚠️  Error {error_pct:.1f}% > threshold {self.error_threshold_pct}%")
            print(f"[4/5] Running polynomial fitting algorithm...")
            
            # Collect historical data for refit
            # Store this cycle's ground truth
            self._historical_data.append({
                'rho_real': rho_real,
                'rho_env': rho_env_actual
            })
            
            if len(self._historical_data) >= 3:
                self._run_polynomial_fitting(self._historical_data[-10:])
                action = 'polynomial_refit'
                
                # Re-predict with new model
                prediction = self._predict_with_polynomial(rho_real)
            else:
                print(f"      Not enough historical data ({len(self._historical_data)} points)")
                action = 'insufficient_data'
        else:
            print(f"[4/5] Error within threshold ✓")
            # Still add to historical data
            self._historical_data.append({
                'rho_real': rho_real,
                'rho_env': rho_env_actual
            })
        
        # Step 5: Estimate percentiles using final model (predicted or refitted)
        print(f"[5/5] Estimating percentiles...")
        
        # Use the predicted (or refitted) rho_env for percentile estimation
        final_rho_env = prediction['rho_env']
        service_time = ground_truth['avg_service_time_us']
        
        percentiles = {}
        for p in [50, 90, 95, 99, 99.9]:
            percentiles[f'p{p}'] = self.agent.compute_delay_percentile_mm1(
                final_rho_env, service_time, p / 100
            )
        
        # Record event
        event = MonitoringEvent(
            timestamp=datetime.now().isoformat(),
            cycle_number=cycle_num,
            packets_captured=self.window_size,
            rho_real=rho_real,
            rho_env=final_rho_env,
            p99_bound=percentiles['p99'],
            model_error_pct=error_pct,
            action_taken=action
        )
        self.state.event_history.append(event)
        self.state.total_packets_processed += self.window_size
        
        # Update current model
        self.state.current_model = {
            'rho_real': rho_real,
            'rho_env': final_rho_env,
            'envelope_load_factor': final_rho_env / rho_real if rho_real > 0 else 1,
            'avg_service_time_us': service_time,
            'avg_delay_real_us': ground_truth['avg_delay_real_us']
        }
        
        # Build result
        result = {
            'cycle': cycle_num,
            'timestamp': event.timestamp,
            'packets_captured': self.window_size,
            'rho_real': rho_real,
            'rho_env_predicted': rho_env_predicted,
            'rho_env_actual': rho_env_actual,
            'envelope_load_factor': final_rho_env / rho_real if rho_real > 0 else 1,
            'percentiles': percentiles,
            'model_error_pct': error_pct,
            'action_taken': action,
            'fit_quality': ground_truth.get('fit_quality', 'unknown')
        }
        
        return result
    
    def run_permanent(self,
                     max_cycles: Optional[int] = None,
                     callback: Optional[Callable[[Dict], None]] = None):
        """
        Run permanent monitoring loop.
        
        Args:
            max_cycles: Stop after N cycles (None = run forever)
            callback: Called with result after each cycle
        """
        self._running = True
        
        def signal_handler(sig, frame):
            print("\n\nShutdown signal received. Stopping monitor...")
            self._running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print(f"\n{'='*70}")
        print(" AUTONOMOUS ENVELOPE MONITOR")
        print(f"{'='*70}")
        print(f" Mode:           {'Live capture' if self.use_live_capture else 'Synthetic'}")
        print(f" Interface:      {self.interface}")
        print(f" Window size:    {self.window_size} packets")
        print(f" Error threshold: {self.error_threshold_pct}%")
        print(f" Max cycles:     {'∞ (infinite)' if max_cycles is None else max_cycles}")
        print(f"{'='*70}")
        print(" Press Ctrl+C to stop\n")
        
        try:
            while self._running:
                # Check if max cycles reached
                if max_cycles and self.state.cycle_count >= max_cycles:
                    print(f"\nReached max cycles ({max_cycles}). Stopping.")
                    break
                
                # Run one cycle
                result = self.run_cycle()
                
                # Call callback if provided
                if callback:
                    callback(result)
                
                # Print summary
                self._print_cycle_summary(result)
                
        except Exception as e:
            print(f"\nError in monitoring loop: {e}")
            raise
        finally:
            self._running = False
            self._print_final_summary()
    
    def _print_cycle_summary(self, result: Dict):
        """Print summary of cycle results."""
        print(f"\n  Cycle {result['cycle']} Summary:")
        print(f"    ρ_real: {result['rho_real']:.4f}")
        print(f"    Predicted ρ_env: {result['rho_env_predicted']:.4f}")
        print(f"    Actual ρ_env:    {result['rho_env_actual']:.4f}")
        print(f"    P50: {result['percentiles']['p50']:.2f} μs")
        print(f"    P90: {result['percentiles']['p90']:.2f} μs")
        print(f"    P99: {result['percentiles']['p99']:.2f} μs")
        print(f"    Error: {result['model_error_pct']:.1f}% | Action: {result['action_taken']}")
    
    def _print_final_summary(self):
        """Print final monitoring summary."""
        print(f"\n{'='*70}")
        print(" FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"  Total cycles:           {self.state.cycle_count}")
        print(f"  Total packets:          {self.state.total_packets_processed:,}")
        print(f"  Polynomial refits:      {sum(1 for e in self.state.event_history if e.action_taken == 'polynomial_refit')}")
        if self.state.polynomial_model:
            print(f"  Final model:            {self.state.polynomial_model['polynomial']}")
            print(f"  Model R²:               {self.state.polynomial_model['r_squared']:.4f}")
        print(f"{'='*70}")
    
    def get_status(self) -> Dict:
        """Get current monitor status."""
        return {
            'running': self._running,
            'cycle_count': self.state.cycle_count,
            'total_packets': self.state.total_packets_processed,
            'current_model': self.state.current_model,
            'polynomial_model': self.state.polynomial_model,
            'event_history': [
                {
                    'cycle': e.cycle_number,
                    'timestamp': e.timestamp,
                    'rho_real': e.rho_real,
                    'rho_env': e.rho_env,
                    'error_pct': e.model_error_pct,
                    'action': e.action_taken
                }
                for e in self.state.event_history
            ]
        }


# Convenience functions

def run_autonomous_monitor(interface: str = 'any',
                           window_size: int = 10000,
                           error_threshold_pct: float = 15.0,
                           link_rate_gbps: float = 10,
                           max_cycles: Optional[int] = None,
                           use_live_capture: bool = True) -> Dict:
    """
    One-shot function to run autonomous monitor.
    
    Example:
        # Run forever with live capture
        run_autonomous_monitor('eth0', max_cycles=None)
        
        # Run 10 cycles with synthetic traffic (for testing)
        run_autonomous_monitor(max_cycles=10, use_live_capture=False)
    """
    monitor = AutonomousEnvelopeMonitor(
        interface=interface,
        window_size=window_size,
        error_threshold_pct=error_threshold_pct,
        link_rate_gbps=link_rate_gbps,
        use_live_capture=use_live_capture
    )
    
    monitor.run_permanent(max_cycles=max_cycles)
    
    return monitor.get_status()
