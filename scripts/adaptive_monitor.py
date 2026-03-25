"""
Adaptive Envelope Monitor

Continuously monitors traffic and auto-recalibrates envelope models
when distribution drift is detected.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from scipy import stats

from .envelope_agent import EnvelopeAgent, EnvelopeResult


@dataclass
class DriftEvent:
    """Records a detected distribution change."""
    timestamp: float
    old_scv: float
    new_scv: float
    scv_change_pct: float
    old_model: Dict
    new_model: Dict
    refit_triggered: bool


@dataclass
class AdaptiveState:
    """Current state of the adaptive monitor."""
    window_fills: int = 0
    current_scv: float = 0.0
    baseline_scv: float = 0.0
    drift_count: int = 0
    last_refit_time: float = 0.0
    current_model: Dict = field(default_factory=dict)
    model_history: List[Dict] = field(default_factory=list)
    drift_history: List[DriftEvent] = field(default_factory=list)


class AdaptiveEnvelopeMonitor:
    """
    Self-calibrating envelope monitor with drift detection.
    
    Maintains sliding window of delays, detects distribution changes,
    and auto-refits envelope models.
    
    Example:
        monitor = AdaptiveEnvelopeMonitor(link_rate_gbps=10)
        
        # Feed packets continuously
        for packet in stream:
            result = monitor.update(packet.delay_us, packet.size_bytes)
            if result and result['drift_detected']:
                print(f"New model: {result['model']}")
    """
    
    def __init__(self,
                 link_rate_gbps: float = 10,
                 window_size: int = 10000,
                 drift_threshold_pct: float = 15.0,
                 min_samples_for_fit: int = 5000,
                 cooldown_packets: int = 5000,
                 auto_refit: bool = True,
                 scv_check_interval: int = 1000):
        """
        Initialize adaptive monitor.
        
        Args:
            link_rate_gbps: Link capacity
            window_size: Sliding window size (packets)
            drift_threshold_pct: SCV change % to trigger drift alert
            min_samples_for_fit: Min samples before fitting
            cooldown_packets: Packets to wait between refits
            auto_refit: Automatically refit on drift detection
            scv_check_interval: Check SCV every N packets
        """
        self.agent = EnvelopeAgent(link_rate_gbps)
        self.link_rate_gbps = link_rate_gbps
        
        # Window management
        self.window_size = window_size
        self.delays = deque(maxlen=window_size)
        self.packet_sizes = deque(maxlen=window_size)
        
        # Configuration
        self.drift_threshold_pct = drift_threshold_pct
        self.min_samples_for_fit = min_samples_for_fit
        self.cooldown_packets = cooldown_packets
        self.auto_refit = auto_refit
        self.scv_check_interval = scv_check_interval
        
        # State
        self.state = AdaptiveState()
        self.packet_count = 0
        self._last_check_packet = 0
        self._packet_sizes_since_refit = []
        
    def update(self, delay_us: float, packet_bytes: float) -> Optional[Dict]:
        """
        Process a new packet delay sample.
        
        Args:
            delay_us: Packet delay in microseconds
            packet_bytes: Packet size in bytes
            
        Returns:
            Result dict if drift detected or model updated, None otherwise
        """
        self.packet_count += 1
        self.delays.append(delay_us)
        self.packet_sizes.append(packet_bytes)
        
        # Store sizes for model fitting
        self._packet_sizes_since_refit.append(packet_bytes)
        
        # Check if we have enough samples
        if len(self.delays) < self.min_samples_for_fit:
            return None
        
        # Periodic SCV check
        if (self.packet_count - self._last_check_packet) >= self.scv_check_interval:
            self._last_check_packet = self.packet_count
            return self._check_drift()
        
        return None
    
    def _check_drift(self) -> Optional[Dict]:
        """Check for distribution drift and refit if needed."""
        delays_array = np.array(self.delays)
        sizes_array = np.array(self.packet_sizes)
        
        # Compute current SCV (squared coefficient of variation)
        current_scv = self._compute_scv(sizes_array)
        self.state.current_scv = current_scv
        
        # First window — establish baseline
        if self.state.window_fills == 0:
            self.state.baseline_scv = current_scv
            self.state.window_fills += 1
            
            # Initial model fit
            if self.auto_refit:
                return self._refit_model()
            return None
        
        # Check for drift
        if self.state.baseline_scv > 0:
            scv_change_pct = abs(current_scv - self.state.baseline_scv) / self.state.baseline_scv * 100
        else:
            scv_change_pct = 0.0
        drift_detected = scv_change_pct > self.drift_threshold_pct
        
        # Check cooldown
        packets_since_refit = self.packet_count - self.state.last_refit_time
        can_refit = packets_since_refit >= self.cooldown_packets
        
        result = {
            'packet_count': self.packet_count,
            'current_scv': current_scv,
            'baseline_scv': self.state.baseline_scv,
            'scv_change_pct': scv_change_pct,
            'drift_detected': drift_detected,
            'can_refit': can_refit,
            'refit_triggered': False
        }
        
        if drift_detected and can_refit and self.auto_refit:
            refit_result = self._refit_model()
            if refit_result:
                result.update(refit_result)
                result['refit_triggered'] = True
                
                # Record drift event
                event = DriftEvent(
                    timestamp=self.packet_count,
                    old_scv=self.state.baseline_scv,
                    new_scv=current_scv,
                    scv_change_pct=scv_change_pct,
                    old_model=self.state.current_model.copy() if self.state.current_model else {},
                    new_model=result.get('model', {}),
                    refit_triggered=True
                )
                self.state.drift_history.append(event)
                self.state.drift_count += 1
                
                # Update baseline
                self.state.baseline_scv = current_scv
        
        return result if (drift_detected or result.get('model')) else None
    
    def _refit_model(self) -> Optional[Dict]:
        """Refit envelope model on current window."""
        if len(self.delays) < self.min_samples_for_fit:
            return None
        
        delays_array = np.array(self.delays)
        avg_packet_bytes = np.mean(list(self.packet_sizes))
        
        # Run Algorithm 1
        result = self.agent.fit_envelope(
            delay_samples=delays_array,
            avg_packet_bytes=avg_packet_bytes,
            link_rate_gbps=self.link_rate_gbps
        )
        
        # Store model
        self.state.current_model = result.copy()
        self.state.model_history.append({
            'packet_count': self.packet_count,
            'model': result.copy()
        })
        self.state.last_refit_time = self.packet_count
        self._packet_sizes_since_refit = []
        
        return {
            'model': result,
            'recommendation': result['diagnostics']['recommendation']
        }
    
    def _compute_scv(self, packet_sizes: np.ndarray) -> float:
        """Compute squared coefficient of variation."""
        mean = np.mean(packet_sizes)
        variance = np.var(packet_sizes, ddof=1)
        if mean == 0:
            return 0.0
        return variance / (mean ** 2)
    
    def learn_polynomial_model(self, 
                               loads_to_test: Optional[List[float]] = None,
                               samples_per_load: int = 20000) -> Dict:
        """
        Learn custom polynomial model by sweeping loads.
        
        Generates synthetic traffic at different loads, fits envelope
        for each, and learns ρ_env = a + b·ρ + c·ρ² mapping.
        
        Args:
            loads_to_test: List of ρ values to test (default: 0.1 to 0.9)
            samples_per_load: Packets to generate per load
            
        Returns:
            Learned polynomial coefficients and quality metrics
        """
        from .traffic_generator import TrafficGenerator
        
        if loads_to_test is None:
            loads_to_test = np.arange(0.1, 0.95, 0.05)
        
        gen = TrafficGenerator(self.link_rate_gbps)
        rho_real_vals = []
        rho_env_vals = []
        
        print(f"Learning polynomial model over {len(loads_to_test)} load points...")
        
        for rho in loads_to_test:
            # Generate traffic at this load
            delays, service_time = gen.generate_sfmix_like(
                rho=rho, 
                num_packets=samples_per_load
            )
            
            # Fit envelope
            result = self.agent.fit_envelope(
                delay_samples=delays,
                avg_packet_bytes=1750,  # SFM-IX mean
                link_rate_gbps=self.link_rate_gbps
            )
            
            rho_real_vals.append(result['rho_real'])
            rho_env_vals.append(result['rho_env'])
            
            print(f"  ρ_real={result['rho_real']:.3f} → ρ_env={result['rho_env']:.3f}")
        
        # Fit polynomial
        coeffs = self.agent.fit_polynomial_mapping(
            np.array(rho_real_vals),
            np.array(rho_env_vals),
            degree=2
        )
        
        a, b, c = coeffs
        
        # Compute R²
        predicted = a + b * np.array(rho_real_vals) + c * np.array(rho_real_vals) ** 2
        ss_res = np.sum((np.array(rho_env_vals) - predicted) ** 2)
        ss_tot = np.sum((np.array(rho_env_vals) - np.mean(rho_env_vals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        model = {
            'coefficients': {'a': a, 'b': b, 'c': c},
            'polynomial': f"ρ_env = {a:.4f} + {b:.4f}·ρ_real + {c:.4f}·ρ_real²",
            'r_squared': r_squared,
            'load_points': len(loads_to_test),
            'rho_real_range': (min(rho_real_vals), max(rho_real_vals)),
            'training_data': {
                'rho_real': rho_real_vals,
                'rho_env': rho_env_vals
            }
        }
        
        # Store as current model
        self.state.current_model['learned_polynomial'] = model
        
        return model
    
    def get_status(self) -> Dict:
        """Get current monitor status."""
        return {
            'packet_count': self.packet_count,
            'window_size': len(self.delays),
            'current_scv': self.state.current_scv,
            'baseline_scv': self.state.baseline_scv,
            'drift_count': self.state.drift_count,
            'refit_count': len(self.state.model_history),
            'current_model': self.state.current_model,
            'drift_history': [
                {
                    'packet': e.timestamp,
                    'scv_change_pct': e.scv_change_pct,
                    'refit_triggered': e.refit_triggered
                }
                for e in self.state.drift_history
            ]
        }
    
    def get_current_bounds(self, percentile: float = 0.99) -> Optional[Dict]:
        """Get current delay bounds for specified percentile."""
        if not self.state.current_model:
            return None
        
        model = self.state.current_model
        rho_env = model.get('rho_env', 0)
        service_time = model.get('avg_service_time_us', 0)
        
        bound = self.agent.compute_delay_percentile_mm1(
            rho_env, service_time, percentile
        )
        
        return {
            'percentile': percentile,
            'bound_us': bound,
            'rho_env': rho_env,
            'service_time_us': service_time
        }
    
    def predict_with_current_model(self, rho_real: float) -> Optional[Dict]:
        """Predict ρ_env using learned polynomial if available."""
        poly = self.state.current_model.get('learned_polynomial')
        if not poly:
            return None
        
        a = poly['coefficients']['a']
        b = poly['coefficients']['b']
        c = poly['coefficients']['c']
        
        rho_env = a + b * rho_real + c * (rho_real ** 2)
        rho_env = min(0.99, rho_env)
        
        return {
            'rho_real': rho_real,
            'rho_env': rho_env,
            'using_learned_model': True
        }


# Convenience function for quick setup
def create_monitor(link_rate_gbps: float = 10,
                   window_size: int = 10000,
                   drift_threshold_pct: float = 15.0) -> AdaptiveEnvelopeMonitor:
    """Create configured adaptive monitor."""
    return AdaptiveEnvelopeMonitor(
        link_rate_gbps=link_rate_gbps,
        window_size=window_size,
        drift_threshold_pct=drift_threshold_pct
    )
