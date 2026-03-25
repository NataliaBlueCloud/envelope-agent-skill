"""
Self-Calibrating Envelope Agent
Implements Algorithm 1 from "Upper bound latency percentiles for high-speed coherent pluggables"

Learn the mapping ρ_real → ρ_env from empirical traffic data.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats


@dataclass
class EnvelopeResult:
    """Result of envelope fitting."""
    rho_real: float
    rho_env: float
    envelope_load_factor: float
    avg_service_time_us: float
    avg_delay_real_us: float
    avg_delay_env_us: float
    p90_bound_us: float
    p99_bound_us: float
    fit_quality: str
    confidence: float
    max_violation: float
    diagnostics: Dict


class EnvelopeAgent:
    """
    Self-calibrating agent for learning M/M/1 envelope models from M/G/1 traffic.
    """
    
    # Pre-fitted polynomial models from the paper (traffic type -> [a, b, c])
    DEFAULT_MODELS = {
        'trimodal': [0.49, 0.13, 0.39],
        'amsix': [0.43, 0.13, 0.47],
        'sfmix': [0.50, 0.16, 0.34],
    }
    
    def __init__(self, link_rate_gbps: float = 10):
        """
        Initialize the envelope agent.
        
        Args:
            link_rate_gbps: Link capacity in Gbps (default: 10)
        """
        self.link_rate_gbps = link_rate_gbps
        self.link_rate_bps = link_rate_gbps * 1e9
        
    def compute_service_time(self, avg_packet_bytes: float) -> float:
        """
        Compute average service time E(X) = 8L/C.
        
        Args:
            avg_packet_bytes: Average packet length in bytes
            
        Returns:
            Service time in microseconds
        """
        service_time_s = (8 * avg_packet_bytes) / self.link_rate_bps
        return service_time_s * 1e6  # Convert to μs
    
    def _simulate_mg1_fifo(self, 
                          arrival_times: np.ndarray,
                          service_times: np.ndarray) -> np.ndarray:
        """
        Simulate FIFO M/G/1 queue using Lindley equation.
        
        W_{n+1} = max(0, W_n + S_n - T_{n+1})
        
        Where W_n is waiting time, S_n is service time, T_{n+1} is inter-arrival.
        
        Args:
            arrival_times: Arrival times (seconds)
            service_times: Service times (microseconds)
            
        Returns:
            Total delays (waiting + service) in microseconds
        """
        n = len(arrival_times)
        wait_times = np.zeros(n)
        
        # Inter-arrival times in microseconds
        inter_arrivals_us = np.diff(arrival_times) * 1e6
        
        # Lindley equation
        for i in range(1, n):
            wait_times[i] = max(0, wait_times[i-1] + 
                               service_times[i-1] - inter_arrivals_us[i-1])
        
        # Total delay = waiting time + service time
        delays = wait_times + service_times
        
        return delays
    
    def compute_delay_percentile_mm1(self, rho: float, 
                                      service_time_us: float, 
                                      percentile: float) -> float:
        """
        Compute delay percentile using M/M/1 formula.
        
        D_q = E(X) * (1/(1-ρ)) * ln(1/(1-q))
        
        Args:
            rho: System load (0 < ρ < 1)
            service_time_us: Average service time in microseconds
            percentile: Target percentile (0-1)
            
        Returns:
            Delay bound in microseconds
        """
        if rho >= 1:
            return float('inf')
        return service_time_us * (1 / (1 - rho)) * np.log(1 / (1 - percentile))
    
    def algorithm1(self, 
                   real_delays_us: np.ndarray,
                   service_time_us: float,
                   percentile_range: Tuple[float, float] = (0.50, 0.99),
                   percentile_step: float = 0.01,
                   rho_candidates: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Algorithm 1: Find minimum ρ_env that upper-bounds real delays.
        
        Finds min(ρ_env) such that D_env ≥ D_real for percentiles above 50%.
        
        Args:
            real_delays_us: Array of real M/G/1 delay samples
            service_time_us: Average service time E(X) in microseconds
            percentile_range: (min, max) percentiles to check
            percentile_step: Step size for percentile sequence
            rho_candidates: Array of ρ values to test (default: 0.01 to 0.99)
            
        Returns:
            (rho_env, avg_delay_env_us) - envelope parameters
        """
        if rho_candidates is None:
            rho_candidates = np.arange(0.01, 0.99, 0.01)
        
        # Compute real delay quantiles from empirical data
        percentiles_seq = np.arange(percentile_range[0], 
                                    percentile_range[1] + percentile_step, 
                                    percentile_step)
        real_quantiles = np.percentile(real_delays_us, percentiles_seq * 100)
        
        # Search for minimum ρ_env
        for rho in rho_candidates:
            # Compute M/M/1 envelope quantiles
            env_quantiles = np.array([
                self.compute_delay_percentile_mm1(rho, service_time_us, p)
                for p in percentiles_seq
            ])
            
            # Check if envelope bounds all real quantiles
            if np.all(real_quantiles < env_quantiles):
                avg_delay_env = service_time_us / (1 - rho)
                return rho, avg_delay_env
        
        # Fallback: return highest tested rho with warning
        return rho_candidates[-1], service_time_us / (1 - rho_candidates[-1])
    
    def fit_polynomial_mapping(self, 
                                rho_real_values: np.ndarray,
                                rho_env_values: np.ndarray,
                                degree: int = 2) -> np.ndarray:
        """
        Fit polynomial mapping ρ_env = f(ρ_real).
        
        Args:
            rho_real_values: Array of real loads
            rho_env_values: Array of corresponding envelope loads
            degree: Polynomial degree (default: 2)
            
        Returns:
            Polynomial coefficients [a, b, c, ...] for ρ_env = a + b·ρ + c·ρ² + ...
        """
        coeffs = np.polyfit(rho_real_values, rho_env_values, degree)
        return coeffs[::-1]  # Reverse to match a + b·x + c·x² format
    
    def estimate_scv(self, packet_sizes_bytes: np.ndarray) -> float:
        """
        Estimate squared coefficient of variation (SCV = C²) from packet sizes.
        
        C²_X = Var(X) / [E(X)]²
        
        Args:
            packet_sizes_bytes: Array of packet sizes
            
        Returns:
            Squared coefficient of variation
        """
        mean = np.mean(packet_sizes_bytes)
        variance = np.var(packet_sizes_bytes)
        return variance / (mean ** 2)
    
    def diagnose_traffic(self, 
                         real_delays_us: np.ndarray,
                         service_time_us: float,
                         rho_real: float,
                         rho_env: float) -> Dict:
        """
        Analyze traffic and provide intelligent diagnostics.
        
        Args:
            real_delays_us: Delay samples
            service_time_us: Average service time
            rho_real: Real system load
            rho_env: Fitted envelope load
            
        Returns:
            Diagnostics dictionary with recommendations
        """
        diagnostics = {
            'exponential_tail_fit': True,
            'low_load_warning': False,
            'high_variance_warning': False,
            'recommendation': 'Standard quadratic envelope suitable'
        }
        
        # Check for low load issues
        if rho_real < 0.3:
            diagnostics['low_load_warning'] = True
            diagnostics['recommendation'] = (
                'Envelope error increases for ρ < 0.3 — '
                'consider piecewise fit or higher-order polynomial'
            )
        
        # Check tail behavior
        log_delays = np.log(real_delays_us[real_delays_us > 0])
        _, p_value = stats.normaltest(log_delays[:min(5000, len(log_delays))])
        
        if p_value < 0.05:
            diagnostics['exponential_tail_fit'] = False
            diagnostics['recommendation'] = (
                'Traffic not well approximated by exponential tail — '
                'consider gamma or log-normal envelope'
            )
        
        # Check envelope tightness
        load_factor = rho_env / rho_real if rho_real > 0 else 1
        if load_factor > 2.5:
            diagnostics['high_variance_warning'] = True
            diagnostics['recommendation'] = (
                f'High envelope load factor ({load_factor:.2f}) — '
                f'envelope may be loose, consider custom traffic-specific fit'
            )
        
        return diagnostics
    
    def fit_envelope(self,
                     delay_samples: Union[List, np.ndarray],
                     avg_packet_bytes: float,
                     link_rate_gbps: Optional[float] = None,
                     custom_model: Optional[str] = None) -> Dict:
        """
        Main entry point: Learn envelope from delay samples.
        
        Args:
            delay_samples: Array of packet delay measurements (μs)
            avg_packet_bytes: Average packet size in bytes
            link_rate_gbps: Link rate (overrides constructor value)
            custom_model: Traffic model name ('trimodal', 'amsix', 'sfmix')
            
        Returns:
            Complete envelope result with diagnostics
        """
        if link_rate_gbps:
            self.link_rate_gbps = link_rate_gbps
            self.link_rate_bps = link_rate_gbps * 1e9
        
        delay_samples = np.array(delay_samples)
        service_time_us = self.compute_service_time(avg_packet_bytes)
        
        # Estimate ρ_real from delay samples
        avg_delay_real = np.mean(delay_samples)
        # From E(D) = E(X)/(1-ρ), solve for ρ
        rho_real = 1 - (service_time_us / avg_delay_real)
        rho_real = max(0.01, min(0.99, rho_real))  # Clamp to valid range
        
        # Run Algorithm 1
        rho_env, avg_delay_env = self.algorithm1(delay_samples, service_time_us)
        
        # Compute percentiles
        p90 = self.compute_delay_percentile_mm1(rho_env, service_time_us, 0.90)
        p99 = self.compute_delay_percentile_mm1(rho_env, service_time_us, 0.99)
        
        # Determine fit quality
        real_p99 = np.percentile(delay_samples, 99)
        max_violation = max(0, real_p99 - p99) / p99 if p99 > 0 else 0
        
        if max_violation < 0.05:
            fit_quality = 'excellent'
        elif max_violation < 0.15:
            fit_quality = 'good'
        elif max_violation < 0.30:
            fit_quality = 'fair'
        else:
            fit_quality = 'poor'
        
        # Get diagnostics
        diagnostics = self.diagnose_traffic(delay_samples, service_time_us, 
                                           rho_real, rho_env)
        
        # Build result
        result = EnvelopeResult(
            rho_real=rho_real,
            rho_env=rho_env,
            envelope_load_factor=rho_env / rho_real if rho_real > 0 else 1,
            avg_service_time_us=service_time_us,
            avg_delay_real_us=avg_delay_real,
            avg_delay_env_us=avg_delay_env,
            p90_bound_us=p90,
            p99_bound_us=p99,
            fit_quality=fit_quality,
            confidence=1 - max_violation,
            max_violation=max_violation,
            diagnostics=diagnostics
        )
        
        return result.__dict__
    
    def predict_from_model(self, 
                          rho_real: float,
                          traffic_type: str = 'sfmix') -> Dict:
        """
        Predict envelope parameters using pre-fitted polynomial model.
        
        Args:
            rho_real: Real system load
            traffic_type: One of 'trimodal', 'amsix', 'sfmix'
            
        Returns:
            Predicted envelope parameters
        """
        if traffic_type not in self.DEFAULT_MODELS:
            raise ValueError(f"Unknown traffic type: {traffic_type}")
        
        a, b, c = self.DEFAULT_MODELS[traffic_type]
        rho_env = a + b * rho_real + c * (rho_real ** 2)
        rho_env = min(0.99, rho_env)  # Cap at 0.99
        
        return {
            'rho_real': rho_real,
            'rho_env': rho_env,
            'polynomial': f"ρ_env = {a} + {b}·ρ_real + {c}·ρ_real²",
            'coefficients': {'a': a, 'b': b, 'c': c}
        }


def fit_envelope_from_pcap(pcap_file: str,
                           link_rate_gbps: float = 10,
                           min_packet_size: Optional[int] = None,
                           max_packet_size: Optional[int] = None,
                           parser: str = 'auto') -> Dict:
    """
    Fit envelope directly from PCAP file.
    
    Reads packet capture, extracts timestamps and sizes, simulates M/G/1,
    and fits envelope model.
    
    Args:
        pcap_file: Path to .pcap or .pcapng file
        link_rate_gbps: Link capacity in Gbps
        min_packet_size: Filter out packets smaller than this (e.g., 100 to remove ACKs)
        max_packet_size: Filter out packets larger than this
        parser: 'scapy', 'dpkt', or 'auto'
        
    Returns:
        Envelope fit result with PCAP summary
        
    Example:
        result = fit_envelope_from_pcap('capture.pcap', link_rate_gbps=10)
        print(f"Processed {result['pcap_summary']['num_packets']} packets")
        print(f"P99 bound: {result['p99_bound_us']:.2f} μs")
        
    Requires:
        pip install scapy  # or dpkt
    """
    from .pcap_parser import read_pcap_auto, filter_by_size, summarize_pcap
    
    # Read PCAP
    timestamps, packet_sizes = read_pcap_auto(pcap_file, prefer=parser)
    
    # Filter if requested
    if min_packet_size is not None or max_packet_size is not None:
        timestamps, packet_sizes = filter_by_size(
            timestamps, packet_sizes, 
            min_size=min_packet_size,
            max_size=max_packet_size
        )
    
    # Get summary before fitting
    pcap_summary = summarize_pcap(timestamps, packet_sizes)
    
    # Fit envelope from trace
    result = fit_envelope_from_trace(
        arrival_times_s=timestamps,
        packet_sizes_bytes=packet_sizes,
        link_rate_gbps=link_rate_gbps
    )
    
    # Add PCAP metadata
    result['pcap_summary'] = pcap_summary
    result['pcap_file'] = pcap_file
    
    return result


# Convenience function for quick usage
def fit_envelope(delay_samples: Union[List, np.ndarray],
                 avg_packet_bytes: float,
                 link_rate_gbps: float = 10) -> Dict:
    """
    Quick-fit envelope from delay samples.
    
    Example:
        result = fit_envelope(delays_us, avg_packet_bytes=1019, link_rate_gbps=10)
        print(f"P99 bound: {result['p99_bound_us']:.2f} μs")
    """
    agent = EnvelopeAgent(link_rate_gbps)
    return agent.fit_envelope(delay_samples, avg_packet_bytes)


def fit_envelope_from_trace(arrival_times_s: Union[List, np.ndarray],
                            packet_sizes_bytes: Union[List, np.ndarray],
                            link_rate_gbps: float = 10,
                            fifo: bool = True) -> Dict:
    """
    Fit envelope from raw packet trace (timestamps + sizes).
    
    This matches the R/simmer approach - takes packet arrival times and sizes,
    simulates the M/G/1 queue, computes delays, then fits envelope.
    
    Args:
        arrival_times_s: Packet arrival timestamps in seconds
        packet_sizes_bytes: Packet sizes in bytes
        link_rate_gbps: Link capacity in Gbps
        fifo: Use FIFO queuing (default: True)
        
    Returns:
        Envelope fit result
        
    Example:
        result = fit_envelope_from_trace(
            arrival_times_s=[0.0, 0.001, 0.002, ...],
            packet_sizes_bytes=[1500, 40, 576, ...],
            link_rate_gbps=10
        )
    """
    agent = EnvelopeAgent(link_rate_gbps)
    
    # Convert to numpy arrays
    arrival_times = np.array(arrival_times_s)
    packet_sizes = np.array(packet_sizes_bytes)
    
    # Compute service times
    service_times_us = (8 * packet_sizes / agent.link_rate_bps) * 1e6
    
    # Simulate M/G/1 queue (Lindley equation for FIFO)
    delays_us = agent._simulate_mg1_fifo(arrival_times, service_times_us)
    
    # Fit envelope
    avg_packet_bytes = np.mean(packet_sizes)
    return agent.fit_envelope(delays_us, avg_packet_bytes)


def fit_envelope_from_distribution(mean_packet_bytes: float,
                                   std_packet_bytes: float,
                                   link_rate_gbps: float = 10,
                                   target_load: float = 0.7,
                                   num_packets: int = 50000,
                                   distribution: str = 'lognormal',
                                   seed: int = 42) -> Dict:
    """
    Fit envelope from distribution parameters (no real trace needed).
    
    Generates synthetic M/G/1 traffic with specified distribution,
    then fits envelope.
    
    Args:
        mean_packet_bytes: Mean packet size
        std_packet_bytes: Standard deviation of packet sizes
        link_rate_gbps: Link capacity
        target_load: Target system load (rho)
        num_packets: Number of packets to simulate
        distribution: 'lognormal', 'bimodal', 'trimodal', 'uniform'
        seed: Random seed
        
    Returns:
        Envelope fit result with generated traffic stats
        
    Example:
        result = fit_envelope_from_distribution(
            mean_packet_bytes=1750,
            std_packet_bytes=2063,
            link_rate_gbps=10,
            target_load=0.7,
            distribution='lognormal'
        )
    """
    from .traffic_generator import TrafficGenerator
    
    np.random.seed(seed)
    agent = EnvelopeAgent(link_rate_gbps)
    gen = TrafficGenerator(link_rate_gbps)
    
    # Generate packet sizes based on distribution
    if distribution == 'lognormal':
        sigma = np.sqrt(np.log(1 + (std_packet_bytes/mean_packet_bytes)**2))
        mu = np.log(mean_packet_bytes) - sigma**2/2
        packet_sizes = np.random.lognormal(mu, sigma, num_packets)
    elif distribution == 'uniform':
        # Approximate uniform with given mean and std
        width = np.sqrt(12) * std_packet_bytes
        low = mean_packet_bytes - width/2
        high = mean_packet_bytes + width/2
        packet_sizes = np.random.uniform(low, high, num_packets)
    elif distribution == 'trimodal':
        # Default trimodal: 40B, 576B, 1500B
        sizes = [40, 576, 1500]
        probs = [7/12, 4/12, 1/12]
        packet_sizes = np.random.choice(sizes, size=num_packets, p=probs)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    packet_sizes = np.clip(packet_sizes, 64, 9000)
    
    # Compute service times
    service_times_us = (8 * packet_sizes / agent.link_rate_bps) * 1e6
    avg_service_time = np.mean(service_times_us)
    
    # Generate Poisson arrivals at target load
    arrival_rate = target_load / avg_service_time  # packets per μs
    inter_arrivals_us = np.random.exponential(1/arrival_rate, num_packets)
    arrival_times_us = np.cumsum(inter_arrivals_us)
    
    # Simulate M/G/1
    delays_us = agent._simulate_mg1_fifo(arrival_times_us/1e6, service_times_us)
    
    # Fit envelope
    result = agent.fit_envelope(delays_us, np.mean(packet_sizes))
    result['synthetic_traffic'] = {
        'distribution': distribution,
        'target_load': target_load,
        'actual_scv': (std_packet_bytes/mean_packet_bytes)**2,
        'num_packets': num_packets
    }
    
    return result


def fit_polynomial_from_multiple_traces(traces: List[Dict],
                                         link_rate_gbps: float = 10) -> Dict:
    """
    Learn polynomial model from multiple packet traces at different loads.
    
    This matches your R code workflow: collect traces at different loads,
    fit envelope for each, then fit polynomial ρ_env = a + b·ρ + c·ρ².
    
    Args:
        traces: List of dicts with keys:
                - 'arrival_times': array of timestamps
                - 'packet_sizes': array of sizes
                - 'label': optional label for this trace
        link_rate_gbps: Link capacity
        
    Returns:
        Polynomial coefficients and fit quality
        
    Example:
        traces = [
            {'arrival_times': t1, 'packet_sizes': s1, 'label': 'load_0.3'},
            {'arrival_times': t2, 'packet_sizes': s2, 'label': 'load_0.5'},
            {'arrival_times': t3, 'packet_sizes': s3, 'label': 'load_0.7'},
        ]
        model = fit_polynomial_from_multiple_traces(traces)
        print(model['polynomial'])
    """
    agent = EnvelopeAgent(link_rate_gbps)
    
    rho_real_vals = []
    rho_env_vals = []
    
    print(f"Processing {len(traces)} traces...")
    
    for trace in traces:
        result = fit_envelope_from_trace(
            trace['arrival_times'],
            trace['packet_sizes'],
            link_rate_gbps
        )
        
        rho_real_vals.append(result['rho_real'])
        rho_env_vals.append(result['rho_env'])
        
        label = trace.get('label', 'unknown')
        print(f"  {label}: ρ_real={result['rho_real']:.3f} → "
              f"ρ_env={result['rho_env']:.3f}")
    
    # Fit polynomial
    coeffs = agent.fit_polynomial_mapping(
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
    
    return {
        'coefficients': {'a': a, 'b': b, 'c': c},
        'polynomial': f"ρ_env = {a:.4f} + {b:.4f}·ρ_real + {c:.4f}·ρ_real²",
        'r_squared': r_squared,
        'training_points': list(zip(rho_real_vals, rho_env_vals))
    }
