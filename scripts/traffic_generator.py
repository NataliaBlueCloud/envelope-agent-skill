"""
Traffic Generator for M/G/1 Queuing Simulations

Generate synthetic packet delays with various packet size distributions
for testing the envelope agent.
"""

import numpy as np
from typing import Callable, Tuple


class TrafficGenerator:
    """
    Generate M/G/1 traffic with different packet size distributions.
    """
    
    def __init__(self, link_rate_gbps: float = 10):
        """
        Initialize traffic generator.
        
        Args:
            link_rate_gbps: Link capacity in Gbps
        """
        self.link_rate_gbps = link_rate_gbps
        self.link_rate_bps = link_rate_gbps * 1e9
    
    def _packet_size_to_service_time(self, packet_bytes: float) -> float:
        """Convert packet size to service time in microseconds."""
        return (8 * packet_bytes / self.link_rate_bps) * 1e6
    
    def generate_trimodal(self, 
                          rho: float,
                          num_packets: int = 100000,
                          seed: int = 42) -> Tuple[np.ndarray, float]:
        """
        Generate tri-modal packet traffic.
        
        Packet sizes: 40B (58%), 576B (33%), 1500B (8%)
        C² = 1.58
        
        Args:
            rho: Target system load (0 < ρ < 1)
            num_packets: Number of packets to generate
            seed: Random seed
            
        Returns:
            (delays_us, avg_service_time_us)
        """
        np.random.seed(seed)
        
        # Packet size distribution
        sizes = [40, 576, 1500]
        probs = [7/12, 4/12, 1/12]
        
        packet_sizes = np.random.choice(sizes, size=num_packets, p=probs)
        service_times = self._packet_size_to_service_time(packet_sizes)
        
        avg_service_time = np.mean(service_times)
        arrival_rate = rho / avg_service_time  # packets per μs
        
        # Generate arrivals (Poisson process)
        inter_arrivals = np.random.exponential(1/arrival_rate, num_packets)
        arrival_times = np.cumsum(inter_arrivals)
        
        # Simulate M/G/1 queue
        delays = self._simulate_fifo_queue(arrival_times, service_times)
        
        return delays, avg_service_time
    
    def generate_amsix_like(self,
                            rho: float,
                            num_packets: int = 100000,
                            seed: int = 42) -> Tuple[np.ndarray, float]:
        """
        Generate AMS-IX-like traffic.
        
        Mean: 1019 bytes, Std: 1162 bytes, C² = 1.30
        
        Args:
            rho: Target system load
            num_packets: Number of packets
            seed: Random seed
            
        Returns:
            (delays_us, avg_service_time_us)
        """
        np.random.seed(seed)
        
        # Log-normal distribution parameters
        mean_bytes = 1019.03
        std_bytes = 1161.66
        
        # Convert to log-normal params
        sigma = np.sqrt(np.log(1 + (std_bytes/mean_bytes)**2))
        mu = np.log(mean_bytes) - sigma**2/2
        
        packet_sizes = np.random.lognormal(mu, sigma, num_packets)
        packet_sizes = np.clip(packet_sizes, 64, 9000)  # MTU bounds
        
        service_times = self._packet_size_to_service_time(packet_sizes)
        avg_service_time = np.mean(service_times)
        arrival_rate = rho / avg_service_time
        
        inter_arrivals = np.random.exponential(1/arrival_rate, num_packets)
        arrival_times = np.cumsum(inter_arrivals)
        
        delays = self._simulate_fifo_queue(arrival_times, service_times)
        
        return delays, avg_service_time
    
    def generate_sfmix_like(self,
                            rho: float,
                            num_packets: int = 100000,
                            seed: int = 42) -> Tuple[np.ndarray, float]:
        """
        Generate SFM-IX-like traffic.
        
        Mean: 1750 bytes, Std: 2063 bytes, C² = 1.39
        
        Args:
            rho: Target system load
            num_packets: Number of packets
            seed: Random seed
            
        Returns:
            (delays_us, avg_service_time_us)
        """
        np.random.seed(seed)
        
        mean_bytes = 1750.41
        std_bytes = 2062.69
        
        sigma = np.sqrt(np.log(1 + (std_bytes/mean_bytes)**2))
        mu = np.log(mean_bytes) - sigma**2/2
        
        packet_sizes = np.random.lognormal(mu, sigma, num_packets)
        packet_sizes = np.clip(packet_sizes, 64, 9000)
        
        service_times = self._packet_size_to_service_time(packet_sizes)
        avg_service_time = np.mean(service_times)
        arrival_rate = rho / avg_service_time
        
        inter_arrivals = np.random.exponential(1/arrival_rate, num_packets)
        arrival_times = np.cumsum(inter_arrivals)
        
        delays = self._simulate_fifo_queue(arrival_times, service_times)
        
        return delays, avg_service_time
    
    def generate_custom(self,
                       rho: float,
                       packet_size_sampler: Callable[[int], np.ndarray],
                       num_packets: int = 100000,
                       seed: int = 42) -> Tuple[np.ndarray, float]:
        """
        Generate traffic with custom packet size distribution.
        
        Args:
            rho: Target system load
            packet_size_sampler: Function(n) -> array of n packet sizes
            num_packets: Number of packets
            seed: Random seed
            
        Returns:
            (delays_us, avg_service_time_us)
        """
        np.random.seed(seed)
        
        packet_sizes = packet_size_sampler(num_packets)
        service_times = self._packet_size_to_service_time(packet_sizes)
        
        avg_service_time = np.mean(service_times)
        arrival_rate = rho / avg_service_time
        
        inter_arrivals = np.random.exponential(1/arrival_rate, num_packets)
        arrival_times = np.cumsum(inter_arrivals)
        
        delays = self._simulate_fifo_queue(arrival_times, service_times)
        
        return delays, avg_service_time
    
    def _simulate_fifo_queue(self, 
                            arrival_times: np.ndarray,
                            service_times: np.ndarray) -> np.ndarray:
        """
        Simulate FIFO M/G/1 queue using Lindley equation.
        
        W_{n+1} = max(0, W_n + S_n - T_{n+1})
        
        Where:
          W_n = waiting time of packet n
          S_n = service time of packet n
          T_{n+1} = inter-arrival time between n and n+1
        """
        n = len(arrival_times)
        wait_times = np.zeros(n)
        
        # Inter-arrival times
        inter_arrivals = np.diff(arrival_times)
        
        # Lindley equation
        for i in range(1, n):
            wait_times[i] = max(0, wait_times[i-1] + 
                               service_times[i-1] - inter_arrivals[i-1])
        
        # Total delay = waiting time + service time
        delays = wait_times + service_times
        
        return delays


# Convenience functions
def generate_trimodal_traffic(rho: float, 
                               num_packets: int = 100000,
                               link_rate_gbps: float = 10) -> np.ndarray:
    """Generate tri-modal traffic delays."""
    gen = TrafficGenerator(link_rate_gbps)
    delays, _ = gen.generate_trimodal(rho, num_packets)
    return delays


def generate_amsix_traffic(rho: float,
                           num_packets: int = 100000,
                           link_rate_gbps: float = 10) -> np.ndarray:
    """Generate AMS-IX-like traffic delays."""
    gen = TrafficGenerator(link_rate_gbps)
    delays, _ = gen.generate_amsix_like(rho, num_packets)
    return delays


def generate_sfmix_traffic(rho: float,
                           num_packets: int = 100000,
                           link_rate_gbps: float = 10) -> np.ndarray:
    """Generate SFM-IX-like traffic delays."""
    gen = TrafficGenerator(link_rate_gbps)
    delays, _ = gen.generate_sfmix_like(rho, num_packets)
    return delays
