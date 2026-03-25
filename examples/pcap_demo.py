"""
Example: PCAP Processing Demo

Shows how to read real packet captures and fit envelope models.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/envelope-agent')

print("="*60)
print("PCAP Processing Demo")
print("="*60)

# First, let's generate a synthetic PCAP-like dataset
# (In real use, you'd have an actual .pcap file)

print("\n1️⃣  Generating synthetic PCAP data for demo...")
print("-"*60)

from scripts.traffic_generator import TrafficGenerator

gen = TrafficGenerator(10)

# Generate traffic
delays, service_times = gen.generate_sfmix_like(
    rho=0.6, num_packets=5000, seed=42
)

# Create packet trace (timestamps + sizes)
arrival_rate = 0.6 / np.mean(service_times)
inter_arrivals = np.random.exponential(1/arrival_rate, 5000)
arrival_times = np.cumsum(inter_arrivals) / 1e6  # seconds

# SFM-IX-like packet sizes
sigma = np.sqrt(np.log(1 + (2062.69/1750.41)**2))
mu = np.log(1750.41) - sigma**2/2
packet_sizes = np.random.lognormal(mu, sigma, 5000)
packet_sizes = np.clip(packet_sizes, 64, 9000)

print(f"   Generated {len(arrival_times)} packets")
print(f"   Duration: {arrival_times[-1] - arrival_times[0]:.3f} seconds")
print(f"   Avg packet: {np.mean(packet_sizes):.0f} bytes")

# Save as mock PCAP data (in real use, you'd have a .pcap file)
print("\n   (In real use: tcpdump -i eth0 -w capture.pcap -c 5000)")

# ========================================================================
print("\n2️⃣  Processing with envelope agent...")
print("-"*60)

from scripts.envelope_agent import fit_envelope_from_trace

# Process the trace (same as processing real PCAP)
result = fit_envelope_from_trace(
    arrival_times_s=arrival_times,
    packet_sizes_bytes=packet_sizes,
    link_rate_gbps=10
)

print(f"   ✓ ρ_real: {result['rho_real']:.3f}")
print(f"   ✓ ρ_env:  {result['rho_env']:.3f}")
print(f"   ✓ Avg delay (real): {result['avg_delay_real_us']:.2f} μs")
print(f"   ✓ Avg delay (env):  {result['avg_delay_env_us']:.2f} μs")
print(f"   ✓ P90 bound: {result['p90_bound_us']:.2f} μs")
print(f"   ✓ P99 bound: {result['p99_bound_us']:.2f} μs")
print(f"   ✓ Fit quality: {result['fit_quality']}")

# ========================================================================
print("\n3️⃣  Using PCAP Parser (if scapy/dpkt installed)")
print("-"*60)

try:
    from scripts.pcap_parser import summarize_pcap
    
    summary = summarize_pcap(arrival_times, packet_sizes)
    
    print(f"   PCAP Summary:")
    print(f"      Total packets: {summary['num_packets']}")
    print(f"      Duration: {summary['duration_s']:.3f} s")
    print(f"      Total bytes: {summary['total_bytes']:,}")
    print(f"      Avg throughput: {summary['avg_throughput_mbps']:.2f} Mbps")
    print(f"      SCV (C²): {summary['scv']:.2f}")
    print(f"      Arrival rate: {summary['inter_arrival']['arrival_rate_pps']:.0f} pps")
    
except ImportError as e:
    print(f"   Note: {e}")
    print("   Install with: pip install scapy")

# ========================================================================
print("\n4️⃣  Real-World Usage Example")
print("-"*60)
print("""
   # Capture traffic
   $ tcpdump -i eth0 -w capture.pcap -c 10000
   
   # Process with Python
   from scripts.envelope_agent import fit_envelope_from_pcap
   
   result = fit_envelope_from_pcap(
       pcap_file='capture.pcap',
       link_rate_gbps=10,
       min_packet_size=100  # Filter out ACKs
   )
   
   print(f"Processed {result['pcap_summary']['num_packets']} packets")
   print(f"P99 latency bound: {result['p99_bound_us']:.2f} μs")
""")

print("\n" + "="*60)
print("PCAP processing ready!")
print("Install dependency: pip install scapy")
print("="*60)
