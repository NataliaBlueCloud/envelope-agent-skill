"""
Example: All Input Methods Demo

Shows how to use the envelope agent with:
1. Pre-computed delay samples
2. Raw packet traces (timestamps + sizes)
3. Distribution parameters only
4. Multiple traces for polynomial learning
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/envelope-agent')

from scripts.envelope_agent import (
    fit_envelope,
    fit_envelope_from_trace,
    fit_envelope_from_distribution,
    fit_polynomial_from_multiple_traces
)
from scripts.traffic_generator import TrafficGenerator

print("="*60)
print("Envelope Agent - All Input Methods Demo")
print("="*60)

# Setup
gen = TrafficGenerator(10)

# ========================================================================
print("\n1️⃣  METHOD 1: Pre-computed Delay Samples")
print("-"*60)
print("   Use when you already have delay measurements")

delays, _ = gen.generate_sfmix_like(rho=0.6, num_packets=10000, seed=42)

result1 = fit_envelope(
    delay_samples=delays,
    avg_packet_bytes=1750,
    link_rate_gbps=10
)

print(f"   ✓ ρ_real: {result1['rho_real']:.3f}")
print(f"   ✓ ρ_env:  {result1['rho_env']:.3f}")
print(f"   ✓ P99:    {result1['p99_bound_us']:.2f} μs")

# ========================================================================
print("\n2️⃣  METHOD 2: Raw Packet Trace (Timestamps + Sizes)")
print("-"*60)
print("   Use when you have packet capture (PCAP) data")

# Simulate a packet trace
delays, service_times = gen.generate_sfmix_like(rho=0.6, num_packets=10000, seed=123)

# Convert to arrival times + packet sizes (like real PCAP)
arrival_rate = 0.6 / np.mean(service_times)  # packets per μs
inter_arrivals = np.random.exponential(1/arrival_rate, 10000)
arrival_times = np.cumsum(inter_arrivals) / 1e6  # Convert to seconds

# Packet sizes from SFM-IX-like distribution
sigma = np.sqrt(np.log(1 + (2062.69/1750.41)**2))
mu = np.log(1750.41) - sigma**2/2
packet_sizes = np.random.lognormal(mu, sigma, 10000)
packet_sizes = np.clip(packet_sizes, 64, 9000)

result2 = fit_envelope_from_trace(
    arrival_times_s=arrival_times,
    packet_sizes_bytes=packet_sizes,
    link_rate_gbps=10
)

print(f"   ✓ Input: {len(arrival_times)} packets")
print(f"   ✓ ρ_real: {result2['rho_real']:.3f}")
print(f"   ✓ ρ_env:  {result2['rho_env']:.3f}")
print(f"   ✓ P99:    {result2['p99_bound_us']:.2f} μs")

# ========================================================================
print("\n3️⃣  METHOD 3: Distribution Parameters Only")
print("-"*60)
print("   Use when you don't have real traces, just statistics")

result3 = fit_envelope_from_distribution(
    mean_packet_bytes=1750,
    std_packet_bytes=2063,
    link_rate_gbps=10,
    target_load=0.6,
    distribution='lognormal',
    num_packets=10000
)

print(f"   ✓ Synthetic traffic: {result3['synthetic_traffic']['distribution']}")
print(f"   ✓ Target load: {result3['synthetic_traffic']['target_load']}")
print(f"   ✓ Actual SCV: {result3['synthetic_traffic']['actual_scv']:.2f}")
print(f"   ✓ ρ_real: {result3['rho_real']:.3f}")
print(f"   ✓ ρ_env:  {result3['rho_env']:.3f}")

# ========================================================================
print("\n4️⃣  METHOD 4: Learn Polynomial from Multiple Traces")
print("-"*60)
print("   Matches your R code: fit envelopes at different loads,")
print("   then fit polynomial ρ_env = a + b·ρ + c·ρ²")

# Generate traces at 3 different loads
traces = []
for load, label in [(0.3, 'low'), (0.5, 'med'), (0.7, 'high')]:
    delays, service_times = gen.generate_sfmix_like(
        rho=load, num_packets=5000, seed=int(load*100)
    )
    
    # Create arrival times
    arrival_rate = load / np.mean(service_times)
    inter_arrivals = np.random.exponential(1/arrival_rate, 5000)
    arrival_times = np.cumsum(inter_arrivals) / 1e6
    
    # Create packet sizes
    packet_sizes = np.random.lognormal(mu, sigma, 5000)
    packet_sizes = np.clip(packet_sizes, 64, 9000)
    
    traces.append({
        'arrival_times': arrival_times,
        'packet_sizes': packet_sizes,
        'label': f'load_{load}'
    })

# Learn polynomial model
model = fit_polynomial_from_multiple_traces(traces, link_rate_gbps=10)

print(f"\n   📈 Learned Polynomial Model:")
print(f"      {model['polynomial']}")
print(f"      R² = {model['r_squared']:.4f}")

# Compare with paper's pre-fitted model
print(f"\n   📊 Comparison with Paper's SFM-IX Model:")
print(f"      Paper: ρ_env = 0.50 + 0.16·ρ + 0.34·ρ²")
print(f"      Ours:  {model['polynomial']}")

# ========================================================================
print("\n" + "="*60)
print("All methods working! ✓")
print("="*60)
