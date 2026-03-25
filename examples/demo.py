"""
Example: Basic usage of the Self-Calibrating Envelope Agent
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/envelope-agent')

from scripts.envelope_agent import EnvelopeAgent, fit_envelope
from scripts.traffic_generator import generate_sfmix_traffic

print("="*60)
print("Self-Calibrating Envelope Agent - Demo")
print("="*60)

# Generate synthetic traffic (SFM-IX-like at 70% load)
print("\n1. Generating synthetic traffic...")
rho_real = 0.70
delays_us = generate_sfmix_traffic(rho=rho_real, num_packets=50000, link_rate_gbps=10)
print(f"   Generated {len(delays_us)} delay samples")
print(f"   Real load: {rho_real}")
print(f"   Mean delay: {np.mean(delays_us):.2f} μs")
print(f"   P99 delay: {np.percentile(delays_us, 99):.2f} μs")

# Fit envelope
print("\n2. Running Algorithm 1 to fit envelope...")
result = fit_envelope(
    delay_samples=delays_us,
    avg_packet_bytes=1750,  # SFM-IX mean
    link_rate_gbps=10
)

# Display results
print("\n" + "="*60)
print("ENVELOPE FIT RESULTS")
print("="*60)
print(f"\n📊 Load Mapping:")
print(f"   ρ_real (measured):     {result['rho_real']:.3f}")
print(f"   ρ_env (envelope):      {result['rho_env']:.3f}")
print(f"   Load factor:           {result['envelope_load_factor']:.2f}x")

print(f"\n⏱️  Delay Bounds:")
print(f"   Avg delay (real):      {result['avg_delay_real_us']:.2f} μs")
print(f"   Avg delay (envelope):  {result['avg_delay_env_us']:.2f} μs")
print(f"   P90 bound:             {result['p90_bound_us']:.2f} μs")
print(f"   P99 bound:             {result['p99_bound_us']:.2f} μs")

print(f"\n✅ Quality Metrics:")
print(f"   Fit quality:           {result['fit_quality']}")
print(f"   Confidence:            {result['confidence']:.1%}")
print(f"   Max violation:         {result['max_violation']:.1%}")

print(f"\n🔍 Diagnostics:")
print(f"   {result['diagnostics']['recommendation']}")

# Use pre-fitted model
print("\n" + "="*60)
print("PRE-FITTED MODEL COMPARISON")
print("="*60)

agent = EnvelopeAgent(link_rate_gbps=10)

for traffic_type in ['trimodal', 'amsix', 'sfmix']:
    prediction = agent.predict_from_model(rho_real, traffic_type)
    print(f"\n{traffic_type.upper()}:")
    print(f"   ρ_env = {prediction['rho_env']:.3f}")
    print(f"   Model: {prediction['polynomial']}")
