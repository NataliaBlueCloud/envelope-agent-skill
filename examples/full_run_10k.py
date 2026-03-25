"""
Full Envelope Agent Run: 10k Samples + PDF Estimation + Polynomial Correction

This script:
1. Generates 10,000 packet samples
2. Estimates PDF of delays
3. Runs Algorithm 1 to find envelope
4. Fits polynomial regression
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/envelope-agent')

from scripts.envelope_agent import EnvelopeAgent, fit_envelope
from scripts.traffic_generator import TrafficGenerator
import json

print("="*70)
print(" ENVELOPE AGENT FULL RUN")
print(" 10k Samples | PDF Estimation | Polynomial Correction")
print("="*70)

# Initialize
agent = EnvelopeAgent(link_rate_gbps=10)
gen = TrafficGenerator(link_rate_gbps=10)

# ============================================================================
print("\n📥 STEP 1: Gathering 10,000 Packet Samples")
print("-"*70)

# Generate SFM-IX-like traffic
delays_us, service_times = gen.generate_sfmix_like(
    rho=0.7, 
    num_packets=10000, 
    seed=42
)

print(f"   ✓ Generated {len(delays_us):,} delay samples")
print(f"   ✓ Mean delay: {np.mean(delays_us):.3f} μs")
print(f"   ✓ Std delay: {np.std(delays_us):.3f} μs")
print(f"   ✓ Min delay: {np.min(delays_us):.3f} μs")
print(f"   ✓ Max delay: {np.max(delays_us):.3f} μs")

# ============================================================================
print("\n📊 STEP 2: Estimating PDF of Delay Distribution")
print("-"*70)

# Compute histogram/PDF
hist, bin_edges = np.histogram(delays_us, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Print PDF data
print("   Delay PDF (first 10 bins):")
for i in range(min(10, len(hist))):
    print(f"      [{bin_edges[i]:6.2f}, {bin_edges[i+1]:6.2f}] μs: {hist[i]:.6f}")

# Key percentiles
percentiles = [50, 90, 95, 99, 99.9]
print("\n   Delay Percentiles:")
for p in percentiles:
    val = np.percentile(delays_us, p)
    print(f"      P{p:5.1f}: {val:7.2f} μs")

# ============================================================================
print("\n🔍 STEP 3: Running Algorithm 1 (Find Envelope)")
print("-"*70)

# Fit envelope using Algorithm 1
result = agent.fit_envelope(
    delay_samples=delays_us,
    avg_packet_bytes=1750.41,  # SFM-IX mean
    link_rate_gbps=10
)

print(f"   ✓ ρ_real (measured):    {result['rho_real']:.4f}")
print(f"   ✓ ρ_env (envelope):     {result['rho_env']:.4f}")
print(f"   ✓ Envelope load factor: {result['envelope_load_factor']:.2f}x")
print(f"   ✓ Avg delay (real):     {result['avg_delay_real_us']:.4f} μs")
print(f"   ✓ Avg delay (envelope): {result['avg_delay_env_us']:.4f} μs")
print(f"   ✓ P90 bound:            {result['p90_bound_us']:.4f} μs")
print(f"   ✓ P99 bound:            {result['p99_bound_us']:.4f} μs")
print(f"   ✓ Fit quality:          {result['fit_quality']}")
print(f"   ✓ Confidence:           {result['confidence']:.2%}")

# Show diagnostics
print(f"\n   📝 Diagnostics:")
print(f"      {result['diagnostics']['recommendation']}")

# ============================================================================
print("\n📈 STEP 4: Polynomial Regression Correction")
print("-"*70)
print("   Fitting ρ_env = a + b·ρ_real + c·ρ_real² across multiple loads...")

# Generate data at multiple loads for polynomial fitting
loads_to_test = np.arange(0.2, 0.91, 0.1)  # 0.2, 0.3, ..., 0.9
rho_real_vals = []
rho_env_vals = []

print(f"\n   Testing {len(loads_to_test)} load points:")

for load in loads_to_test:
    # Generate traffic at this load
    delays, _ = gen.generate_sfmix_like(rho=load, num_packets=5000, seed=int(load*100))
    
    # Fit envelope
    res = agent.fit_envelope(delays, avg_packet_bytes=1750.41, link_rate_gbps=10)
    
    rho_real_vals.append(res['rho_real'])
    rho_env_vals.append(res['rho_env'])
    
    print(f"      ρ_real={res['rho_real']:.3f} → ρ_env={res['rho_env']:.3f}")

# Fit polynomial: ρ_env = a + b·ρ_real + c·ρ_real²
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

print(f"\n   📊 Fitted Polynomial Model:")
print(f"      ρ_env = {a:.4f} + {b:.4f}·ρ_real + {c:.4f}·ρ_real²")
print(f"      R² = {r_squared:.6f}")

# ============================================================================
print("\n📊 STEP 5: Model Comparison")
print("-"*70)

print("   Comparison with Paper's Models:")
print(f"   {'Model':<20} {'Polynomial':<40} {'R²':<10}")
print(f"   {'-'*70}")

paper_models = {
    'SFM-IX (paper)': (0.50, 0.16, 0.34),
    'AMS-IX (paper)': (0.43, 0.13, 0.47),
    'Tri-modal (paper)': (0.49, 0.13, 0.39),
    'Our Fit': (a, b, c)
}

for name, (ca, cb, cc) in paper_models.items():
    # Compute R² for this model
    pred = ca + cb * np.array(rho_real_vals) + cc * np.array(rho_real_vals)**2
    ss_res_model = np.sum((np.array(rho_env_vals) - pred) ** 2)
    r2_model = 1 - (ss_res_model / ss_tot) if ss_tot > 0 else 0
    
    poly_str = f"{ca:.2f} + {cb:.2f}·ρ + {cc:.2f}·ρ²"
    print(f"   {name:<20} {poly_str:<40} {r2_model:.4f}")

# ============================================================================
print("\n🎯 STEP 6: Validation")
print("-"*70)

# Test prediction at unseen load
test_load = 0.65
test_delays, _ = gen.generate_sfmix_like(rho=test_load, num_packets=5000, seed=999)
test_result = agent.fit_envelope(test_delays, avg_packet_bytes=1750.41, link_rate_gbps=10)

# Predict using our polynomial
predicted_rho_env = a + b * test_result['rho_real'] + c * (test_result['rho_real'] ** 2)
predicted_rho_env = min(0.99, predicted_rho_env)

print(f"   Test at ρ_real = {test_result['rho_real']:.3f}:")
print(f"      Actual ρ_env (Algorithm 1):  {test_result['rho_env']:.4f}")
print(f"      Predicted ρ_env (polynomial): {predicted_rho_env:.4f}")
print(f"      Prediction error: {abs(predicted_rho_env - test_result['rho_env']):.4f}")

# ============================================================================
print("\n" + "="*70)
print(" RUN COMPLETE")
print("="*70)

# Save results
final_result = {
    'samples_collected': len(delays_us),
    'single_fit': {
        'rho_real': result['rho_real'],
        'rho_env': result['rho_env'],
        'p99_bound_us': result['p99_bound_us']
    },
    'polynomial_model': {
        'a': a,
        'b': b,
        'c': c,
        'formula': f"ρ_env = {a:.4f} + {b:.4f}·ρ_real + {c:.4f}·ρ_real²",
        'r_squared': r_squared
    },
    'training_data': {
        'rho_real_values': [float(x) for x in rho_real_vals],
        'rho_env_values': [float(x) for x in rho_env_vals]
    }
}

print("\n📄 Final Result:")
print(json.dumps(final_result, indent=2))

print("\n✅ All steps completed successfully!")
