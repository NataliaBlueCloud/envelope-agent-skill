"""
Example: Adaptive Envelope Monitor - Full Demo

Demonstrates continuous monitoring with auto-refit on drift detection.
Shows both sliding window updates and batch learning.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/envelope-agent')

from scripts.adaptive_monitor import AdaptiveEnvelopeMonitor
from scripts.traffic_generator import TrafficGenerator

print("="*60)
print("Adaptive Envelope Monitor - Full Demo")
print("="*60)

# Create monitor with settings tuned for demo
tr = AdaptiveEnvelopeMonitor(
    link_rate_gbps=10,
    window_size=3000,
    drift_threshold_pct=15.0,
    scv_check_interval=500
)

gen = TrafficGenerator(10)

print("\n📊 PHASE 1: SFM-IX-like Traffic (C²≈1.39)")
print("   Feeding 4000 packets...")

# Generate batch of SFM-IX traffic
delays1, _ = gen.generate_sfmix_like(rho=0.6, num_packets=4000, seed=42)
packet_sizes1 = np.random.lognormal(7.47, 0.84, 4000)  # SFM-IX distribution
packet_sizes1 = np.clip(packet_sizes1, 64, 9000)

for i in range(4000):
    result = tr.update(delays1[i], packet_sizes1[i])
    if result and result.get('model'):
        print(f"   ✓ Model fitted at packet {i}")
        print(f"     ρ_real={result['model']['rho_real']:.3f}, "
              f"ρ_env={result['model']['rho_env']:.3f}")

print("\n📊 PHASE 2: Tri-Modal Traffic (C²≈1.58)")
print("   Distribution changed! Feeding 4000 packets...")

# Generate tri-modal traffic (different distribution)
delays2, _ = gen.generate_trimodal(rho=0.6, num_packets=4000, seed=99)
# Tri-modal packet sizes: 40B (58%), 576B (33%), 1500B (8%)
sizes = [40, 576, 1500]
probs = [7/12, 4/12, 1/12]
packet_sizes2 = np.random.choice(sizes, size=4000, p=probs)

for i in range(4000):
    result = tr.update(delays2[i], packet_sizes2[i])
    
    if result:
        if result.get('drift_detected'):
            print(f"   🚨 DRIFT at packet {4000+i}!")
            print(f"      SCV changed by {result['scv_change_pct']:.1f}%")
            if result.get('refit_triggered'):
                print(f"      ✓ Auto-refit: ρ_env={result['model']['rho_env']:.3f}")

print("\n📊 Status Summary")
print("-"*50)
status = tr.get_status()
print(f"   Total packets processed: {status['packet_count']}")
print(f"   Drift events detected: {status['drift_count']}")
print(f"   Model refits: {status['refit_count']}")

print("\n📊 Learning Custom Polynomial Model")
print("-"*50)
print("   Sweeping loads: 0.3, 0.5, 0.7, 0.9")

model = tr.learn_polynomial_model(
    loads_to_test=[0.3, 0.5, 0.7, 0.9],
    samples_per_load=3000
)

print(f"\n   📈 Learned model:")
print(f"      {model['polynomial']}")
print(f"      R² = {model['r_squared']:.4f}")

# Demonstrate prediction
pred = tr.predict_with_current_model(rho_real=0.6)
if pred:
    print(f"\n   🔮 Prediction for ρ_real=0.6:")
    print(f"      ρ_env = {pred['rho_env']:.3f}")

print("\n" + "="*60)
print("Demo complete! Adaptive monitor ready for deployment.")
print("="*60)
