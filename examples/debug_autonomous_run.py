"""
Autonomous Monitor - Debug Mode

Runs the autonomous monitor with full step-by-step output and saves all captured data.
"""

import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/envelope-agent')

from scripts.autonomous_monitor import AutonomousEnvelopeMonitor
from scripts.envelope_agent import EnvelopeAgent
from scripts.traffic_generator import TrafficGenerator

# Create output directory
output_dir = f"/home/ubuntu/envelope_monitor_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Configuration
CONFIG = {
    'interface': 'eth0',
    'window_size': 1000,  # Smaller for demo (1k instead of 10k)
    'error_threshold_pct': 15.0,
    'link_rate_gbps': 10,
    'max_cycles': 3,
    'use_live_capture': False  # Synthetic mode
}

print("="*70)
print(" AUTONOMOUS MONITOR - DEBUG MODE")
print("="*70)
print(f"\nConfiguration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print(f"\nOutput files will be saved to: {output_dir}")

# Initialize
agent = EnvelopeAgent(CONFIG['link_rate_gbps'])
gen = TrafficGenerator(CONFIG['link_rate_gbps'])

# Store all data
all_data = []

# Run cycles
for cycle in range(1, CONFIG['max_cycles'] + 1):
    print(f"\n{'='*70}")
    print(f" CYCLE {cycle}")
    print(f"{'='*70}")
    
    # Step 1: Generate/Capture packets
    print("\n[STEP 1] Capturing packets...")
    
    # Generate synthetic traffic
    rho = 0.5 + np.random.uniform(-0.1, 0.1)
    delays, service_times = gen.generate_sfmix_like(
        rho=rho,
        num_packets=CONFIG['window_size'],
        seed=cycle * 100
    )
    
    # Create packet sizes
    sigma = np.sqrt(np.log(1 + (2062.69/1750.41)**2))
    mu = np.log(1750.41) - sigma**2/2
    packet_sizes = np.random.lognormal(mu, sigma, CONFIG['window_size'])
    packet_sizes = np.clip(packet_sizes, 64, 9000)
    
    # Create timestamps
    arrival_rate = rho / np.mean(service_times)
    inter_arrivals = np.random.exponential(1/arrival_rate, CONFIG['window_size'])
    timestamps = np.cumsum(inter_arrivals) / 1e6  # Convert to seconds
    
    print(f"  ✓ Generated {len(delays)} packets")
    print(f"  ✓ ρ_real (target): {rho:.4f}")
    print(f"  ✓ Avg packet size: {np.mean(packet_sizes):.1f} bytes")
    
    # Save raw capture data
    capture_file = f"{output_dir}/cycle_{cycle:02d}_capture.txt"
    with open(capture_file, 'w') as f:
        f.write("timestamp_sec,packet_size_bytes,delay_us\n")
        for i in range(min(100, len(delays))):  # Save first 100 for brevity
            f.write(f"{timestamps[i]:.6f},{packet_sizes[i]:.1f},{delays[i]:.4f}\n")
        f.write(f"... ({len(delays)} total packets)\n")
    print(f"  ✓ Saved capture data: {capture_file}")
    
    # Step 2: Calculate ρ_real from actual data
    print("\n[STEP 2] Calculating actual ρ_real from captured data...")
    
    avg_packet_bytes = np.mean(packet_sizes)
    service_time_us = agent.compute_service_time(avg_packet_bytes)
    avg_delay_real = np.mean(delays)
    rho_real = 1 - (service_time_us / avg_delay_real)
    rho_real = max(0.01, min(0.99, rho_real))
    
    print(f"  ✓ Service time: {service_time_us:.4f} μs")
    print(f"  ✓ Avg delay: {avg_delay_real:.4f} μs")
    print(f"  ✓ Calculated ρ_real: {rho_real:.4f}")
    
    # Step 3: Predict using saved polynomial (if exists)
    print("\n[STEP 3] Predicting using saved polynomial model...")
    
    if len(all_data) >= 2:
        # Fit polynomial from previous data
        rho_real_vals = [d['rho_real'] for d in all_data]
        rho_env_vals = [d['rho_env_actual'] for d in all_data]
        
        coeffs = agent.fit_polynomial_mapping(
            np.array(rho_real_vals),
            np.array(rho_env_vals),
            degree=2
        )
        a, b, c = coeffs
        rho_env_predicted = a + b * rho_real + c * (rho_real ** 2)
        rho_env_predicted = min(0.99, rho_env_predicted)
        
        print(f"  ✓ Saved model: ρ_env = {a:.4f} + {b:.4f}·ρ_real + {c:.4f}·ρ_real²")
        print(f"  ✓ Predicted ρ_env: {rho_env_predicted:.4f}")
        have_prediction = True
    else:
        print(f"  ! No saved model yet (need at least 2 previous cycles)")
        rho_env_predicted = None
        have_prediction = False
    
    # Step 4: Fit envelope to get actual ρ_env (ground truth)
    print("\n[STEP 4] Fitting envelope to get actual ρ_env (ground truth)...")
    
    fit_result = agent.fit_envelope(
        delay_samples=delays,
        avg_packet_bytes=avg_packet_bytes,
        link_rate_gbps=CONFIG['link_rate_gbps']
    )
    
    rho_env_actual = fit_result['rho_env']
    
    print(f"  ✓ Actual ρ_env: {rho_env_actual:.4f}")
    print(f"  ✓ Envelope load factor: {fit_result['envelope_load_factor']:.2f}x")
    print(f"  ✓ Fit quality: {fit_result['fit_quality']}")
    
    # Step 5: Calculate error
    print("\n[STEP 5] Calculating prediction error...")
    
    if have_prediction:
        error_pct = abs(rho_env_predicted - rho_env_actual) / rho_env_actual * 100
        print(f"  ✓ Predicted ρ_env: {rho_env_predicted:.4f}")
        print(f"  ✓ Actual ρ_env:    {rho_env_actual:.4f}")
        print(f"  ✓ Error:           {error_pct:.2f}%")
    else:
        error_pct = 0.0
        print(f"  ! Cannot calculate error (no prediction)")
    
    # Step 6: Check threshold and refit if needed
    print("\n[STEP 6] Checking error threshold...")
    
    action = 'normal'
    if have_prediction and error_pct > CONFIG['error_threshold_pct']:
        print(f"  ⚠️  Error {error_pct:.1f}% > threshold {CONFIG['error_threshold_pct']}%")
        print(f"  → Running polynomial fitting...")
        
        # Refit polynomial
        rho_real_vals = [d['rho_real'] for d in all_data] + [rho_real]
        rho_env_vals = [d['rho_env_actual'] for d in all_data] + [rho_env_actual]
        
        coeffs = agent.fit_polynomial_mapping(
            np.array(rho_real_vals),
            np.array(rho_env_vals),
            degree=2
        )
        a, b, c = coeffs
        
        # Calculate R²
        predicted = a + b * np.array(rho_real_vals) + c * np.array(rho_real_vals)**2
        ss_res = np.sum((np.array(rho_env_vals) - predicted) ** 2)
        ss_tot = np.sum((np.array(rho_env_vals) - np.mean(rho_env_vals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"  ✓ New model: ρ_env = {a:.4f} + {b:.4f}·ρ_real + {c:.4f}·ρ_real²")
        print(f"  ✓ R² = {r_squared:.4f}")
        
        # Re-predict with new model
        rho_env_predicted = a + b * rho_real + c * (rho_real ** 2)
        rho_env_predicted = min(0.99, rho_env_predicted)
        error_pct = abs(rho_env_predicted - rho_env_actual) / rho_env_actual * 100
        
        print(f"  ✓ Re-predicted ρ_env: {rho_env_predicted:.4f}")
        print(f"  ✓ New error: {error_pct:.2f}%")
        action = 'polynomial_refit'
    else:
        if have_prediction:
            print(f"  ✓ Error {error_pct:.1f}% within threshold")
        else:
            print(f"  ✓ First cycle - building model")
        action = 'normal'
    
    # Step 7: Estimate percentiles
    print("\n[STEP 7] Estimating delay percentiles...")
    
    # Use predicted (or actual if no prediction) rho_env
    rho_env_final = rho_env_predicted if have_prediction else rho_env_actual
    
    percentiles = {}
    for p in [50, 90, 95, 99]:
        percentiles[f'p{p}'] = agent.compute_delay_percentile_mm1(
            rho_env_final, service_time_us, p / 100
        )
    
    print(f"  ✓ Using ρ_env = {rho_env_final:.4f}")
    print(f"  ✓ P50: {percentiles['p50']:.2f} μs")
    print(f"  ✓ P90: {percentiles['p90']:.2f} μs")
    print(f"  ✓ P95: {percentiles['p95']:.2f} μs")
    print(f"  ✓ P99: {percentiles['p99']:.2f} μs")
    
    # Save cycle results
    results_file = f"{output_dir}/cycle_{cycle:02d}_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Cycle {cycle} Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Input:\n")
        f.write(f"  Packets captured: {len(delays)}\n")
        f.write(f"  Avg packet size: {avg_packet_bytes:.2f} bytes\n")
        f.write(f"\nStep 2 - Calculated ρ_real:\n")
        f.write(f"  ρ_real: {rho_real:.4f}\n")
        f.write(f"\nStep 3 - Polynomial Prediction:\n")
        if have_prediction:
            f.write(f"  Predicted ρ_env: {rho_env_predicted:.4f}\n")
        else:
            f.write(f"  No prediction (first cycle)\n")
        f.write(f"\nStep 4 - Ground Truth (Envelope Fit):\n")
        f.write(f"  Actual ρ_env: {rho_env_actual:.4f}\n")
        f.write(f"  Fit quality: {fit_result['fit_quality']}\n")
        f.write(f"\nStep 5 - Error Analysis:\n")
        f.write(f"  Error: {error_pct:.2f}%\n")
        f.write(f"  Threshold: {CONFIG['error_threshold_pct']}%\n")
        f.write(f"  Action: {action}\n")
        f.write(f"\nStep 7 - Percentile Estimates:\n")
        f.write(f"  P50: {percentiles['p50']:.2f} μs\n")
        f.write(f"  P90: {percentiles['p90']:.2f} μs\n")
        f.write(f"  P95: {percentiles['p95']:.2f} μs\n")
        f.write(f"  P99: {percentiles['p99']:.2f} μs\n")
    
    print(f"\n  ✓ Saved results: {results_file}")
    
    # Store data for next cycles
    all_data.append({
        'cycle': cycle,
        'rho_real': rho_real,
        'rho_env_actual': rho_env_actual,
        'rho_env_predicted': rho_env_predicted,
        'error_pct': error_pct,
        'action': action,
        'percentiles': percentiles
    })
    
    print(f"\n{'='*70}")
    print(f" CYCLE {cycle} COMPLETE")
    print(f"{'='*70}")

# Save summary
print(f"\n{'='*70}")
print(" FINAL SUMMARY")
print(f"{'='*70}")

summary_file = f"{output_dir}/summary.txt"
with open(summary_file, 'w') as f:
    f.write("Autonomous Monitor Run Summary\n")
    f.write("="*50 + "\n\n")
    f.write(f"Total cycles: {len(all_data)}\n")
    f.write(f"Total packets: {len(all_data) * CONFIG['window_size']}\n\n")
    
    f.write("Cycle-by-Cycle Results:\n")
    f.write("-"*50 + "\n")
    for d in all_data:
        f.write(f"\nCycle {d['cycle']}:\n")
        f.write(f"  ρ_real: {d['rho_real']:.4f}\n")
        f.write(f"  ρ_env_actual: {d['rho_env_actual']:.4f}\n")
        if d['rho_env_predicted']:
            f.write(f"  ρ_env_predicted: {d['rho_env_predicted']:.4f}\n")
            f.write(f"  error: {d['error_pct']:.2f}%\n")
        f.write(f"  action: {d['action']}\n")
        f.write(f"  P99: {d['percentiles']['p99']:.2f} μs\n")

print(f"\n✓ All data saved to: {output_dir}")
print(f"✓ Summary file: {summary_file}")
print(f"\nFiles created:")
for f in os.listdir(output_dir):
    print(f"  - {f}")

print("\n" + "="*70)
print(" RUN COMPLETE")
print("="*70)
