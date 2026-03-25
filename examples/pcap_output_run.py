"""
Autonomous Monitor - PCAP Output Mode

Runs the autonomous monitor and saves captured data as actual PCAP files
that can be opened in Wireshark.
"""

import numpy as np
import sys
import os
from datetime import datetime

sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/envelope-agent')

from scripts.autonomous_monitor import AutonomousEnvelopeMonitor
from scripts.envelope_agent import EnvelopeAgent
from scripts.traffic_generator import TrafficGenerator

# Try to import scapy for PCAP writing
try:
    from scapy.all import Ether, IP, TCP, Raw, wrpcap, Packet
    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False
    print("Warning: scapy not installed. Installing...")
    os.system("pip install scapy -q")
    from scapy.all import Ether, IP, TCP, Raw, wrpcap, Packet

# Create output directory
output_dir = f"/home/ubuntu/envelope_pcap_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Configuration
CONFIG = {
    'interface': 'eth0',
    'window_size': 1000,  # 1k packets per cycle
    'error_threshold_pct': 15.0,
    'link_rate_gbps': 10,
    'max_cycles': 3,
    'use_live_capture': False  # Synthetic mode
}

print("="*70)
print(" AUTONOMOUS MONITOR - PCAP OUTPUT MODE")
print("="*70)
print(f"\nConfiguration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print(f"\nOutput files will be saved to: {output_dir}")
print(f"\nPCAP files can be opened in Wireshark!")

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
    
    # Step 1: Generate packets
    print("\n[STEP 1] Generating packets...")
    
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
    timestamps = np.cumsum(inter_arrivals) / 1e6
    
    print(f"  ✓ Generated {len(delays)} packets")
    print(f"  ✓ Target ρ: {rho:.4f}")
    
    # SAVE AS PCAP FILE
    pcap_file = f"{output_dir}/cycle_{cycle:02d}_capture.pcap"
    
    print(f"\n  [SAVING PCAP] Creating {pcap_file}...")
    
    # Create Scapy packets
    packets = []
    base_time = timestamps[0]
    
    for i in range(len(delays)):
        # Create a simple TCP packet with payload based on packet size
        payload_size = int(packet_sizes[i]) - 54  # Ethernet(14) + IP(20) + TCP(20)
        payload_size = max(0, payload_size)
        
        pkt = Ether()/IP(src="192.168.1.1", dst="192.168.1.2")/TCP()/Raw(load=b'X'*payload_size)
        
        # Set timestamp
        pkt.time = timestamps[i]
        
        packets.append(pkt)
    
    # Write PCAP file
    wrpcap(pcap_file, packets)
    
    file_size = os.path.getsize(pcap_file)
    print(f"  ✓ PCAP saved: {pcap_file}")
    print(f"  ✓ File size: {file_size:,} bytes")
    print(f"  ✓ Packets: {len(packets)}")
    
    # Also save CSV for reference
    csv_file = f"{output_dir}/cycle_{cycle:02d}_data.csv"
    with open(csv_file, 'w') as f:
        f.write("timestamp_sec,packet_size_bytes,delay_us\n")
        for i in range(len(delays)):
            f.write(f"{timestamps[i]:.6f},{packet_sizes[i]:.1f},{delays[i]:.4f}\n")
    print(f"  ✓ CSV saved: {csv_file}")
    
    # Continue with analysis...
    print("\n[STEP 2-7] Running envelope analysis...")
    
    avg_packet_bytes = np.mean(packet_sizes)
    service_time_us = agent.compute_service_time(avg_packet_bytes)
    avg_delay_real = np.mean(delays)
    rho_real = 1 - (service_time_us / avg_delay_real)
    rho_real = max(0.01, min(0.99, rho_real))
    
    # Fit envelope
    fit_result = agent.fit_envelope(
        delay_samples=delays,
        avg_packet_bytes=avg_packet_bytes,
        link_rate_gbps=CONFIG['link_rate_gbps']
    )
    
    rho_env_actual = fit_result['rho_env']
    
    # Predict using polynomial if available
    if len(all_data) >= 2:
        rho_real_vals = [d['rho_real'] for d in all_data]
        rho_env_vals = [d['rho_env_actual'] for d in all_data]
        coeffs = agent.fit_polynomial_mapping(
            np.array(rho_real_vals), np.array(rho_env_vals), degree=2
        )
        a, b, c = coeffs
        rho_env_predicted = a + b * rho_real + c * (rho_real ** 2)
        rho_env_predicted = min(0.99, rho_env_predicted)
        error_pct = abs(rho_env_predicted - rho_env_actual) / rho_env_actual * 100
        have_pred = True
    else:
        rho_env_predicted = None
        error_pct = 0.0
        have_pred = False
    
    # Check threshold
    action = 'normal'
    if have_pred and error_pct > CONFIG['error_threshold_pct']:
        action = 'polynomial_refit'
    
    # Percentiles
    rho_env_final = rho_env_predicted if have_pred else rho_env_actual
    percentiles = {}
    for p in [50, 90, 95, 99]:
        percentiles[f'p{p}'] = agent.compute_delay_percentile_mm1(
            rho_env_final, service_time_us, p / 100
        )
    
    # Save analysis results
    results_file = f"{output_dir}/cycle_{cycle:02d}_analysis.txt"
    with open(results_file, 'w') as f:
        f.write(f"Cycle {cycle} Analysis Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"PCAP File: cycle_{cycle:02d}_capture.pcap\n")
        f.write(f"CSV File: cycle_{cycle:02d}_data.csv\n\n")
        f.write(f"Input Data:\n")
        f.write(f"  Packets: {len(delays)}\n")
        f.write(f"  Avg packet size: {avg_packet_bytes:.2f} bytes\n")
        f.write(f"  Target ρ: {rho:.4f}\n\n")
        f.write(f"Step 2 - Calculated ρ_real: {rho_real:.4f}\n\n")
        f.write(f"Step 3 - Polynomial Prediction:\n")
        if have_pred:
            f.write(f"  Model: ρ_env = {a:.4f} + {b:.4f}·ρ_real + {c:.4f}·ρ_real²\n")
            f.write(f"  Predicted ρ_env: {rho_env_predicted:.4f}\n")
        else:
            f.write(f"  No prediction yet\n")
        f.write(f"\nStep 4 - Ground Truth:\n")
        f.write(f"  Actual ρ_env: {rho_env_actual:.4f}\n")
        f.write(f"  Fit quality: {fit_result['fit_quality']}\n\n")
        f.write(f"Step 5 - Error: {error_pct:.2f}%\n")
        f.write(f"Step 6 - Action: {action}\n\n")
        f.write(f"Step 7 - Percentiles (using ρ_env={rho_env_final:.4f}):\n")
        f.write(f"  P50: {percentiles['p50']:.2f} μs\n")
        f.write(f"  P90: {percentiles['p90']:.2f} μs\n")
        f.write(f"  P99: {percentiles['p99']:.2f} μs\n")
    
    print(f"  ✓ Analysis saved: {results_file}")
    
    # Store data
    all_data.append({
        'cycle': cycle,
        'rho_real': rho_real,
        'rho_env_actual': rho_env_actual,
        'rho_env_predicted': rho_env_predicted,
        'error_pct': error_pct,
        'action': action,
        'percentiles': percentiles
    })

# Final summary
print(f"\n{'='*70}")
print(" FINAL SUMMARY")
print(f"{'='*70}")

summary_file = f"{output_dir}/summary.txt"
with open(summary_file, 'w') as f:
    f.write("Autonomous Monitor - PCAP Run Summary\n")
    f.write("="*50 + "\n\n")
    f.write(f"Total cycles: {len(all_data)}\n")
    f.write(f"Total packets: {len(all_data) * CONFIG['window_size']}\n\n")
    f.write("PCAP Files Created:\n")
    for i in range(1, len(all_data) + 1):
        f.write(f"  - cycle_{i:02d}_capture.pcap\n")
    f.write(f"\nOpen in Wireshark: wireshark {output_dir}/cycle_01_capture.pcap\n\n")
    
    f.write("Results:\n")
    f.write("-"*50 + "\n")
    for d in all_data:
        f.write(f"\nCycle {d['cycle']}:\n")
        f.write(f"  ρ_real: {d['rho_real']:.4f}\n")
        f.write(f"  ρ_env: {d['rho_env_actual']:.4f}\n")
        if d['rho_env_predicted']:
            f.write(f"  predicted: {d['rho_env_predicted']:.4f}\n")
            f.write(f"  error: {d['error_pct']:.2f}%\n")
        f.write(f"  P99: {d['percentiles']['p99']:.2f} μs\n")

print(f"\n✓ All files saved to: {output_dir}")
print(f"✓ Summary: {summary_file}")
print(f"\nPCAP files created:")
for f in sorted(os.listdir(output_dir)):
    if f.endswith('.pcap'):
        size = os.path.getsize(f"{output_dir}/{f}")
        print(f"  📁 {f} ({size:,} bytes)")

print(f"\nTo open in Wireshark:")
print(f"  wireshark {output_dir}/cycle_01_capture.pcap")

print("\n" + "="*70)
print(" RUN COMPLETE - PCAP FILES READY")
print("="*70)
