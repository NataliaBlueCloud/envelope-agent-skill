# Self-Calibrating Envelope Agent

AI agent for learning M/M/1 envelope models from M/G/1 traffic data. Implements Algorithm 1 from the paper *"Upper bound latency percentiles for high-speed coherent pluggables"*.

## Overview

The Self-Calibrating Envelope Agent learns the mapping from real traffic load (ρ_real) to envelope model load (ρ_env) automatically from packet traces or delay samples.

```
Real Packet Traces ──► Algorithm 1 ──► Learned Model ──► Tight Delay Bounds
```

## Key Features

- **Algorithm 1 Implementation**: Finds minimum ρ_env that upper-bounds real M/G/1 delays
- **Adaptive Monitoring**: Continuous monitoring with auto-recalibration on traffic drift
- **Multiple Input Formats**: Delay samples, PCAP files, distribution parameters
- **Polynomial Learning**: Learns traffic-specific ρ_env = a + b·ρ + c·ρ² models
- **Intelligent Diagnostics**: Detects poor tail fit, low-load issues, high variance

## Installation

```bash
git clone https://github.com/NataliaBlueCloud/envelope-agent-skill.git
cd envelope-agent-skill
pip install numpy scipy

# Optional: for PCAP support
pip install scapy  # or dpkt
```

## Quick Start

### From Delay Samples
```python
from scripts.envelope_agent import fit_envelope

result = fit_envelope(
    delay_samples=delay_data_us,
    avg_packet_bytes=1750,
    link_rate_gbps=10
)

print(f"ρ_env: {result['rho_env']:.3f}")
print(f"P99 bound: {result['p99_bound_us']:.2f} μs")
```

### From PCAP File
```python
from scripts.envelope_agent import fit_envelope_from_pcap

result = fit_envelope_from_pcap(
    pcap_file='capture.pcap',
    link_rate_gbps=10
)
```

### Adaptive Monitoring
```python
from scripts.adaptive_monitor import AdaptiveEnvelopeMonitor

monitor = AdaptiveEnvelopeMonitor(
    link_rate_gbps=10,
    window_size=10000,
    drift_threshold_pct=15.0
)

for packet in traffic_stream:
    result = monitor.update(packet.delay_us, packet.size_bytes)
    if result and result.get('drift_detected'):
        print(f"Drift detected! New model fitted.")
```

## Paper Example

Run the publication-ready example:

```bash
python3 examples/paper_example.py
```

This demonstrates:
- 3,000 households → 10 Gb/s MAN link scenario
- 10,000 packet samples
- PDF estimation and envelope fitting
- 33% tighter bounds vs. static models

## Repository Structure

```
envelope-agent-skill/
├── SKILL.md                    # Skill documentation
├── scripts/
│   ├── envelope_agent.py       # Core Algorithm 1 implementation
│   ├── adaptive_monitor.py     # Continuous monitoring
│   ├── traffic_generator.py    # M/G/1 traffic simulator
│   └── pcap_parser.py          # PCAP file reader
├── examples/
│   ├── paper_example.py        # Publication example
│   ├── full_run_10k.py         # Complete pipeline demo
│   ├── pcap_demo.py            # PCAP processing
│   ├── adaptive_demo.py        # Adaptive monitoring
│   ├── all_methods_demo.py     # All input methods
│   └── demo.py                 # Basic usage
└── references/
    ├── algorithm.md            # Algorithm 1 specification
    └── models.md               # Pre-fitted polynomial models
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{koneva2025envelope,
  title={Upper bound latency percentiles for high-speed coherent pluggables: 
         An empirical model and its AI agent assistant},
  author={Koneva, Nataliia and S{\'a}nchez-Maci{\'a}n, Alfonso and 
          Hern{\'a}ndez, Jos{\'e} Alberto and others},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details.

## Related

- Original paper repository: [MG1-to-MM1-Envelope-Approximation](https://github.com/NataliaBlueCloud/MG1-to-MM1-Envelope-Approximation)
