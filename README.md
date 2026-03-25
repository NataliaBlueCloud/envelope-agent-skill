# Self-Calibrating Envelope Agent

**An [OpenClaw](https://github.com/openclaw/openclaw) Agent Skill for AI-Driven Network Queuing Analysis**

This agent skill implements an autonomous system that learns M/M/1 envelope models from real M/G/1 network traffic. It captures packet traces, fits tight delay bounds, and automatically adapts its polynomial model when prediction errors exceed a threshold — all without manual tuning.

Based on the research paper: *"Upper bound latency percentiles for high-speed coherent pluggables: An empirical model and its AI agent assistant"* by Koneva et al.

---

## What It Does

The Self-Calibrating Envelope Agent solves the problem of loose delay bounds in network analysis. Instead of using static polynomial models (which can be 30-60% conservative), this agent:

1. **Captures live packets** (10,000 samples per window)
2. **Predicts delay bounds** using a learned polynomial model: ρ_env = a + b·ρ_real + c·ρ_real²
3. **Validates predictions** against actual envelope fits
4. **Adapts automatically** — refits the polynomial when error > 15%
5. **Outputs tight percentiles** (P50, P90, P99) with confidence metrics

### Key Results

- **33% tighter bounds** vs. static SFM-IX model
- **R² > 0.99** polynomial fit quality
- **Zero manual tuning** — fully autonomous operation
- **Continuous monitoring** — runs forever, adapting to traffic changes

---

## How It Works (Autonomous Mode)

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Capture 10k     │────▶│ Predict ρ_env│────▶│ Compare with    │
│ Packets         │     │ (Polynomial) │     │ Actual Fit      │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                        │
                           ┌────────────────────────────┘
                           │ Error > 15%?
                           ▼
              ┌────────────────────┐
              │ YES: Refit Model   │────▶ Estimate P50/P90/P99
              │ NO: Continue       │
              └────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/NataliaBlueCloud/envelope-agent-skill.git
cd envelope-agent-skill
pip install numpy scipy

# Optional: for PCAP support
pip install scapy  # or dpkt

# Optional: for live capture
sudo apt-get install tcpdump  # Linux
brew install tcpdump          # macOS
```

---

## Quick Start

### 1. One-Shot Analysis from Delay Samples

```python
from scripts.envelope_agent import fit_envelope

result = fit_envelope(
    delay_samples=delay_data_us,  # Your measured delays
    avg_packet_bytes=1750,         # Mean packet size
    link_rate_gbps=10
)

print(f"ρ_env: {result['rho_env']:.3f}")
print(f"P99 bound: {result['p99_bound_us']:.2f} μs")
```

### 2. Live Capture with tcpdump

```python
from scripts.envelope_agent import capture_and_analyze_tcpdump

# Capture 30 seconds on eth0 and analyze
result = capture_and_analyze_tcpdump(
    interface='eth0',
    duration_seconds=30,
    link_rate_gbps=10
)

print(f"Captured: {result['capture_info']['packets_captured']} packets")
print(f"P99 delay bound: {result['p99_bound_us']:.2f} μs")
```

### 3. Autonomous Permanent Monitoring

```python
from scripts.autonomous_monitor import run_autonomous_monitor

# Run forever — captures, predicts, refits automatically
run_autonomous_monitor(
    interface='eth0',
    window_size=10000,          # 10k samples per cycle
    error_threshold_pct=15.0,   # Refit if error > 15%
    link_rate_gbps=10,
    max_cycles=None             # None = run forever
)
```

### 4. From PCAP File

```python
from scripts.envelope_agent import fit_envelope_from_pcap

result = fit_envelope_from_pcap(
    pcap_file='network_capture.pcap',
    link_rate_gbps=10
)

print(f"Packets: {result['pcap_summary']['num_packets']}")
print(f"P99: {result['p99_bound_us']:.2f} μs")
```

---

## Paper Example

Run the publication-ready demonstration:

```bash
python3 examples/paper_example.py
```

This replicates the paper's scenario:
- 3,000 households aggregated at Access Central Office
- 10 Gb/s fiber link to Metropolitan Area Network
- 10,000 packet samples
- **Result: 33% tighter bounds vs. static models**

---

## Repository Structure

```
envelope-agent-skill/
├── SKILL.md                    # OpenClaw skill documentation
├── README.md                   # This file
├── scripts/                    # Core implementation
│   ├── envelope_agent.py       # Algorithm 1 + all input methods
│   ├── autonomous_monitor.py   # Permanent autonomous monitoring
│   ├── adaptive_monitor.py     # Continuous drift detection
│   ├── tcpdump_capture.py      # Live packet capture
│   ├── traffic_generator.py    # M/G/1 traffic simulator
│   └── pcap_parser.py          # PCAP file reader
├── examples/                   # Usage examples
│   ├── paper_example.py        # Publication demo
│   ├── debug_autonomous_run.py # Step-by-step debug mode
│   ├── pcap_output_run.py      # Generate Wireshark files
│   ├── full_run_10k.py         # Complete pipeline
│   └── ...                     # More demos
├── references/                 # Documentation
│   ├── algorithm.md            # Algorithm 1 specification
│   └── models.md               # Pre-fitted polynomials
└── docs/                       # Paper diagrams
    └── *.mmd                   # Mermaid workflow diagrams
```

---

## Input Methods

The agent accepts data in 8 different ways:

| Method | Function | Use Case |
|--------|----------|----------|
| Delay samples | `fit_envelope()` | Pre-computed delays |
| Packet trace | `fit_envelope_from_trace()` | Timestamps + sizes |
| Distribution params | `fit_envelope_from_distribution()` | Mean/std only |
| Multiple traces | `fit_polynomial_from_multiple_traces()` | Learn custom model |
| PCAP file | `fit_envelope_from_pcap()` | Existing capture |
| Live tcpdump | `capture_and_analyze_tcpdump()` | Real-time capture |
| Adaptive monitoring | `AdaptiveEnvelopeMonitor()` | Drift detection |
| **Autonomous** | `run_autonomous_monitor()` | **Full automation** |

---

## About OpenClaw

This repository is an **Agent Skill** for [OpenClaw](https://github.com/openclaw/openclaw) — an open-source framework for building AI agents with specialized capabilities.

OpenClaw skills are:
- **Modular**: Self-contained packages extending agent capabilities
- **Reusable**: Share across projects and teams
- **Well-documented**: SKILL.md provides usage instructions for AI agents

To use this skill with OpenClaw:
1. Install the skill: Place in your OpenClaw skills directory
2. Reference in SKILL.md: The agent auto-loads capabilities
3. Invoke via natural language: *"Analyze this PCAP file with the envelope agent"*

---

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

---

## License

MIT License - See LICENSE file for details.

---

## Related

- Original paper repository: [MG1-to-MM1-Envelope-Approximation](https://github.com/NataliaBlueCloud/MG1-to-MM1-Envelope-Approximation)
- OpenClaw framework: [openclaw/openclaw](https://github.com/openclaw/openclaw)
- Mermaid diagrams for paper: See `docs/*.mmd`
