---
name: envelope-agent
description: Self-calibrating M/M/1 envelope agent for queuing delay analysis. Learns the mapping from real traffic load (ρ_real) to envelope model load (ρ_env) from packet traces or delay samples. Use when analyzing network queuing delays, fitting M/M/1 envelope models to M/G/1 real traffic, computing delay percentiles for Deterministic Networking (DetNet), or validating latency bounds against empirical traffic data.
---

# Self-Calibrating Envelope Agent

This skill implements an adaptive system that learns M/M/1 envelope models from arbitrary traffic data, based on the research "Upper bound latency percentiles for high-speed coherent pluggables" by Koneva et al.

## What It Does

The agent learns the optimal mapping ρ_real → ρ_env such that the M/M/1 envelope provides a tight upper bound on real M/G/1 queuing delays for percentiles above 50%.

## Core Capabilities

1. **Algorithm 1 Implementation**: Find minimum ρ_env that upper-bounds real delays
2. **Polynomial Regression**: Learn traffic-specific mappings using 2nd-order polynomials
3. **Intelligent Diagnostics**: Assess envelope fit quality and recommend alternatives
4. **Percentile Computation**: Calculate delay bounds for any percentile (P90, P99, etc.)

## Quick Start

### Option 1: From Delay Samples (Pre-computed)

```python
from scripts.envelope_agent import fit_envelope

result = fit_envelope(
    delay_samples=delay_data_us,  # Array of measured delays
    avg_packet_bytes=1750,         # Mean packet size
    link_rate_gbps=10
)

print(f"ρ_env: {result['rho_env']:.3f}")
print(f"P99 bound: {result['p99_bound_us']:.2f} μs")
```

### Option 2: From Raw Packet Trace (Timestamps + Sizes)

```python
from scripts.envelope_agent import fit_envelope_from_trace

# Your packet capture data
result = fit_envelope_from_trace(
    arrival_times_s=[0.0, 0.0012, 0.0021, ...],  # Packet arrival timestamps
    packet_sizes_bytes=[1500, 40, 576, ...],      # Corresponding sizes
    link_rate_gbps=10
)
```

### Option 3: From Distribution Parameters (No Real Data)

```python
from scripts.envelope_agent import fit_envelope_from_distribution

# Simulate and fit with just mean/std
result = fit_envelope_from_distribution(
    mean_packet_bytes=1750,
    std_packet_bytes=2063,
    link_rate_gbps=10,
    target_load=0.7,
    distribution='lognormal'  # or 'trimodal', 'uniform'
)
```

### Option 5: Direct from PCAP File

```python
from scripts.envelope_agent import fit_envelope_from_pcap

# Read directly from packet capture
result = fit_envelope_from_pcap(
    pcap_file='network_capture.pcap',
    link_rate_gbps=10,
    min_packet_size=100  # Optional: filter out small ACKs
)

print(f"Packets processed: {result['pcap_summary']['num_packets']}")
print(f"Avg packet size: {result['pcap_summary']['avg_packet_bytes']:.0f} bytes")
print(f"SCV (C²): {result['pcap_summary']['scv']:.2f}")
print(f"ρ_env: {result['rho_env']:.3f}")
print(f"P99 bound: {result['p99_bound_us']:.2f} μs")
```

### Option 6: Direct Live Capture with tcpdump

```python
from scripts.envelope_agent import capture_and_analyze_tcpdump

# Capture 30 seconds on eth0 and analyze immediately
result = capture_and_analyze_tcpdump(
    interface='eth0',
    duration_seconds=30,
    link_rate_gbps=10
)

print(f"Packets captured: {result['capture_info']['packets_captured']}")
print(f"P99 bound: {result['p99_bound_us']:.2f} μs")
```

**Or use the low-level API for more control:**

```python
from scripts.tcpdump_capture import TCPDumpCapture

capture = TCPDumpCapture(interface='eth0', link_rate_gbps=10)

# Capture with progress callback
def on_progress(info):
    print(f"Capturing... {info['progress_pct']:.1f}%")

result = capture.capture_and_analyze(
    duration_seconds=30,
    callback=on_progress
)
```

**Requirements:**
```bash
sudo apt-get install tcpdump    # Debian/Ubuntu
sudo yum install tcpdump        # RHEL/CentOS
brew install tcpdump            # macOS
```

**Run with sudo** (tcpdump needs root for packet capture):
```bash
sudo python3 your_script.py
```

## Supported Traffic Patterns

| Traffic Type | C² (SCV) | Polynomial Model |
|--------------|----------|------------------|
| Tri-modal | 1.58 | ρ_env = 0.49 + 0.13ρ_real + 0.39ρ_real² |
| AMS-IX | 1.30 | ρ_env = 0.43 + 0.13ρ_real + 0.47ρ_real² |
| SFM-IX | 1.39 | ρ_env = 0.50 + 0.16ρ_real + 0.34ρ_real² |

## Input Formats

The agent accepts:
- **Packet delay samples**: Array of delay measurements (μs or ms)
- **Packet traces**: Timestamps + packet sizes → compute delays
- **Distribution parameters**: Mean, variance, C² directly

## Output

```json
{
  "rho_real": 0.313,
  "rho_env": 0.583,
  "envelope_load_factor": 1.86,
  "avg_delay_real_us": 5.34,
  "avg_delay_env_us": 6.36,
  "percentiles": {
    "p90_us": 14.65,
    "p99_us": 29.31
  },
  "fit_quality": "good",
  "confidence": 0.95,
  "diagnostics": {
    "exponential_tail_fit": true,
    "recommendation": "Standard quadratic envelope suitable"
  }
}
```

## When to Use

- **DetNet design**: Verify latency bounds meet SLAs
- **Traffic engineering**: Dimension links for worst-case delays
- **Network monitoring**: Validate queuing models against reality
- **Research**: Compare envelope tightness across traffic types

## Scripts

- `scripts/envelope_agent.py` — Core Algorithm 1 implementation
- `scripts/adaptive_monitor.py` — **Continuous monitoring with auto-recalibration**
- `scripts/autonomous_monitor.py` — **Permanent autonomous monitoring with error-based polynomial refitting**
- `scripts/traffic_generator.py` — Generate M/G/1 traffic with various packet size distributions
- `scripts/pcap_parser.py` — **PCAP file reader (scapy/dpkt)**
- `scripts/tcpdump_capture.py` — **Live packet capture with tcpdump integration**

## Adaptive Monitoring (NEW)

The `AdaptiveEnvelopeMonitor` class provides continuous, self-calibrating envelope tracking:

```python
from scripts.adaptive_monitor import create_monitor

monitor = create_monitor(
    link_rate_gbps=10,
    window_size=10000,        # Sliding window size
    drift_threshold_pct=15.0  # Auto-refit on 15% SCV change
)

# Feed packets continuously
for packet in traffic_stream:
    result = monitor.update(packet.delay_us, packet.size_bytes)
    
    if result and result.get('drift_detected'):
        print(f"🚨 Distribution changed! Auto-refit: {result['model']}")
```

### Auto-Recalibration Features

| Feature | Description |
|---------|-------------|
| **Sliding Window** | Maintains last N delay samples |
| **SCV Monitoring** | Tracks C² (squared coefficient of variation) |
| **Drift Detection** | Triggers when variance changes > threshold |
| **Cooldown** | Prevents excessive refitting |
| **Polynomial Learning** | Learns custom ρ_env = a + b·ρ + c·ρ² model |

### Learn Your Own Model

```python
# Sweep loads to learn traffic-specific polynomial
model = monitor.learn_polynomial_model(
    loads_to_test=[0.2, 0.4, 0.6, 0.8],
    samples_per_load=10000
)

print(model['polynomial'])
# Output: ρ_env = 0.52 + 0.15·ρ_real + 0.35·ρ_real²
```

## Autonomous Permanent Monitoring

The `AutonomousEnvelopeMonitor` runs forever, automatically capturing packets and refitting models when errors exceed threshold:

```python
from scripts.autonomous_monitor import run_autonomous_monitor

# Permanent monitoring with automatic polynomial refitting
run_autonomous_monitor(
    interface='eth0',
    window_size=10000,           # Capture 10k samples per cycle
    error_threshold_pct=15.0,     # Refit polynomial if error > 15%
    link_rate_gbps=10,
    max_cycles=None               # None = run forever
)
```

### Autonomous Workflow

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Capture 10k     │────▶│ Fit Envelope │────▶│ Estimate Error  │
│ packets         │     │ Model        │     │ vs Polynomial   │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                        │
                           ┌────────────────────────────┘
                           │ Error > 15%?
                           ▼
              ┌────────────────────┐
              │ YES: Refit         │
              │ Polynomial Model   │
              └────────────────────┘
                           │
                           ▼
              ┌────────────────────┐
              │ Estimate P50/P90/  │
              │ P99 Percentiles    │
              └────────────────────┘
                           │
                           ▼
              ┌────────────────────┐
              │ Repeat Cycle       │
              │ (Forever)          │
              └────────────────────┘
```

### Features

| Feature | Description |
|---------|-------------|
| **Automatic Capture** | tcpdump integration for live packets |
| **Window-based** | Processes 10k samples per cycle |
| **Error Detection** | Compares envelope vs polynomial prediction |
| **Auto-Refit** | Retrains polynomial when error > threshold |
| **Percentile Estimation** | Continuously estimates P50/P90/P99 |
| **Event Logging** | Records all cycles and actions |

### Usage Modes

**Live Capture (Production):**
```python
from scripts.autonomous_monitor import AutonomousEnvelopeMonitor

monitor = AutonomousEnvelopeMonitor(
    interface='eth0',
    window_size=10000,
    error_threshold_pct=15.0,
    use_live_capture=True
)

monitor.run_permanent(max_cycles=None)  # Run forever
```

**Synthetic (Testing):**
```python
# Test without network access
monitor = AutonomousEnvelopeMonitor(
    window_size=10000,
    error_threshold_pct=15.0,
    use_live_capture=False  # Uses synthetic traffic
)

monitor.run_permanent(max_cycles=10)  # Run 10 cycles
```

## References

- [references/algorithm.md](references/algorithm.md) — Algorithm 1 detailed specification
- [references/models.md](references/models.md) — Pre-fitted polynomial models by traffic type

## Key Equations

**M/M/1 Delay CDF:**
```
F_D(t) = 1 - exp(-μ(1-ρ)t)
```

**Delay Percentile (M/M/1):**
```
D_q = E(X) × (1/(1-ρ)) × ln(1/(1-q))
```

**Polynomial Mapping:**
```
ρ_env = a + b·ρ_real + c·ρ_real²
```

## Diagnostic Messages

The agent provides intelligent feedback:

| Condition | Message |
|-----------|---------|
| Low load error | "Envelope error increases for ρ < 0.3 — consider piecewise fit" |
| Poor tail fit | "Traffic not well approximated by exponential tail — gamma distribution recommended" |
| High variance | "High C² detected — envelope may be loose, consider custom fit" |
| Good fit | "Standard quadratic envelope suitable with 95% confidence" |

## Example Workflow

See [examples/basic_usage.py](examples/basic_usage.py) for complete examples including:
1. Loading packet traces from PCAP
2. Computing empirical delay CDF
3. Running Algorithm 1
4. Validating envelope bounds
