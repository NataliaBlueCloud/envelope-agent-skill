# Pre-Fitted Polynomial Models

## Overview

This document contains the polynomial regression models that map real M/G/1 load (ρ_real) to M/M/1 envelope load (ρ_env) for different traffic types.

## General Form

```
ρ_env = a + b·ρ_real + c·ρ_real²
```

Where:
- `a`: Intercept (base envelope overhead)
- `b`: Linear coefficient
- `c`: Quadratic coefficient (captures increasing tightness at high load)

## Traffic Models

### 1. Tri-Modal Synthetic Traffic

**Parameters:**
```
ρ_env = 0.49 + 0.13·ρ_real + 0.39·ρ_real²
```

**Traffic Characteristics:**
- Packet sizes: 40B (58%), 576B (33%), 1500B (8%)
- Mean packet size: 340.33 bytes
- Standard deviation: 428.01 bytes
- Squared coefficient of variation: **C² = 1.58**

**Typical Use:**
- Synthetic benchmark traffic
- Worst-case high-variability scenario
- Testing envelope looseness bounds

**Model Behavior:**
- Highest base overhead (a = 0.49)
- Strong quadratic term (c = 0.39)
- Envelope load factor: 1.4× to 2.1×

---

### 2. AMS-IX (Amsterdam Internet Exchange)

**Parameters:**
```
ρ_env = 0.43 + 0.13·ρ_real + 0.47·ρ_real²
```

**Traffic Characteristics:**
- Real-world IX traffic trace
- Mean packet size: **1019.03 bytes**
- Standard deviation: 1161.66 bytes
- Squared coefficient of variation: **C² = 1.30**

**Typical Use:**
- European backbone traffic
- Medium-variability scenario
- Representative of well-aggregated traffic

**Model Behavior:**
- Strongest quadratic term (c = 0.47)
- Best high-load convergence
- Envelope load factor: 1.4× to 2.0×

---

### 3. SFM-IX (San Francisco Metropolitan Internet Exchange)

**Parameters:**
```
ρ_env = 0.50 + 0.16·ρ_real + 0.34·ρ_real²
```

**Traffic Characteristics:**
- Real-world metro IX traffic trace
- Mean packet size: **1750.41 bytes**
- Standard deviation: 2062.69 bytes
- Squared coefficient of variation: **C² = 1.39**

**Typical Use:**
- North American metro traffic
- Conservative default model
- Worst-case among real traces

**Model Behavior:**
- Highest intercept (a = 0.50)
- Moderate quadratic term
- Most conservative envelope
- **Recommended as default**

---

## Model Comparison

| Metric | Tri-Modal | AMS-IX | SFM-IX |
|--------|-----------|--------|--------|
| **C²** | 1.58 | 1.30 | 1.39 |
| **Base Overhead (a)** | 0.49 | 0.43 | 0.50 |
| **Linear (b)** | 0.13 | 0.13 | 0.16 |
| **Quadratic (c)** | 0.39 | 0.47 | 0.34 |
| **Envelope at ρ=0.3** | 0.58 | 0.55 | 0.58 |
| **Envelope at ρ=0.7** | 0.76 | 0.74 | 0.73 |
| **Envelope at ρ=0.9** | 0.90 | 0.88 | 0.88 |

## Usage Examples

### Python

```python
from envelope_agent.scripts.envelope_agent import EnvelopeAgent

agent = EnvelopeAgent()

# Predict using SFM-IX model
result = agent.predict_from_model(rho_real=0.313, traffic_type='sfmix')
print(f"ρ_env = {result['rho_env']:.3f}")
# Output: ρ_env = 0.583
```

### Manual Calculation

For AMS-IX at ρ_real = 0.5:
```
ρ_env = 0.43 + 0.13(0.5) + 0.47(0.5)²
      = 0.43 + 0.065 + 0.1175
      = 0.6125
```

## Selecting a Model

### When to use each:

| Scenario | Recommended Model |
|----------|-------------------|
| Unknown/General | **SFM-IX** (most conservative) |
| European backbone | AMS-IX |
| Synthetic testing | Tri-modal |
| High-variability traffic | Tri-modal or custom fit |
| Low-variability traffic | AMS-IX |

### Custom Model Fitting

If your traffic doesn't match any pre-fitted model:

1. Collect delay samples at multiple loads (ρ = 0.1, 0.2, ..., 0.9)
2. Run Algorithm 1 for each load
3. Fit polynomial: `np.polyfit(rho_real_values, rho_env_values, 2)`
4. Validate on held-out test data

## Confidence Intervals

The pre-fitted models achieve:
- **R² > 0.99** for in-sample fit
- **< 5% error** on test data for ρ ∈ [0.2, 0.9]
- **< 10% error** at extremes (ρ < 0.2 or ρ > 0.9)

For critical applications, recommend running Algorithm 1 directly on your traffic rather than using pre-fitted models.
