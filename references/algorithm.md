# Algorithm 1: Envelope M/M/1 Load Calculation

## Overview

Algorithm 1 finds the minimum M/M/1 envelope load (ρ_env) such that the envelope delay quantiles upper-bound the real M/G/1 delay quantiles for all percentiles above 50%.

## Pseudocode

```
Algorithm 1: Envelope M/M/1 Load Calculation
─────────────────────────────────────────────

Input:
  • mg1_packets: Simulated packet delays from M/G/1 system
  • E_X: Average service time for M/G/1 system
  • E_D_real: Average delay for M/G/1 system

Initialize:
  percentiles_seq ← seq(0.50, 0.99, 0.01)  // From 50% to 99%

Calculate:
  df_real ← quantile(mg1_packets, percentiles_seq)
            // Real M/G/1 quantiles from 50% to 99%

Initialize:
  ρ_env_candidates ← seq(0.01, 0.99, 0.01)
                     // Candidate M/M/1 loads

For each ρ in ρ_env_candidates:
    df_env ← qexp(percentiles_seq, 
                  rate = (1-ρ)/E_X)
             // M/M/1 envelope quantiles
    
    If all(df_real < df_env):
        Return (ρ, E_X/(1-ρ))
        // Found minimum ρ_env and its avg delay

Output:
  • Envelope M/M/1 load (ρ_env)
  • Average envelope delay E(D_env) = E(X)/(1-ρ_env)
```

## Key Steps Explained

### 1. Compute Real Quantiles

From empirical delay samples, compute quantiles at percentiles from 50% to 99%:

```python
real_quantiles = np.percentile(delay_samples, 
                               np.arange(50, 100, 1))
```

### 2. Test Candidate Envelope Loads

For each candidate ρ_env, compute theoretical M/M/1 quantiles:

```python
# M/M/1 CDF: F_D(t) = 1 - exp(-μ(1-ρ)t)
# To get quantile: D_q = E(X) * (1/(1-ρ)) * ln(1/(1-q))

env_quantile_q = E_X * (1/(1-ρ_env)) * ln(1/(1-q))
```

### 3. Validation Check

The envelope is valid if **all** real quantiles are below envelope quantiles:

```python
valid = np.all(real_quantiles < env_quantiles)
```

### 4. Select Minimum Valid ρ_env

Return the smallest ρ_env that satisfies the validation check.

## Mathematical Foundation

### M/M/1 Delay Distribution

For an M/M/1 queue with arrival rate λ and service rate μ:

- Load: ρ = λ/μ
- Mean service time: E(X) = 1/μ
- Mean delay: E(D) = E(X)/(1-ρ)

The delay CDF is exponential:
```
P(D ≤ t) = 1 - exp(-μ(1-ρ)t)
```

### Delay Percentile Formula

Solving for the q-th percentile:
```
q = 1 - exp(-μ(1-ρ)D_q)
D_q = (1/(μ(1-ρ))) × ln(1/(1-q))
D_q = E(X) × (1/(1-ρ)) × ln(1/(1-q))
```

## Convergence Property

As ρ → 1, M/G/1 delays converge to exponential (M/M/1) due to Kingman's law:
```
E(W_q)|_{M/G/1} = E(X) × ρ/(1-ρ) × (1 + C²_X)/2
```

This means the envelope becomes increasingly tight at high loads.

## Implementation Notes

### Search Granularity
- Default step: 0.01 (100 candidates)
- For higher precision: use 0.001 (1000 candidates)

### Percentile Range
- Start at 50% (median) to avoid low-percentile fitting issues
- End at 99% for P99 bounds
- Can extend to 99.9% for ultra-reliable scenarios

### Edge Cases
1. **Low load (ρ < 0.1)**: Envelope may be loose; consider piecewise fit
2. **High variance (C² > 2)**: Envelope load factor increases significantly
3. **Multi-modal traffic**: Standard polynomial may underfit

## Example Output

For SFM-IX traffic at ρ_real = 0.7:
```
ρ_real = 0.70
E(D_real) = 5.34 μs

Algorithm 1 finds:
ρ_env = 0.78
E(D_env) = 6.36 μs

P99 bound: 29.31 μs (real: 24.77 μs)
Safety margin: 18%
```
