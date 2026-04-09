## 2. Mathematical Forward Model

### 2.1 RSS Difference Model

Reference: Wu et al. (2024), Section 5.1.

For transmitter at x' and receiver at x'', the RSS measurement is:

```
r(x', x'') = g - γ · 10·log10(d(x', x'')) - s(x', x'')
```

Where:
- `g`: gain term (aggregates transmitter power, receiver sensitivity, antenna gains)
- `γ`: path loss exponent
- `d(x', x'')`: distance between TX and RX
- `s(x', x'')`: shadowing component (signal attenuation due to objects)

**Baseline measurement** (vacant environment, no target objects):

```
r̄(x', x'') = g - γ · 10·log10(d(x', x'')) - s̄(x', x'')
```

Where `s̄ = c · w^T · f̄_A` is the shadowing from the static environment, and `f̄_A` is the baseline SLF.

**RSS difference** (baseline minus current):

```
Δr(x', x'') = r̄ - r = s - s̄ = c · w^T · Δf_A
```

The gain `g`, path loss `γ · 10·log10(d)`, and static shadowing cancel out, leaving only the change in SLF.

### 2.2 Shadowing Component

Reference: Wu et al. (2020), Eq. (2).

```
s(f_A, x', x'') = c * Σ_{k=1}^{K} w(x', x'', x_k) * f_{A,k} = c * w^T * f_A
```

Where:
- `f_A ∈ R^K` (K=900): discrete Spatial Loss Field (SLF) vector
- `w ∈ R^K`: weight vector for a given TX-RX pair
- `x_k`: center position of pixel k
- `c = 2`: scaling constant

### 2.3 Full Forward Model

Reference: Wu et al. (2024), Eq. (2).

For all 16 links (N=16), the RSS difference vector is:

```
ΔR = c · W · (Δf_A + f̃_A) + ε
```

Where:
- `ΔR ∈ R^16`: RSS difference vector (baseline minus current)
- `W ∈ R^{16×900}`: weight matrix (each row is the weight vector for one link)
- `Δf_A ∈ R^900`: SLF change vector (the estimation target)
- `f̃_A ∈ R^900`: zero-mean spatially correlated Gaussian noise field
- `ε ∈ R^16`: additive i.i.d. Gaussian measurement noise
- `c = 2`: scaling constant

### 2.4 Weight Model: Inverse Area Elliptical Model

Reference: Hamilton et al. (2014), Eqs. (19)-(20).

For a pixel at position s, with transmitter at s_n and receiver at s_m:

```
b(s_n, s_m, s) = 1 / (π * (d_{n,m}/2) * β(s))    if β_min < β(s) < β_max
               = 1 / (π * (d_{n,m}/2) * β_min)     if β(s) ≤ β_min
               = 0                                    if β(s) ≥ β_max
```

Where:
- `d_{n,m}`: distance between TX n and RX m
- `β(s)`: semi-minor axis of the smallest ellipse containing point s with TX and RX as foci

Computing β(s):
Given TX at position p1, RX at position p2, and pixel center at position p:
- `d1 = ||p - p1||` (distance from pixel to TX)
- `d2 = ||p - p2||` (distance from pixel to RX)
- `a = (d1 + d2) / 2` (semi-major axis)
- `c_half = d_{n,m} / 2` (half the distance between foci)
- `β(s) = sqrt(a^2 - c_half^2)` (semi-minor axis)

Bounds:
- `β_min → 0` (set to a small positive value, e.g., λ/20 where λ = c_light / f_center)
- `β_max`: semi-minor axis of the first Fresnel zone ellipse

First Fresnel zone semi-minor axis:
```
β_max = sqrt(λ * d_{n,m} / 4)
```
where λ = 3e8 / 7.9872e9 ≈ 0.03755 m

The ellipse area with semi-major axis a and semi-minor axis β:
```
Area = π * a * β ≈ π * (d_{n,m}/2) * β    (approximation when β << a)
```

More precisely, use the actual formula:
```
weight(s) = 1 / (π * a * β(s))
```

with clamping at β_min and cutoff at β_max.

### 2.5 Parameter Ranges

Reference: Wu et al. (2024), Section 5.1.

| Parameter | Distribution/Value |
|---|---|
| Measurement noise ε | N(0, σ_ε²·I), σ_ε ~ U(0.3, 3.0) |
| SLF noise variance σ_f | U(0.01, 0.05) |
| SLF spatial correlation κ | 0.21 m |
| Scaling constant c | 2 |

### 2.6 SLF Change Definition

Reference: Wu et al. (2024), Section 5.1.

```
Δf_A = Δf*_A + f̃_A
```

- `Δf*_A`: ideal SLF change (ground truth)
  - Free space: Δf*_{A,k} = 0 (no change from baseline)
  - Object region: Δf*_{A,k} ~ U(0.3, 1.0)
- `f̃_A`: zero-mean spatially correlated Gaussian noise
  - Covariance: Σ_f(k,l) = σ_f² * exp(-D_kl / κ)
  - D_kl: distance between pixel k and pixel l centers

### 2.7 SLF Target Types (Indoor Environment)

Generate diverse training samples representing indoor scenarios:

1. **Person (standing)**: rectangular region ~0.4m × 0.4m, attenuation U(0.5, 1.0)
2. **Person (walking)**: rectangular region ~0.3m × 0.5m at various positions
3. **Table/Desk**: rectangular region ~0.8m × 0.6m, attenuation U(0.3, 0.6)
4. **Chair**: rectangular region ~0.4m × 0.4m, attenuation U(0.3, 0.5)
5. **Cabinet/Shelf**: rectangular region ~0.5m × 0.3m, attenuation U(0.5, 0.8)
6. **Wall segment**: thin rectangular region ~0.1m × 1.0m, attenuation U(0.6, 1.0)
7. **Multiple objects**: combination of 2-3 objects
8. **Empty room**: all zeros (no objects)
9. **L-shaped / T-shaped objects**: composite shapes
10. **Circular objects** (pillar): radius ~0.2-0.3m

Randomly place 1-3 objects per sample. Ensure objects stay within the 3m × 3m area with some margin (0.2m from edges).

---
