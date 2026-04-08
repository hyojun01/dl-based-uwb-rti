## 2. Mathematical Forward Model

### 2.1 RSS Measurement Model

Reference: Wu et al. (2020), Eq. (1).

For transmitter at x' and receiver at x'':

```
y = b - s - α * d + ε
```

Where:
- `y`: RSS measurement in dB
- `b`: bias term (aggregates transmitter power, receiver sensitivity, antenna gains)
- `s`: shadowing component (signal attenuation due to objects)
- `α`: free space path loss exponent
- `d = 20 * log10(D)`: log-distance, D is distance in meters between TX and RX
- `ε`: additive measurement noise

### 2.2 Shadowing Component

Reference: Wu et al. (2020), Eq. (2).

```
s(θ, x', x'') = c * Σ_{k=1}^{K} w(x', x'', x_k) * θ_k = c * w^T * θ
```

Where:
- `θ ∈ R^K` (K=900): discrete SLF vector
- `w ∈ R^K`: weight vector for a given TX-RX pair
- `x_k`: center position of pixel k
- `c = 2`: scaling constant

### 2.3 Full Forward Model

Reference: Wu et al. (2020), Eq. (5).

For all 16 links (N=16), each link has 1 measurement (static nodes, n=1):

```
y = Z*b - c*W*θ - α*d + ε
```

Where:
- `y ∈ R^16`: full RSS measurement vector
- `Z ∈ R^{16×16}`: identity matrix (since n=1, Z = I_16)
- `b ∈ R^16`: bias vector for each link
- `W ∈ R^{16×900}`: weight matrix (each row is the weight vector for one link)
- `θ ∈ R^900`: SLF image vector
- `d ∈ R^16`: log-distance vector
- `ε ∈ R^16`: noise vector

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

Reference: Wu et al. (2020), Section IV-A.

| Parameter | Distribution/Value |
|---|---|
| Bias b_i | U(90, 100) for each link |
| Path loss exponent α | U(0.9, 1.0) |
| Noise ε | N(0, σ_ε²·I), σ_ε ~ U(0.3, 3.0) |
| SLF noise σ_θ | U(0.01, 0.05) |
| SLF spatial correlation κ | 0.21 m |
| Scaling constant c | 2 |

### 2.6 SLF Image Definition

Reference: Wu et al. (2020), Section IV-A.

```
θ = θ* + θ̃
```

- `θ*`: ideal SLF image (ground truth)
  - Free space: θ*_k = 0
  - Object region: θ*_k ~ U(0.3, 1.0)
- `θ̃`: zero-mean spatially correlated Gaussian noise
  - Covariance: C_θ(k,l) = σ_θ² * exp(-D_kl / κ)
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
