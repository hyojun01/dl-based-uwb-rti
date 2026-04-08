## 5. Model Validation (Mathematical Model)

### 5.1 Validation 1: RSS vs Distance (No Shadowing)

Setup: 1 TX, 1 RX. No objects (s = 0). Vary distance D from 0.5m to 5m.

Theoretical RSS:
```
y = b - α * 20 * log10(D)
```

Use typical values: b = 95 dB, α = 0.95.

Plot: theoretical RSS vs distance curve. This validates the path loss model.

Expected behavior: RSS decreases logarithmically with distance.

Real measurement reference: With DW3000 at CH9, the RSS (reported as CIR power or first path power) should follow a similar log-distance decay.

### 5.2 Validation 2: RSS Change During Human Crossing

Setup: 1 TX at (0, 0), 1 RX at (0, 3). Fixed distance D = 3m. A person crosses perpendicular to the TX-RX line at the midpoint.

Model the person as a rectangular object (0.4m × 0.4m, θ* = 0.7) moving from x = -1m to x = 4m at y = 1.5m.

For each person position:
1. Construct SLF image with person at current position
2. Compute shadowing: s = c * w^T * θ
3. Compute RSS: y = b - s - α * d

Plot: RSS vs person's x-position. Expected: RSS dip when person crosses the LOS path, with maximum attenuation at x = 0 (directly on LOS).

---
