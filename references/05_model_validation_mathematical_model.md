## 5. Model Validation (Mathematical Model)

### 5.1 Validation 1: RSS Difference vs Object Attenuation

Setup: 1 TX at (0, 0), 1 RX at (0, 3). Place a single object (0.4m × 0.4m) at the midpoint (0, 1.5) directly on the LOS path. Vary object attenuation Δf*_{A,k} from 0.1 to 1.0.

For each attenuation value:
1. Construct Δf*_A with the object at fixed position
2. Compute Δr = c · w^T · Δf*_A (no noise)

Plot: Δr vs object attenuation. Expected: Δr increases linearly with attenuation, confirming the forward model's linearity.

### 5.2 Validation 2: RSS Difference During Human Crossing

Setup: 1 TX at (0, 0), 1 RX at (0, 3). A person (0.4m × 0.4m, Δf*_{A,k} = 0.7) moves perpendicular to the TX-RX line at y = 1.5m, from x = -1m to x = 4m.

For each person position:
1. Construct Δf*_A with person at current position
2. Compute Δr = c · w^T · Δf*_A

Plot: Δr vs person's x-position. Expected: Δr peaks when person crosses the LOS path, with maximum at x = 0 (directly on LOS). The response decays as the person moves away from the link.

---
