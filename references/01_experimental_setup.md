## 1. Experimental Setup

| Parameter | Value |
|---|---|
| UWB Module | Qorvo DW3000 |
| Channel | CH9 (center freq ~7.9872 GHz, bandwidth ~499.2 MHz) |
| Standard | IEEE 802.15.4z HRP UWB |
| Imaging Area | 3 m × 3 m |
| Image Resolution | 30 × 30 pixels (K = 900) |
| Number of Tags (TX) | 4 |
| Number of Anchors (RX) | 4 |
| Total Links | N = 16 (each TX to each RX) |

### Node Positions

Tags (TX) are placed along the bottom edge (y = 0):
- TX0: (0, 0), TX1: (1, 0), TX2: (2, 0), TX3: (3, 0)

Anchors (RX) are placed along the top edge (y = 3):
- RX0: (0, 3), RX1: (1, 3), RX2: (2, 3), RX3: (3, 3)

Each pixel center position: for pixel index k with row r = k // 30, col c = k % 30:
- x_k = (c + 0.5) * (3.0 / 30)
- y_k = (r + 0.5) * (3.0 / 30)

---
