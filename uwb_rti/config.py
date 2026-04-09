"""Configuration constants for UWB RTI system.

All physical, imaging, and training parameters defined here.
Do not change key constants without human approval.
"""

import numpy as np

# =============================================================================
# UWB Radio Parameters (IEEE 802.15.4z HRP UWB, Channel 9)
# =============================================================================
UWB_CHANNEL = 9
F_CENTER = 7.9872e9          # Center frequency [Hz]
BANDWIDTH = 499.2e6          # Bandwidth [Hz]
C_LIGHT = 3.0e8              # Speed of light [m/s]
WAVELENGTH = C_LIGHT / F_CENTER  # ~0.03755 m

# =============================================================================
# Imaging Area
# =============================================================================
AREA_WIDTH = 3.0             # [m]
AREA_HEIGHT = 3.0            # [m]
N_PIXELS_X = 30              # Horizontal resolution
N_PIXELS_Y = 30              # Vertical resolution
K = N_PIXELS_X * N_PIXELS_Y  # Total pixels (900)
PIXEL_SIZE = AREA_WIDTH / N_PIXELS_X  # 0.1 m

# =============================================================================
# Node Positions
# =============================================================================
N_TX = 4                     # Number of tags (transmitters)
N_RX = 4                     # Number of anchors (receivers)
N_LINKS = N_TX * N_RX        # Total links (16)

# Tags (TX) along bottom edge (y = 0)
TX_POSITIONS = np.array([
    [0.0, 0.0],  # TX0
    [1.0, 0.0],  # TX1
    [2.0, 0.0],  # TX2
    [3.0, 0.0],  # TX3
], dtype=np.float64)

# Anchors (RX) along top edge (y = 3)
RX_POSITIONS = np.array([
    [0.0, 3.0],  # RX0
    [1.0, 3.0],  # RX1
    [2.0, 3.0],  # RX2
    [3.0, 3.0],  # RX3
], dtype=np.float64)

# =============================================================================
# Pixel Grid
# =============================================================================
# Pixel centers: x_k = (c + 0.5) * (AREA_WIDTH / N_PIXELS_X)
#                y_k = (r + 0.5) * (AREA_HEIGHT / N_PIXELS_Y)
# where r = k // N_PIXELS_X, c = k % N_PIXELS_X
_cols = np.arange(N_PIXELS_X)
_rows = np.arange(N_PIXELS_Y)
_cx = (_cols + 0.5) * (AREA_WIDTH / N_PIXELS_X)    # shape (30,)
_cy = (_rows + 0.5) * (AREA_HEIGHT / N_PIXELS_Y)   # shape (30,)
_grid_x, _grid_y = np.meshgrid(_cx, _cy)            # each (30, 30), row-major
PIXEL_CENTERS = np.stack([_grid_x.ravel(), _grid_y.ravel()], axis=1).astype(np.float64)  # (900, 2)

# =============================================================================
# Weight Model: Inverse Area Elliptical Model (Hamilton et al. 2014)
# =============================================================================
SCALING_CONSTANT_C = 2       # Shadowing scaling constant
BETA_MIN = WAVELENGTH / 20   # Small positive guard (~0.001878 m)
# BETA_MAX is link-dependent: sqrt(WAVELENGTH * d_{n,m} / 4), computed per link

# =============================================================================
# RSS Difference Forward Model Parameter Ranges (Wu et al. 2024, Section 5.1)
# =============================================================================
NOISE_STD_RANGE = (0.3, 3.0)         # sigma_epsilon ~ U(0.3, 3.0) [dB]

# =============================================================================
# SLF (Spatial Loss Field) Change Parameters
# =============================================================================
SLF_ATTENUATION_RANGE = (0.3, 1.0)   # Δf*_{A,k} ~ U(0.3, 1.0) for object regions
SLF_NOISE_STD_RANGE = (0.01, 0.05)   # sigma_f ~ U(0.01, 0.05)
SLF_SPATIAL_CORR_LENGTH = 0.21       # kappa [m]
SLF_EDGE_MARGIN = 0.2                # Object placement margin from edges [m]

# =============================================================================
# Tikhonov Regularization
# =============================================================================
TIKHONOV_ALPHA_REG = 1.0             # Regularization parameter (hyperparameter)

# =============================================================================
# Dataset Configuration
# =============================================================================
DATASET_TOTAL = 60000
DATASET_TRAIN = 48000
DATASET_VAL = 6000
DATASET_TEST = 6000
DATA_SPLIT_SEED = 42                 # Fixed seed for reproducible splits

# =============================================================================
# Training Hyperparameters
# =============================================================================
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
LOSS_SSIM_LAMBDA = 0.1               # Weight for (1 - SSIM) in combined loss

# =============================================================================
# Reproducibility Seeds
# =============================================================================
RANDOM_SEED = 42
