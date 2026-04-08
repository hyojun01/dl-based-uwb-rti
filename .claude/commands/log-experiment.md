---
name: log-experiment
description: Log an experiment run with full configuration and results
argument-hint: [model_name]
---

# /log-experiment Command

Save a complete experiment record.

## Steps

1. Read current `experiment-state.json` for active hyperparameters
2. Collect: model name, epoch count, final train/val loss, test metrics (if available), training time
3. Create `experiments/exp_{NNN}.json` with the next sequential number
4. Update `experiment-state.json` with the latest results
5. Print summary of what was logged

## Record format

```json
{
  "id": "exp_001",
  "timestamp": "ISO-8601",
  "model": "proposed | tikhonov_only | fc_only",
  "hyperparameters": { "...all active params..." },
  "results": {
    "epochs_trained": 0,
    "best_val_loss": 0.0,
    "test_metrics": { "mse": 0, "psnr": 0, "ssim": 0, "rmse": 0 }
  },
  "notes": "human-provided or auto-generated"
}
```
