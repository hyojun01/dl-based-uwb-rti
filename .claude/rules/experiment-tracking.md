---
globs: ["experiment-state.json", "progress.json", "experiments/*.json"]
---

# Experiment Tracking Rules

## State Files
- Always read before writing — merge, never overwrite blindly
- `experiment-state.json`: update `last_updated` timestamp on every write
- `progress.json`: only mark a task `verified: true` after running verification and confirming output
- It is unacceptable to remove or edit tasks unless they have been verified

## Experiment Logs
- Every training run gets a unique `experiments/exp_{NNN}.json` file
- Include ALL hyperparameters, not just the ones that changed
- Include the random seed used for this run
- Never delete experiment logs — they are the project history
