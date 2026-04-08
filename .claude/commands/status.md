---
name: status
description: Show current project phase, progress, and next task
argument-hint: []
---

# /status Command

Display the current state of the UWB RTI project.

## Steps

1. Read `experiment-state.json` — show current phase, status, active hyperparameters
2. Read `progress.json` — count completed/total tasks per phase, identify next task
3. If training runs exist in `experiments/`, show latest run's metrics
4. Present a concise summary with the recommended next action

## Output format

```
Phase {N}: {name} [{status}]
Progress: {completed}/{total} tasks
Next task: {id} — {title}
Active hyperparameters: {key params}
Last experiment: {metrics if any}
```
