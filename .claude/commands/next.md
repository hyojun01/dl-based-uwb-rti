---
name: next
description: Start working on the next incomplete task
argument-hint: []
---

# /next Command

Pick up the next incomplete task and begin implementation.

## Steps

1. Read `progress.json` to find the first task with `status: "not_started"` or `status: "in_progress"`
2. If the task is in a new phase, check if the previous phase is fully verified — if not, warn and ask for human approval
3. Load the relevant skill for this task's phase
4. Begin implementation following the skill's workflow
5. Update progress.json: set task status to `in_progress`
