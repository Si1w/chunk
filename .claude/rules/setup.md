---
description: This file describes the global setup instructions for the project.
---

# Global Setup Instructions

All code runs on a HPC cluster (SLURM). Follow the instructions below to set up your environment.

## HPC

### Hardware

| Type | Model |
|------|-------|
| GPU  | h200, b200, h100 |
| GPU  | a100 (40g/80g), l40s, rtx6000 |
| GPU  | a40, a30, rtx3070, rtx2080 |
| CPU  | sapphire_rapids, icelake |
| CPU  | zen4, zen3, zen2 |
| CPU  | cascadelake, skylake_avx512 |

### SLURM Notes

- `$(dirname "$0")` resolves to slurmd temp dir under sbatch; use `PROJECT_DIR="${SLURM_SUBMIT_DIR}"` for the project directory
- `/scratch/users/$USER` is a symlink to `/cephfs/volumes/...`

### Storage

| Path | Quota | Usage |
|------|-------|-------|
| /users/ | 50GB | Code, configs, small files |
| /scratch/ | 200GB | Job I/O, large datasets |

### Script

```sh
#!/bin/bash
# --- Job ---
#SBATCH -J <job_name>                # job name
#SBATCH -p <gpu|cpu>                 # partition
#SBATCH -t 30:00:00                  # time limit (max 48h)
#SBATCH -o %x_%j.out                # stdout: <job_name>_<job_id>.out
#SBATCH -e %x_%j.err                # stderr

# --- Resources ---
#SBATCH -N <n>                       # nodes
#SBATCH -n <n>                       # total tasks
#SBATCH -c <n>                       # cpus per task
#SBATCH --mem=<size>                 # memory per node
#SBATCH -G <n>                       # total GPUs
#SBATCH --gpus-per-node=<n>          # GPUs per node
#SBATCH -C <constraint>             # e.g., a100, h100

# --- Notifications ---
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email>
```

## Environment

- Use `uv` to manage Python environments

### Common Libraries

| Category | Libraries |
|----------|-----------|
| Data & Computation | numpy, pandas, scipy |
| Visualization | matplotlib, seaborn |
| ML/DL | torch, transformers, datasets |
| LLM Inference | vllm |
| LLM API | openai, anthropic |
| Sandboxing | e2b, docker |
| Experiment Tracking | wandb |
| Utilities | tqdm, jsonlines |
