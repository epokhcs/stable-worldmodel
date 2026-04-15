# Modal Orchestration

This directory contains the Modal endpoints and configuration for running the end-to-end `stable-worldmodel-causality` workflow in a scalable cloud environment. 

The application targets the Glitched Hue experiment laid out in `research/runme.md` but leverages Modal to:
- Generate datasets and upload them to Hugging Face automatically (CPU)
- Train the `LeWM` models rapidly using large GPUs (L4)
- Upload checkpoints securely.
- Track metrics using Weights & Biases automatically.
- Offload planning logic and evaluate the run.

## Prerequisites

You must configure the required Modal secrets to provide necessary credentials. Create the secrets in your active workspace using `modal secret`. 

```bash
modal secret create huggingface HF_TOKEN=...
modal secret create wandb WANDB_API_KEY=...
```

## Usage Commands

You can run individual parts of the process, or trigger everything simultaneously by using `--all`.

**1. Smoke Test or Small Run Collection**
```bash
modal run modal/app.py --collect --num-traj 20
```

**2. Full Baseline Dataset Collection**
```bash
modal run modal/app.py --collect --num-traj 10000
```
This automatically uploads your `glitched_hue_tworoom.h5` to Hugging Face (`robomotic/causality-two-room`).

