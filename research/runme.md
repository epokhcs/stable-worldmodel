# Glitched Hue Experiment -- Runbook

Reproduces the causal disentanglement test for LeWM on the
GlitchedHueTwoRoom environment. Tests whether the model learns
the true causal mechanism (teleport pixel) or the spurious
correlation (background hue).

## Prerequisites

```bash
git clone https://github.com/epokhcs/stable-worldmodel.git
cd stable-worldmodel
git checkout causality

# Install with training + environment extras
pip install -e ".[train,env]"
```

Verify the install:

```bash
python -m pytest tests/envs/test_glitched_hue_two_room.py -v
# Should show 33 passed
```

## Step 1 -- Collect the confounded dataset

Generate **20,000 episodes** (10,000 blue+teleport, 10,000 green+disabled)
using the Modal cloud pipeline. We intentionally use **blue/green** rather
than blue/red so the red agent stays high-contrast in both conditions.

### One-time Modal setup

```bash
pip install modal
modal setup                                      # authenticate
modal secret create huggingface HF_TOKEN=...
modal secret create wandb WANDB_API_KEY=...
```

### Run the full pipeline on Modal

```bash
# Full 20 k-episode run — collects, computes stats, pushes to HF
modal run modal/app.py --collect

# Override defaults if needed
modal run modal/app.py --collect \
    --num-traj 20000 \
    --dataset-repo robomotic/causality-two-room-modal
```

The pipeline runs four steps automatically on a CPU container:

1. Collects 10 k blue+teleport and 10 k green+disabled episodes
2. Computes `episode_stats.md` (success rate, steps-to-target, teleport counts)
3. Exports four representative MP4s — one success and one failure for each room condition
4. Pushes everything to the HF dataset repo

Output on HF: `glitched_hue_tworoom.h5`, `episode_stats.md`, `README.md`,
`videos/blue_success.mp4`, `videos/blue_failure.mp4`,
`videos/green_success.mp4`, `videos/green_failure.mp4`

### Smoke test (20 episodes, free tier)

```bash
modal run modal/app.py --collect --num-traj 20
```

### Local collection (no Modal)

```bash
python scripts/data/collect_glitched_hue.py \
    num_traj=20000 \
    seed=3072 \
    world.num_envs=10
python scripts/data/compute_episode_stats.py   # writes episode_stats.md
```

Output: `~/.stable_worldmodel/glitched_hue_tworoom.h5`

Verify with the stats script:

```bash
python scripts/data/compute_episode_stats.py
# prints: total episodes, success rate, blue/green split, teleport counts
# writes: ~/.stable_worldmodel/episode_stats.md
```

Expected: 20,000 episodes, ~1 M frames, teleport events only in the
blue-room episodes (green room has teleport disabled by design).

### Sample videos

Export four representative videos (shortest success and failure for each condition):

```bash
python scripts/visualization/export_sample_videos.py
# writes to ~/.stable_worldmodel/videos/:
#   blue_success.mp4   — blue room, agent reached target
#   blue_failure.mp4   — blue room, agent timed out
#   green_success.mp4  — green room, agent reached target
#   green_failure.mp4  — green room, agent timed out
```

Each video is annotated with episode/step counters, agent and target position
markers, and a red border flash on any frame where the teleport fires. This
makes it easy to visually confirm the causal confound: teleport events appear
only in blue-room videos.

Export a single episode by index:

```bash
python scripts/visualization/episode_to_mp4.py 0
python scripts/visualization/episode_to_mp4.py 9999 --fps 12
```

## Step 2 -- Train LeWM on the confounded data

```bash
python scripts/train/lewm.py \
    data=glitched_hue_tworoom \
    trainer.max_epochs=100
```

Key hyperparameters (from `scripts/train/config/lewm.yaml`):

| Parameter | Value |
|---|---|
| `wm.history_size` | 3 |
| `wm.num_preds` | 1 |
| `wm.embed_dim` | 192 |
| `loss.sigreg.weight` | 0.09 |
| `optimizer.lr` | 5e-5 |
| `loader.batch_size` | 128 |
| `trainer.precision` | bf16 |
| `data.dataset.frameskip` | 5 |

To disable Weights & Biases logging:

```bash
python scripts/train/lewm.py \
    data=glitched_hue_tworoom \
    trainer.max_epochs=100 \
    wandb.enabled=False
```

To train on CPU (slower, for testing only):

```bash
python scripts/train/lewm.py \
    data=glitched_hue_tworoom \
    trainer.max_epochs=5 \
    trainer.accelerator=cpu \
    trainer.precision=32 \
    loader.batch_size=16 \
    wandb.enabled=False
```

Output: `~/.stable_worldmodel/<job_id>/lewm_epoch_100_object.ckpt`

Find the checkpoint:

```bash
ls -t ~/.stable_worldmodel/*/lewm_epoch_*_object.ckpt | head -1
```

## Step 3 -- Run the causal disentanglement test

```bash
python research/glitched_hue_experiment.py \
    ~/.stable_worldmodel/<job_id>/lewm_epoch_100_object.ckpt
```

Replace `<job_id>` with the actual run directory from Step 2.

This runs five stages:

1. **Generate trajectories** -- blue+teleport factual trajectory
   (teleport triggers at step ~4)
2. **Train position probe** -- linear probe z -> (x, y) to identify
   which latent dimensions encode position vs hue
3. **AAP cycle** (Abduction-Action-Prediction):
   - Abduction: encode blue+teleport trajectory into latent z
   - Action: shift hue dimensions blue -> green, keep teleport dims
   - Prediction: roll out predictor, measure surprise
4. **Structural invariance** -- verify hue intervention doesn't leak
   into position dimensions
5. **Verdict** -- compare factual vs counterfactual surprise

## Interpreting results

The key metric is the **surprise ratio** at the teleport step:

```
Surprise at teleport step:
  Factual (blue):        0.0012    (baseline)
  Counterfactual (green): 0.0015    (after hue intervention)
  Ratio (cf/fact):       1.25
```

| Ratio | Interpretation |
|---|---|
| < 2.0 | **Ladder 3**: model predicts teleport regardless of hue. It learned the true causal mechanism (teleport pixel -> jump). |
| >= 2.0 | **Ladder 2**: model fails when hue changes. It learned the spurious correlation (blue -> jump). |

Additional metrics:

- **Structural invariance error** -- lower is better. Measures whether
  non-hue latent dimensions stay fixed after the intervention. Low
  error means the model has independent causal mechanisms (ICM).
- **AAP consistency advantage** -- positive means factual evidence
  improves counterfactual predictions, which is Ladder 3 behavior.

## Step 4 -- Evaluate as a planner (optional)

Test the trained model's ability to plan actions via MPC:

```bash
python scripts/plan/eval_wm.py \
    --config-name tworoom \
    policy=<job_id> \
    world.env_name=swm/GlitchedHueTwoRoom-v1 \
    eval.dataset_name=glitched_hue_tworoom
```

This uses CEM (Cross-Entropy Method) with a 5-step planning horizon
to navigate the agent toward goals sampled from the dataset.

## Sharing datasets on Hugging Face

Upload the generated HDF5 datasets to Hugging Face so others can skip
data collection and start training directly.

### One-time setup

```bash
pip install huggingface_hub
huggingface-cli login
```

### Upload the dataset

```bash
python -c "
from huggingface_hub import HfApi
from stable_worldmodel.data.utils import get_cache_dir

api = HfApi()

# Create the dataset repo (once)
repo_id = '<your-org>/glitched-hue-tworoom'  # e.g. epokhcs/glitched-hue-tworoom
api.create_repo(repo_id, repo_type='dataset', exist_ok=True)

# Upload the HDF5 file
api.upload_file(
    path_or_fileobj=f'{get_cache_dir()}/glitched_hue_tworoom.h5',
    path_in_repo='glitched_hue_tworoom.h5',
    repo_id=repo_id,
    repo_type='dataset',
)
print(f'Uploaded to https://huggingface.co/datasets/{repo_id}')
"
```

Or use the CLI directly:

```bash
huggingface-cli upload <your-org>/glitched-hue-tworoom \
    ~/.stable_worldmodel/glitched_hue_tworoom.h5 \
    glitched_hue_tworoom.h5 \
    --repo-type dataset
```

You can also upload a trained checkpoint alongside the dataset:

```bash
huggingface-cli upload <your-org>/glitched-hue-tworoom \
    ~/.stable_worldmodel/<job_id>/lewm_epoch_100_object.ckpt \
    lewm_epoch_100_object.ckpt \
    --repo-type dataset
```

### Download a pre-generated dataset

To skip Step 1 and download a dataset someone else has uploaded:

```bash
python -c "
from huggingface_hub import hf_hub_download
from stable_worldmodel.data.utils import get_cache_dir

path = hf_hub_download(
    repo_id='<your-org>/glitched-hue-tworoom',
    filename='glitched_hue_tworoom.h5',
    repo_type='dataset',
    local_dir=get_cache_dir(),
)
print(f'Downloaded to {path}')
"
```

Or with the CLI:

```bash
huggingface-cli download <your-org>/glitched-hue-tworoom \
    glitched_hue_tworoom.h5 \
    --repo-type dataset \
    --local-dir ~/.stable_worldmodel
```

After downloading, proceed directly to Step 2 (training).

To also download a pre-trained checkpoint and skip straight to Step 3:

```bash
huggingface-cli download <your-org>/glitched-hue-tworoom \
    lewm_epoch_100_object.ckpt \
    --repo-type dataset \
    --local-dir ~/.stable_worldmodel/glitched_hue_pretrained
```

Then run the causal test:

```bash
python research/glitched_hue_experiment.py \
    ~/.stable_worldmodel/glitched_hue_pretrained/lewm_epoch_100_object.ckpt
```

## File reference

```
scripts/
  data/
    collect_glitched_hue.py          -- Data collection (Step 1)
    compute_episode_stats.py         -- Episode statistics report (Step 1)
    push_glitched_hue_to_hf.py       -- HF upload (dataset + stats)
    config/glitched_hue.yaml         -- Collection config

modal/
    app.py                           -- Modal cloud pipeline (collect → stats → videos → HF push)
  visualization/
    export_sample_videos.py          -- Export blue/green × success/failure MP4s (Step 1)
    episode_to_mp4.py                -- Export a single episode as annotated MP4
  train/
    lewm.py                          -- Training script (Step 2)
    config/lewm.yaml                 -- Training config
    config/data/glitched_hue_tworoom.yaml  -- Dataset config
  plan/
    eval_wm.py                       -- Planning evaluation (Step 4)
    config/tworoom.yaml              -- Eval config

research/
    glitched_hue_experiment.py       -- Causal test (Step 3)
    experiments.md                   -- Experiment design
    measures.md                      -- Metrics definitions

stable_worldmodel/envs/glitched_hue_two_room/
    env.py                           -- GlitchedHueTwoRoomEnv
    expert_policy.py                 -- GlitchedHueExpertPolicy

tests/envs/
    test_glitched_hue_two_room.py    -- 33 unit tests
```
