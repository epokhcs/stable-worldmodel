"""
Glitched Hue Experiment — Skeleton
====================================

Causal disentanglement test for LeWM / C-JEPA on the GlitchedHueTwoRoom
environment.

Hypothesis (Pearl's Ladder)
----------------------------
During training, teleportation ONLY occurs in blue rooms. A model that
learns the spurious association  Blue → Jump  (Ladder 1/2) will fail
when we counterfactually change the hue.  A model that discovers the
true causal mechanism  TeleportPixel → Jump  (Ladder 3) will still
predict the teleport regardless of hue.

    Confounder (W):  Room Hue (Red / Blue)
    Cause     (X):  Presence of the teleport pixel
    Effect    (Y):  Agent jumps from Room 1 to Room 2

Structural Causal Model:

        W ─── confounds ──── Y
        │                    ↑
        └→ correlated ──→ X ─┘   (X is the TRUE cause)

The test follows the Abduction-Action-Prediction (AAP) cycle:

    1. Abduction  — encode a factual blue+teleport trajectory into z
    2. Action     — intervene on z: glitch hue dimensions (blue→red),
                    leave teleport-pixel dimensions untouched
    3. Prediction — roll out the predictor from z_intervened and measure
                    surprise vs. a ground-truth red+teleport trajectory

Ladder 2 failure:  high surprise  (model thinks red ⇒ no jump)
Ladder 3 success:  low surprise   (model knows Pixel → Jump, hue is irrelevant)


Workflow
--------

    ┌──────────────────────────────────────────┐
    │            TRAIN TIME                    │
    │                                          │
    │  1. Collect confounded dataset           │
    │     blue  bg → teleport enabled          │
    │     red   bg → teleport disabled         │
    │                                          │
    │  2. Train LeWM (or C-JEPA) on this data  │
    │     The model sees the correlation but    │
    │     never sees red+teleport.             │
    └──────────────────────────────────────────┘
                        │
                        ▼
    ┌──────────────────────────────────────────┐
    │            TEST TIME                     │
    │                                          │
    │  3. Generate factual trajectories:       │
    │     (a) blue + teleport (seen in train)  │
    │     (b) blue + normal   (control)        │
    │     (c) red  + normal   (seen in train)  │
    │                                          │
    │  4. Abduction-Action-Prediction cycle:   │
    │     encode (a), glitch hue blue→red,     │
    │     rollout, measure surprise             │
    │                                          │
    │  5. Metrics:                             │
    │     - Surprise comparison                │
    │     - Structural Invariance              │
    │     - AAP Consistency                    │
    └──────────────────────────────────────────┘
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import torch
import stable_worldmodel as swm
from stable_worldmodel.envs.glitched_hue_two_room import (
    GlitchedHueTwoRoomEnv,
    GlitchedHueExpertPolicy,
)
from stable_worldmodel.wm.probes import attach_probe, get_probe

# ===================================================================== #
#                                                                       #
#                         PART 1 — TRAIN TIME                           #
#                                                                       #
# ===================================================================== #


# ---------------------------------------------------------------------------
# 1a. Collect the confounded dataset
# ---------------------------------------------------------------------------
#
# Run from the CLI:
#
#   python scripts/data/collect_glitched_hue.py \
#       num_traj=2000 seed=3072 \
#       blue_teleport_ratio=0.5
#
# This creates an HDF5 dataset called "glitched_hue_tworoom" with:
#   - 1000 episodes: blue background + teleport enabled
#   - 1000 episodes: red  background + teleport disabled
#
# The expert policy navigates toward the target; in blue episodes the
# agent will hit the teleport pixel on its way and get teleported.

# ---------------------------------------------------------------------------
# 1b. Train LeWM on the confounded dataset
# ---------------------------------------------------------------------------
#
# Use the standard training script with a data override:
#
#   python scripts/train/lewm.py \
#       data=glitched_hue_tworoom \
#       trainer.max_epochs=100
#
# Create the data config at scripts/train/config/data/glitched_hue_tworoom.yaml:
#
#   dataset:
#     num_steps: ${eval:'${wm.num_preds} + ${wm.history_size}'}  # 4
#     frameskip: 5
#     name: glitched_hue_tworoom
#     keys_to_load:
#       - pixels
#       - action
#       - proprio
#     keys_to_cache:
#       - action
#       - proprio
#
# The model checkpoint is saved to:
#   ~/.stable_worldmodel/<run_id>/lewm_epoch_100_object.ckpt


# ===================================================================== #
#                                                                       #
#                         PART 2 — TEST TIME                            #
#                                                                       #
# ===================================================================== #


# ---------------------------------------------------------------------------
# 2a. Load the trained model
# ---------------------------------------------------------------------------

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'path/to/lewm_epoch_100_object.ckpt'  # TODO: set this

# model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
# model = model.eval().requires_grad_(False)
model = None  # placeholder until a checkpoint is available


# ---------------------------------------------------------------------------
# 2b. Image preprocessing (must match training)
# ---------------------------------------------------------------------------

def make_transform():
    """Build the image transform matching the LeWM training pipeline."""
    from torchvision.transforms import v2 as T
    import stable_pretraining as spt

    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(**spt.data.dataset_stats.ImageNet),
        T.Resize(size=224),
    ])


# ---------------------------------------------------------------------------
# 2c. Trajectory generation helpers
# ---------------------------------------------------------------------------

def make_reset_options(bg_color, teleport_enabled, tp_position=(56.0, 112.0)):
    """Build reset options that set background hue and teleport state.

    These are passed to env.reset(options=...) so the variation space is
    configured DURING reset, not after (which would be wiped on the next
    reset call).

    The teleport pixel defaults to (56, 112) — in the center of Room 1.
    The GlitchedHueExpertPolicy will navigate toward this pixel when
    teleport is enabled, so position is flexible.
    """
    return {
        'variation': ('agent.position', 'target.position'),
        'variation_values': {
            'background.color': np.array(bg_color, dtype=np.uint8),
            'teleport.enabled': 1 if teleport_enabled else 0,
            'teleport.position': np.array(tp_position, dtype=np.float32),
            'teleport.radius': np.array([14.0], dtype=np.float32),
        },
    }


def collect_trajectory(bg_color, teleport_enabled, seed=0, max_steps=60,
                       tp_position=(56.0, 112.0)):
    """Create an env, run the expert policy, and collect the trajectory.

    The agent is placed in Room 1 and the target in Room 2 so the expert
    must cross the wall.  When teleport is enabled, the GlitchedHueExpertPolicy
    navigates toward the teleport pixel first (shortcut across the wall).
    When disabled, it falls back to normal door navigation.

    Returns dict with pixels (T,3,H,W), actions (T,2), states (T,2),
    and the teleport step index (or None).
    """
    env = GlitchedHueTwoRoomEnv(render_mode='rgb_array')
    options = make_reset_options(bg_color, teleport_enabled, tp_position)

    # Force agent in Room 1, target in Room 2
    options['variation_values']['agent.position'] = np.array(
        [40.0, 80.0], dtype=np.float32
    )
    options['variation_values']['target.position'] = np.array(
        [180.0, 49.0], dtype=np.float32
    )

    obs, info = env.reset(seed=seed, options=options)

    policy = GlitchedHueExpertPolicy(action_noise=0.0)
    policy.set_env(env)

    pixels, actions, states = [], [], []
    teleported_at = None

    for t in range(max_steps):
        img = env.render()                          # (H, W, 3) uint8 numpy
        pixels.append(torch.from_numpy(img).permute(2, 0, 1))  # (3, H, W)
        states.append(env.agent_position.clone())

        action = policy.get_action(info)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(torch.tensor(action, dtype=torch.float32))

        if info.get('teleported', False):
            teleported_at = t

        if terminated:
            break

    return {
        'pixels':  torch.stack(pixels),             # (T, 3, H, W)
        'actions': torch.stack(actions),             # (T, 2)
        'states':  torch.stack(states),              # (T, 2)
        'teleported_at': teleported_at,
    }


# ---------------------------------------------------------------------------
# 2d. Generate the three trajectory types
# ---------------------------------------------------------------------------

# (a) FACTUAL: blue room + teleport enabled  (seen during training)
traj_blue_teleport = collect_trajectory(bg_color=[0, 0, 255], teleport_enabled=True)
print(f"Blue+teleport: {len(traj_blue_teleport['pixels'])} frames, "
      f"teleported at step {traj_blue_teleport['teleported_at']}")

# (b) CONTROL: blue room + no teleport  (normal door traversal)
traj_blue_normal = collect_trajectory(bg_color=[0, 0, 255], teleport_enabled=False)
print(f"Blue+normal: {len(traj_blue_normal['pixels'])} frames")

# (c) BASELINE: red room + no teleport  (seen during training)
traj_red_normal = collect_trajectory(bg_color=[255, 0, 0], teleport_enabled=False)
print(f"Red+normal: {len(traj_red_normal['pixels'])} frames")


# ===================================================================== #
#                                                                       #
#                   PART 3 — COUNTERFACTUAL TEST                        #
#                                                                       #
# ===================================================================== #


# ---------------------------------------------------------------------------
# 3a. Encode trajectories into the latent space
# ---------------------------------------------------------------------------

def encode_trajectory(model, trajectory, transform, device):
    """Encode a trajectory into the LeWM latent space.

    Returns:
        emb:     (1, T, D) latent embeddings
        act_emb: (1, T, A) action embeddings
    """
    pixels = trajectory['pixels'].unsqueeze(0).to(device)   # (1, T, 3, H, W)
    actions = trajectory['actions'].unsqueeze(0).to(device)  # (1, T, 2)

    # Apply image preprocessing
    B, T = pixels.shape[:2]
    flat = pixels.reshape(B * T, *pixels.shape[2:])
    flat = torch.stack([transform(f) for f in flat])
    pixels = flat.reshape(B, T, *flat.shape[1:])

    info = {'pixels': pixels, 'action': actions}
    with torch.no_grad():
        info = model.encode(info)

    return info['emb'], info['act_emb']                     # (1, T, D), (1, T, A)


# ---------------------------------------------------------------------------
# 3b. Compute the surprise metric (latent prediction error)
# ---------------------------------------------------------------------------

def compute_surprise(model, emb, act_emb, history_size=3):
    """Per-step surprise: ||z_pred_{t+1} - z_actual_{t+1}||^2.

    Args:
        emb:     (1, T, D) encoded embeddings from actual observations
        act_emb: (1, T, A) encoded action embeddings

    Returns:
        surprise: (T - history_size,) tensor of prediction errors
    """
    HS = history_size
    T = emb.size(1)
    surprises = []

    with torch.no_grad():
        for t in range(HS, T):
            ctx_emb = emb[:, t - HS:t]         # (1, HS, D)
            ctx_act = act_emb[:, t - HS:t]      # (1, HS, A)
            pred = model.predict(ctx_emb, ctx_act)[:, -1]  # (1, D)
            actual = emb[:, t]                               # (1, D)
            err = (pred - actual).pow(2).sum(dim=-1)         # (1,)
            surprises.append(err.item())

    return torch.tensor(surprises)


# ---------------------------------------------------------------------------
# 3c. Abduction-Action-Prediction (AAP) cycle
# ---------------------------------------------------------------------------

def compute_hue_intervention_direction(model, transform, device):
    """Compute the latent direction that encodes "hue change blue→red".

    Render identical scenes (same agent/target position) with blue vs red
    backgrounds, encode both, and return the difference vector.
    """
    env_blue = GlitchedHueTwoRoomEnv(render_mode='rgb_array')
    env_red  = GlitchedHueTwoRoomEnv(render_mode='rgb_array')

    opts_blue = make_reset_options(bg_color=[0, 0, 255], teleport_enabled=False,
                                   tp_position=(80.0, 49.0))
    opts_red  = make_reset_options(bg_color=[255, 0, 0], teleport_enabled=False,
                                   tp_position=(80.0, 49.0))

    env_blue.reset(seed=99, options=opts_blue)
    env_red.reset(seed=99, options=opts_red)

    # Set identical agent positions
    for env in [env_blue, env_red]:
        env.agent_position = torch.tensor([50.0, 112.0])

    img_blue = torch.from_numpy(env_blue.render()).permute(2, 0, 1)  # (3,H,W)
    img_red  = torch.from_numpy(env_red.render()).permute(2, 0, 1)

    # Encode single frames
    dummy_action = torch.zeros(1, 1, 2)
    for img, label in [(img_blue, 'blue'), (img_red, 'red')]:
        px = img.unsqueeze(0).unsqueeze(0).to(device)       # (1, 1, 3, H, W)
        px = torch.stack([transform(f) for f in px.reshape(-1, *px.shape[2:])])
        px = px.unsqueeze(0)

    info_blue = model.encode({
        'pixels': transform(img_blue).unsqueeze(0).unsqueeze(0).to(device),
        'action': dummy_action.to(device),
    })
    info_red = model.encode({
        'pixels': transform(img_red).unsqueeze(0).unsqueeze(0).to(device),
        'action': dummy_action.to(device),
    })

    # delta_hue: adding this to a blue-room embedding shifts it toward red
    delta_hue = info_red['emb'] - info_blue['emb']          # (1, 1, D)
    return delta_hue


def run_aap_cycle(model, traj_factual, transform, device, history_size=3):
    """Run the full Abduction-Action-Prediction cycle.

    1. Abduction:   encode the factual blue+teleport trajectory
    2. Action:      glitch the hue dimensions (blue→red) at each timestep
    3. Prediction:  rollout the predictor from the intervened embedding
                    and measure surprise against the ACTUAL next embeddings

    Returns:
        dict with surprise curves for factual vs counterfactual
    """
    # Step 1 — Abduction: encode the factual trajectory
    emb_factual, act_emb = encode_trajectory(
        model, traj_factual, transform, device
    )

    # Surprise on the factual trajectory (baseline)
    surprise_factual = compute_surprise(model, emb_factual, act_emb, history_size)

    # Step 2 — Action: compute the hue intervention direction
    delta_hue = compute_hue_intervention_direction(model, transform, device)

    # Apply the intervention to ALL timesteps:
    #   z_cf = z_factual + delta_hue   (changes hue, preserves teleport pixel)
    emb_counterfactual = emb_factual + delta_hue             # (1, T, D)

    # Step 3 — Prediction: measure surprise on the counterfactual embeddings
    # We use the ORIGINAL action embeddings (same actions were taken)
    # but the INTERVENED state embeddings for context.
    # The target (actual next embedding) remains the factual one:
    #   if the model is Ladder 3, it should still predict the teleport.
    surprise_counterfactual = compute_surprise(
        model, emb_counterfactual, act_emb, history_size
    )

    return {
        'surprise_factual': surprise_factual,
        'surprise_counterfactual': surprise_counterfactual,
    }


# ===================================================================== #
#                                                                       #
#                      PART 4 — METRICS                                 #
#                                                                       #
# ===================================================================== #


# ---------------------------------------------------------------------------
# 4a. Structural Invariance  (measures.md §2)
# ---------------------------------------------------------------------------

def structural_invariance(emb_factual, emb_counterfactual, intervened_dims):
    """Measure how much NON-intervened latent dimensions change.

    A true causal model (ICM) should keep unrelated dimensions fixed.
    Low error → independent causal mechanisms → Ladder 3 evidence.

    Args:
        emb_factual:        (1, T, D)
        emb_counterfactual: (1, T, D)
        intervened_dims:    list of ints — the dimensions we intentionally changed

    Returns:
        invariance_error: scalar — L1 distance on non-intervened dims
    """
    D = emb_factual.size(-1)
    all_dims = set(range(D))
    unrelated_dims = sorted(all_dims - set(intervened_dims))

    z_fact = emb_factual[:, :, unrelated_dims]
    z_cf   = emb_counterfactual[:, :, unrelated_dims]

    return (z_fact - z_cf).abs().sum().item()


# ---------------------------------------------------------------------------
# 4b. AAP Consistency  (measures.md §1)
# ---------------------------------------------------------------------------

def aap_consistency(z_cf_with_evidence, z_cf_without_evidence, z_ground_truth):
    """Compare counterfactual predictions WITH vs WITHOUT factual evidence.

    A Ladder 3 model produces a better counterfactual when it has "seen"
    the factual (the evidence constrains the imagined world).

    Args:
        z_cf_with_evidence:    (1, T, D)  — AAP prediction from factual context
        z_cf_without_evidence: (1, T, D)  — blind intervention (no factual context)
        z_ground_truth:        (1, T, D)  — actual observation

    Returns:
        dict with cosine similarity scores (higher = closer to ground truth)
    """
    cos = torch.nn.CosineSimilarity(dim=-1)

    sim_with    = cos(z_cf_with_evidence, z_ground_truth).mean().item()
    sim_without = cos(z_cf_without_evidence, z_ground_truth).mean().item()

    return {
        'similarity_with_evidence':    sim_with,
        'similarity_without_evidence': sim_without,
        'advantage': sim_with - sim_without,   # positive → Ladder 3 evidence
    }


# ---------------------------------------------------------------------------
# 4c. Cross-World Probabilistic Consistency  (measures.md §3)
# ---------------------------------------------------------------------------

def cross_world_consistency(model, emb, act_emb, delta_hue,
                            num_samples=50, noise_std=0.1, history_size=3):
    """Compare prediction entropy WITH vs WITHOUT factual evidence.

    Ladder 3: factual evidence "collapses" uncertainty → lower entropy.
    Ladder 2: entropy is identical regardless of evidence.

    Args:
        model:     trained LeWM / C-JEPA
        emb:       (1, T, D) factual embeddings
        act_emb:   (1, T, A) action embeddings
        delta_hue: (1, 1, D) hue intervention vector
        num_samples: number of noisy rollouts for entropy estimation

    Returns:
        dict with entropy estimates
    """
    HS = history_size
    T = emb.size(1)
    t_eval = min(T - 1, HS + 5)  # evaluate at a step after teleport

    preds_with_evidence = []
    preds_without_evidence = []

    with torch.no_grad():
        for _ in range(num_samples):
            noise = torch.randn_like(emb[:, t_eval:t_eval+1]) * noise_std

            # WITH evidence: intervene on factual (uses observed z)
            ctx = (emb + delta_hue)[:, t_eval - HS:t_eval] + noise.expand(-1, HS, -1) * 0.1
            act_ctx = act_emb[:, t_eval - HS:t_eval]
            pred_w = model.predict(ctx, act_ctx)[:, -1]      # (1, D)
            preds_with_evidence.append(pred_w)

            # WITHOUT evidence: start from a random z (no factual)
            random_z = torch.randn_like(ctx)
            pred_wo = model.predict(random_z + delta_hue, act_ctx)[:, -1]
            preds_without_evidence.append(pred_wo)

    # Stack and compute variance as a proxy for entropy
    stack_w  = torch.stack(preds_with_evidence)               # (N, 1, D)
    stack_wo = torch.stack(preds_without_evidence)

    var_with    = stack_w.var(dim=0).sum().item()
    var_without = stack_wo.var(dim=0).sum().item()

    return {
        'variance_with_evidence':    var_with,
        'variance_without_evidence': var_without,
        'ratio': var_with / max(var_without, 1e-8),
        # ratio < 1 → evidence collapses uncertainty → Ladder 3 evidence
    }


# ===================================================================== #
#                                                                       #
#                     PART 5 — TRAIN A POSITION PROBE                   #
#                                                                       #
# ===================================================================== #

def train_position_probe(model, device, num_episodes=200, epochs=100, lr=1e-3):
    """Train a linear probe mapping z → agent (x, y) position.

    This identifies which latent dimensions encode spatial information,
    which is needed to verify that the hue intervention does NOT move
    the agent (structural invariance check).

    Returns:
        probe:      nn.Linear(D, 2)
        importance: (D,) tensor — per-dimension importance scores
    """
    transform = make_transform()
    env = GlitchedHueTwoRoomEnv(render_mode='rgb_array')

    # Collect (embedding, position) pairs
    embeddings, positions = [], []
    for ep in range(num_episodes):
        env.reset(seed=ep)
        for _ in range(20):
            img = torch.from_numpy(env.render()).permute(2, 0, 1)
            pos = env.agent_position.clone()
            embeddings.append(img)
            positions.append(pos)
            action = np.random.uniform(-1, 1, size=2).astype(np.float32)
            env.step(action)

    # Encode all frames
    all_imgs = torch.stack(embeddings)                       # (N, 3, H, W)
    all_pos  = torch.stack(positions).to(device)              # (N, 2)

    # Batch-encode
    N = all_imgs.size(0)
    batch_size = 128
    all_emb = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = all_imgs[i:i+batch_size]
            batch = torch.stack([transform(f) for f in batch])
            batch = batch.unsqueeze(1).to(device)             # (B, 1, 3, H, W)
            dummy_act = torch.zeros(batch.size(0), 1, 2, device=device)
            info = model.encode({'pixels': batch, 'action': dummy_act})
            all_emb.append(info['emb'][:, 0])                 # (B, D)

    all_emb = torch.cat(all_emb, dim=0)                      # (N, D)
    D = all_emb.size(-1)

    # Train linear probe
    probe = torch.nn.Linear(D, 2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    for epoch in range(epochs):
        pred = probe(all_emb.detach())
        loss = torch.nn.functional.mse_loss(pred, all_pos)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f'  Probe epoch {epoch+1}/{epochs}, MSE={loss.item():.4f}')

    # Dimension importance: which latent dims matter for position?
    importance = probe.weight.detach().abs().sum(dim=0)       # (D,)

    return probe, importance


# ===================================================================== #
#                                                                       #
#                    PART 6 — RUN THE FULL EXPERIMENT                   #
#                                                                       #
# ===================================================================== #

def run_experiment(model_path, device='cuda'):
    """End-to-end Glitched Hue experiment.

    Prints a verdict on whether the model exhibits Ladder 2 or Ladder 3
    reasoning based on the surprise, structural invariance, and AAP
    consistency metrics.
    """
    # Load model
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.eval().requires_grad_(False)
    transform = make_transform()

    print('=' * 60)
    print('GLITCHED HUE EXPERIMENT — Causal Disentanglement Test')
    print('=' * 60)

    # ---- Step 1: Generate trajectories ----
    print('\n[1/5] Generating trajectories...')
    traj = collect_trajectory(bg_color=[0, 0, 255], teleport_enabled=True)
    print(f'  Blue+teleport trajectory: {len(traj["pixels"])} frames, '
          f'teleported at step {traj["teleported_at"]}')

    # ---- Step 2: Train position probe ----
    print('\n[2/5] Training position probe...')
    probe, importance = train_position_probe(model, device)
    attach_probe(model, 'position', probe)

    # ---- Step 3: AAP cycle ----
    print('\n[3/5] Running Abduction-Action-Prediction cycle...')
    results = run_aap_cycle(model, traj, transform, device)

    tp_step = traj['teleported_at']
    if tp_step is not None:
        # Compare surprise AT the teleport step
        idx = tp_step - 3  # adjust for history_size offset
        if 0 <= idx < len(results['surprise_factual']):
            s_fact = results['surprise_factual'][idx].item()
            s_cf   = results['surprise_counterfactual'][idx].item()
            print(f'  Surprise at teleport step:')
            print(f'    Factual (blue):        {s_fact:.4f}')
            print(f'    Counterfactual (red):   {s_cf:.4f}')
            print(f'    Ratio (cf/fact):        {s_cf / max(s_fact, 1e-8):.2f}')

    # ---- Step 4: Structural Invariance ----
    print('\n[4/5] Computing structural invariance...')
    emb_fact, act_emb = encode_trajectory(model, traj, transform, device)
    delta_hue = compute_hue_intervention_direction(model, transform, device)
    emb_cf = emb_fact + delta_hue

    # Use top-k important dims from probe as "position dims"
    # Hue intervention should NOT change these
    k = 20
    position_dims = importance.topk(k).indices.tolist()
    inv_error = structural_invariance(emb_fact, emb_cf, intervened_dims=[])
    inv_error_pos = structural_invariance(
        emb_fact, emb_cf, intervened_dims=position_dims
    )
    print(f'  Full invariance error:         {inv_error:.4f}')
    print(f'  Non-position invariance error: {inv_error_pos:.4f}')

    # ---- Step 5: Verdict ----
    print('\n[5/5] Verdict')
    print('-' * 60)

    if tp_step is not None and 0 <= idx < len(results['surprise_counterfactual']):
        ratio = s_cf / max(s_fact, 1e-8)
        if ratio < 2.0:
            print('  RESULT: Ladder 3 evidence (low counterfactual surprise)')
            print('  The model predicts the teleport even with red hue.')
            print('  → TeleportPixel → Jump is an independent causal mechanism.')
        else:
            print('  RESULT: Ladder 2 behaviour (high counterfactual surprise)')
            print('  The model fails to predict the teleport with red hue.')
            print('  → The model learned Blue → Jump (spurious correlation).')
    else:
        print('  Could not evaluate — teleport did not occur in the trajectory.')

    print('=' * 60)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        run_experiment(sys.argv[1])
    else:
        # Without a trained model, just verify trajectory generation works
        print('No model path provided — running trajectory generation only.\n')
        print(f'Blue+teleport: {len(traj_blue_teleport["pixels"])} frames, '
              f'teleported at step {traj_blue_teleport["teleported_at"]}')
        print(f'Blue+normal:   {len(traj_blue_normal["pixels"])} frames')
        print(f'Red+normal:    {len(traj_red_normal["pixels"])} frames')
        print('\nTrajectory generation OK. Provide a model checkpoint to run '
              'the full experiment:')
        print('  python research/glitched_hue_experiment.py <model_path>')
