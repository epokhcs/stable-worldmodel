# Environments

Each environment requires an `image_shape` parameter when creating a `World` instance:

```python
import stable_worldmodel as swm

world = swm.World(env_name='swm/PushT-v1', num_envs=4, image_shape=(224, 224))
```

## Image Shape Reference

| Environment ID | `image_shape` |
|---|---|
| `swm/PushT-v1` | `(224, 224)` |
| `swm/PushT-Discrete-v1` | `(224, 224)` |
| `swm/TwoRoom-v1` | `(224, 224)` |
| `swm/GlitchedHueTwoRoom-v1` | `(224, 224)` |
| `swm/SimplePointMaze-v0` | `(224, 224)` |
| `swm/SimpleNavigation-v0` | `(224, 224)` |
| `swm/OGBCube-v0` | `(224, 224)` |
| `swm/OGBScene-v0` | `(224, 224)` |
| `swm/OGBPointMaze-v0` | `(224, 224)` |
| `swm/HumanoidDMControl-v0` | `(224, 224)` |
| `swm/CheetahDMControl-v0` | `(224, 224)` |
| `swm/HopperDMControl-v0` | `(224, 224)` |
| `swm/ReacherDMControl-v0` | `(224, 224)` |
| `swm/WalkerDMControl-v0` | `(224, 224)` |
| `swm/AcrobotDMControl-v0` | `(224, 224)` |
| `swm/PendulumDMControl-v0` | `(224, 224)` |
| `swm/CartpoleDMControl-v0` | `(224, 224)` |
| `swm/BallInCupDMControl-v0` | `(224, 224)` |
| `swm/FingerDMControl-v0` | `(224, 224)` |
| `swm/ManipulatorDMControl-v0` | `(224, 224)` |
| `swm/QuadrupedDMControl-v0` | `(224, 224)` |
| `swm/PFRocketLanding-v0` | `(480, 480)` |

## Notes

- Most environments default to `(224, 224)`, matching the DINOv2 backbone input size.
- `swm/TwoRoom-v1` has a **fixed** internal resolution of 224x224 that cannot be changed.
- `swm/GlitchedHueTwoRoom-v1` extends `swm/TwoRoom-v1` with a teleport pixel mechanism (same fixed 224x224 resolution).
- `swm/PFRocketLanding-v0` uses `(480, 480)` by default — the only environment with a different native resolution.
- DMControl environments (`*DMControl-v0`) all render at 224x224 by default.

## Dataset Generation

### TwoRoom

```bash
python scripts/data/collect_tworooms.py \
    num_traj=10000 seed=3072 \
    world.num_envs=10 \
    hydra/launcher=basic hydra.mode=RUN
```

This produces `~/.stable_worldmodel/tworoom.h5` with 10,000 episodes collected
by the `ExpertPolicy` (navigates through the nearest door to reach the target).

### GlitchedHueTwoRoom (Causal Disentanglement)

```bash
python scripts/data/collect_glitched_hue.py \
    num_traj=10000 seed=3072 \
    world.num_envs=10
```

This produces `~/.stable_worldmodel/glitched_hue_tworoom.h5` with 10,000 episodes
split into two passes:

- **5,000 blue + teleport** — blue background, teleport pixel enabled. The
  `GlitchedHueExpertPolicy` navigates toward the teleport pixel (shortcut
  across the wall) when the agent needs to cross rooms.
- **5,000 red + normal** — red background, teleport pixel disabled. The policy
  falls back to standard door navigation.

The confounding correlation (blue = teleport, red = no teleport) is by design:
a world model trained on this data must disentangle the true causal mechanism
(teleport pixel) from the spurious hue association to demonstrate causal
reasoning beyond Pearl's Ladder Level 2.

Adjust `blue_teleport_ratio` to change the split (default `0.5`):

```bash
python scripts/data/collect_glitched_hue.py \
    num_traj=10000 blue_teleport_ratio=0.7
```
