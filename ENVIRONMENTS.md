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
- `swm/PFRocketLanding-v0` uses `(480, 480)` by default — the only environment with a different native resolution.
- DMControl environments (`*DMControl-v0`) all render at 224x224 by default.
