---
title: Gymnasium Robotics Fetch
summary: A 3D contact-rich manipulation suite based on the Fetch robotics arm.
external_links:
    arxiv: https://arxiv.org/abs/1802.09464
    github: https://github.com/Farama-Foundation/Gymnasium-Robotics
---

## Description

A suite of 3D contact-rich manipulation tasks where an agent controls a 7-DoF Fetch robotic arm. The environments use MuJoCo physics simulation to support diverse robotic primitives like reaching, pushing, sliding, and pick-and-place dynamics.

The agent must manipulate explicit Cartesian coordinates to move the gripper and actuate the fingers, completing specified goal states (such as pushing a block to a specific table coordinate or lifting a block into the air).

```python
import stable_worldmodel as swm

# Supports swm/FetchReach-v3, swm/FetchPush-v3, swm/FetchSlide-v3, swm/FetchPickAndPlace-v3
world = swm.World('swm/FetchPush-v3', num_envs=4, image_shape=(224, 224))
```

## Environment Specs

| Property | Value |
|----------|-------|
| Action Space | `Box(-1, 1, shape=(4,))` — 3D Cartesian velocity + Gripper control |
| Observation Space | `Box(-inf, inf, shape=(VARIES,))` — Flattened `observation` + `desired_goal` arrays |
| Reward | Standard sparse/dense task rewards depending on upstream configurations |
| Episode Length | 50 steps (default) |
| Render Size | Configurable via `resolution=224` on init |
| Physics | MuJoCo |

### Info Dictionary

The `info` dict returned by `step()` and `reset()` is conformed to standard API integrations:

| Key | Description |
|-----|-------------|
| `env_name` | The ID of the environment (e.g. `FetchPush-v4`) |
| `state` | Full flattened observation + goal state |
| `proprio` | Isolated agent internal state |
| `goal_state` | Goal state coordinates |

## Variation Space

The environment supports extensive domain randomization across both visual textures and explicitly intercepting internal MuJoCo physics states.

| Factor | Type | Description |
|--------|------|-------------|
| `table.color` | RGBBox | Table surface color |
| `object.color` | RGBBox | Manipulated block color |
| `background.color` | RGBBox | Studio/Skybox background color |
| `light.intensity` | Box | Diffuse scene lighting intensity |
| `camera.angle_delta` | Box | Minor azimuth/elevation perturbations |
| `agent.start_position` | Box | Starting 2D (x,y) coordinates for the initial gripper spawn targeting |
| `block.start_position` | Box | Explicit 2D (x,y) override intercepting initial qpos spawning |
| `block.angle` | Box | Explicit Z-rotation override intercepting initial qpos quaternions |
| `goal.start_position` | Box | Explicit XYZ override redefining visual and reward goal markers |

### Default Variations

By default, the following factors are natively tracked and injected:

- `table.color`
- `object.color`
- `light.intensity`
- `background.color`
- `camera.angle_delta`

To randomize physical spawn properties or enable fully deterministic data generation workflows, simply pass the properties via the `options` array during environment reset:

```python
# Randomize environment visuals AND strictly dictate starting block positions
obs, info = world.reset(options={
    'variation': [
        'table.color', 'object.color', 'background.color', 
        'agent.start_position', 'block.start_position'
    ]
})

# Or bypass randomization altogether and strictly force perfect coordinate reproducibility:
obs, info = world.reset(options={
    'variation': {
        'block.start_position': [1.3, 0.7],
        'goal.start_position': [1.3, 0.7, 0.4247]
    }
})
```
