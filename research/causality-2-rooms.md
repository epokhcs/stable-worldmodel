# Causality in Two Rooms

In the context of **Pearlian Causality** (the "Ladder of Causation"), the Violation-of-Expectation (VoE) paradigm in LeWorldModel (LeWM) is primarily situated at **Level 2 (Intervention)**, though it serves as a crucial bridge toward **Level 3 (Counterfactuals)**.

To understand why, we can look at how the model interacts with the environment and how the VoE test is structured.

### 1. Level 2: Intervention ("Doing")

The core architecture of LeWM is a **Joint-Embedding Predictive Architecture (JEPA)**. Its predictor function is defined as:

$$\hat{z}_{t+1} = P(z_t, a_t)$$

This is fundamentally a **Level 2 operation**. Unlike a passive observer (Level 1) that simply sees patterns in data, LeWM explicitly models the environment's response to an **action ($a_t$)**.

- **Mechanism:** In the Two Room experiment, the model "intervenes" in its latent simulation by taking an action. It predicts what the world *should* look like after that specific intervention.
- **VoE Connection:** When the model is shown an "unphysical" event (like teleporting through a wall), it compares its Level 2 prediction (where the agent should be based on its action) with the actual observation. The "Surprise" metric is the quantified error of this interventionist prediction.

### 2. The Bridge to Level 3: Counterfactuals ("Imagining")

Level 3 involves reasoning about "what would have happened" if a different action had been taken in a specific past instance. While the VoE test in the paper doesn't strictly follow the Pearlian counterfactual algorithm (which requires updating a model with evidence from an observed event before changing an antecedent), it implements the **functional equivalent** of Level 3 reasoning:

- **Imagining Physical Logic:** By identifying a "Physical Violation," the model is essentially saying: *"Given the state I just saw ($z_t$) and the action I took ($a_t$), if the laws of physics were obeyed, the agent would not be behind that wall."*
- **World Modeling as Simulation:** Because LeWM can simulate future states in a latent space without actually executing them in the real world (used for Model Predictive Control), it possesses the "imagination" required for Level 3 tasks. It can generate "what if" trajectories to evaluate physical plausibility.

### Note on expert policy behavior and observed collection statistics

The `GlitchedHueExpertPolicy` encodes the experimental protocol directly: it is
teleport-aware in blue episodes and falls back to standard door navigation in
green episodes.

**Green room (`teleport.enabled = 0`)**

The policy reads `teleport.enabled` from the environment's variation space at
every step. When it is `0`, `teleport_available` is unconditionally `False` and
the policy immediately delegates to `_door_or_target_waypoint()` — the same
two-stage door-then-target logic as the base `ExpertPolicy`. Independently,
`env.step()` skips the proximity check entirely when `teleport.enabled = 0`, and
`_render_frame()` does not draw the white marker. The result is structurally
guaranteed: **zero teleportation events in green episodes**, as confirmed by the
`glitched_hue_tworoom_half` statistics (0 out of 10,000 episodes).

**Blue room (`teleport.enabled = 1`)**

The policy navigates toward the teleport pixel at `[56, 112]` — the centre of
Room 1 — whenever two conditions hold simultaneously:

1. The target is in the opposite room (wall crossing is required).
2. The pixel is within `1.5 ×` the distance to the nearest door (the
   `teleport_preference` multiplier).

When the agent comes within the 10 px `teleport.radius`, `env.step()` calls
`_mirror_position()` and sets `_teleported_this_episode = True`, after which the
policy reverts to door navigation for the remainder of that episode.

**Why only ~29 % of blue episodes use the teleport**

The pixel is fixed on the left side of the wall (Room 1). This makes the
shortcut geometrically one-directional:

- If agent and target spawn in the **same room** → direct navigation, no wall
  crossing, teleport never considered.
- If the agent spawns in **Room 2** and the target is in Room 1 → the pixel is
  on the far side of the wall and unreachable without already crossing it; policy
  falls back to the door.
- Only when the agent spawns in **Room 1** with the target in Room 2 is the
  pixel on the agent's side and the shortcut available.

With uniformly random agent/target positions, same-room and wrong-side spawns
account for the majority of episodes. Adding `action_noise = 0.5` (the `_half`
variant) means some agents that do approach the 10 px marker still miss it.
Together these factors produce the observed **2,889 / 10,000 (28.9 %)** teleport
rate in the blue condition.

**The confound is structurally sound**

The training corpus presents a perfect spurious correlation:

| Condition | Teleport events | Teleport marker visible |
|---|---:|---|
| Blue room | 2,889 (28.9 %) | Yes |
| Green room | 0 (0.0 %) | No |

Every teleportation the world model ever observes during training co-occurs with
a blue background. A model that binds teleport mechanics to hue rather than to
the pixel marker will fail the VoE probe at evaluation time — which is precisely
the causal disentanglement the experiment is designed to detect.

### Note on color choice in the Glitched Hue setup

For the Glitched Hue experiment, the choice of room colors matters because the
agent is rendered in **red** while JEPA training is primarily pixel-based. If we
used a red room, the agent would have lower visual contrast in that condition,
creating an unintended confound: room hue would also change how easy the agent
is to localize. Using **blue/green** keeps the hue manipulation global while
preserving clear visibility of the red agent in both rooms.

### 3. Why it isn't strictly Level 1

A Level 1 system (Association/Seeing) would only detect that a teleportation event is **statistically rare** (visual novelty). However, the authors demonstrate that LeWM distinguishes between **visual surprise** (color changes) and **physical surprise** (teleporting).

- A Level 1 model might be surprised by both equally because both are "unseen" in training.
- LeWM is significantly *more* surprised by the physical violation, proving it has moved beyond association to understand the **causal constraints** (the "mechanism") of the room.

### Summary Table

| Level | Name | LeWM VoE Implementation |
| :--- | :--- | :--- |
| Level 1 | Association | No. It distinguishes physical "illogic" from mere visual novelty. |
| Level 2 | Intervention | **Yes (Primary).** It predicts the outcome of actions ($a_t$) and uses the prediction error as a surprise signal. |
| Level 3 | Counterfactuals | **Conceptual.** It uses its internal "causal" world model to reject observations that contradict its "imagined" physical reality. |

**Conclusion:** The VoE paradigm in LeWM is a **Level 2 validation of an emergent causal model**. It proves that the model hasn't just learned what things *look* like, but how the world *works* under intervention, which is the prerequisite for true counterfactual reasoning.