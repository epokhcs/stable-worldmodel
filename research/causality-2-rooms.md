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