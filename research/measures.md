# Measuring Counterfactual Success in LeWorldModel

To distinguish between "fancy interpolation" (Ladder 2) and genuine counterfactual reasoning (Ladder 3), I propose using the following metrics. These tests verify if the **LeWorldModel (LeWM)**'s internal "imagination" is truly constrained by a **Structural Causal Model (SCM)** rather than just a library of memorized trajectories.

My approach is built around Pearl's **Abduction-Action-Prediction (AAP)** cycle.

### 1. The AAP Consistency Metric
I propose a **Counterfactual Consistency Metric** to measure if the model can use specific factual observations to constrain hypothetical simulations.

*   **Step 1: Abduction ($z \mid \text{obs}$):** Given a factual video where an agent fails to push a block (because the block has a hidden property like "heaviness"), I expect the model to encode this property into its latent state $z$.
*   **Step 2: Action ($\text{do}(a)$):** I then intervene by changing the action (e.g., "What if I pushed with double the force?").
*   **Step 3: Prediction ($z_{\text{CF}}$):** The model predicts the counterfactual outcome.

**The Metric:** I compare the counterfactual prediction ($z_{\text{CF}}$) against a "Ladder 2" prediction that lacks access to the factual evidence.
*   **Consistency Score:** $\text{Similarity}(z_{\text{CF}}, \text{Ground Truth Counterfactual})$.
*   If the model predicts the block moves in the counterfactual only when it has "observed" the weight correctly in the factual, I consider it to be performing true Level 3 reasoning.

### 2. The "Structural Invariance" Metric
I propose the **Structural Invariance** metric to detect "bleeding"—where changing one causal variable (like a door being open) inadvertently changes unrelated variables (like the color of the floor). This ensures the model has learned **Independent Causal Mechanisms (ICM)**.

*   **The Test:** I perform a **Latent Intervention** on a single coordinate (e.g., counterfactually changing the agent’s $x$-position).
*   **The Metric:** I measure the **Partial Derivative** or **Mutual Information** change in unrelated latent dimensions.

$$\text{Invariance Error} = \sum |z_{unrelated}^{factual} - z_{unrelated}^{counterfactual}|$$

A "Level 3" model should have an Invariance Error near zero. If the error is high, I conclude the model is merely interpolating between known "blobs" of data rather than respecting structural independence.

### 3. "Cross-World" Probabilistic Consistency
I define a **Cross-World Probabilistic Consistency** metric to check if factual evidence (like wind blowing) correctly "collapses" uncertainty in the counterfactual world.

In a true causal model, the following should hold:
$$P(y_x \mid x', y') \neq P(y \mid \text{do}(x))$$

*   **The Logic:** I expect the counterfactual probability (given evidence $y'$) to be more "certain" than a simple intervention probability.
*   **The Metric:** I measure the **Entropy** of the prediction. 
*   If the model is only performing Ladder 2 operations, its prediction of "where the ball goes" will be identical regardless of whether it "saw" the wind blowing.
*   If it is Ladder 3, the factual evidence should "collapse" the uncertainty in the hypothetical world, leading to a **lower entropy (higher confidence)** prediction than a blind intervention.

### Conclusion on Feasibility
I find that the **SIGReg** (Sketched-Isotropic-Gaussian Regularizer) in LeWM is a powerful tool for this approach. By forcing the latents into an isotropic Gaussian distribution, it naturally discourages the causal entanglement that usually plagues world models. 

If my proposed metrics show that LeWM's latent state for "Room Geometry" remains static while I counterfactually move the "Agent State," I will have successfully proven that it is not just interpolating pixel-sequences, but is instead simulating a **Structural Causal Model** of the world.