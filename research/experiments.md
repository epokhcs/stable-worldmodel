# Proposed Counterfactual Experiments for LeWorldModel

I propose the following experiments to better distinguish **Ladder 2 (Intervention)** from **Ladder 3 (Counterfactuals)** within the **LeWorldModel (LeWM)** framework. In Pearl’s causality, Level 3 reasoning requires a three-step process: **Abduction** (inferring the hidden state from evidence), **Action** (changing a variable), and **Prediction** (simulating the consequence). 

My proposal focuses on testing whether LeWM treats the environment as a set of independent, manipulable causal mechanisms rather than mere associations.

### 1. The "Sliding Door" Test (Abduction & Counterfactual Planning)
I propose this experiment to test whether the model can perform true abduction into the latent space.
* **The Mechanism:**
    * **Abduction:** The model observes a closed door and must abduce into the latent state $z_t$ that the binary state `door=closed`.
    * **Action:** I will manually perturb the latent $z_t$ to flip the dimension representing the door to `open`.
    * **Prediction:** The predictor rollouts should then show the agent successfully entering the next room.
* **Feasibility:** Since LeWM's latent space captures spatial structure and physical constraints, and SIGReg helps factorize environment dynamics, I expect the model to learn that the "door state" is a discrete component of the environment.
* **Refinement:** To make this a more rigorous Level 3 test, I will create a context where the agent **fails** to pass through the door, then ask: *"Given I failed to pass, what would have happened if I had pushed the 'Open' button 5 steps ago?"* This forces the model to re-simulate the past with a changed antecedent.

### 2. "Twin" Trajectories (Causal Path Independence)
I propose a "Counterfactual Invariance" test to check if the model is truly Markovian or if it is over-fitting to specific histories.
* **The Mechanism:** This tests **Counterfactual Identity**. If the agent reaches point $X$ in Room 2, I will verify if the counterfactual *"If I had taken the North door instead of the South door"* yields an identical current position $z_t$, despite having a different causal history.
* **LeWM Check:** Given that LeWM achieves "temporal latent path straightening," I expect the model to map these complex histories into smooth, equivalent latent trajectories.
* **The Test:** I will check if the **surprise metric** (prediction error) increases if I swap the histories. If the model functions at Ladder 2, it should only care about the current $z_t$. If it is capable of Ladder 3 reasoning, it should maintain the "memory" of the alternative path while still predicting future states correctly.

### 3. "Movable Perturbations" (Disentangled Counterfactuals)
I propose what I consider the strongest test for LeWM: evaluating its ability to separate **Agent Location** from **Object Location**.
* **The Probe:**
    * **Observation:** Agent is at $(x_1, y_1)$, Object (Cube) is at $(x_2, y_2)$.
    * **Counterfactual:** *"What if the cube were at $(x_3, y_3)$ instead?"*
* **The Reasoning:** I want to see if the model can keep the "background" (the room geometry) constant while only shifting the latent dimensions associated with the cube.
* **Verification:** If the model's predictor respects room boundaries for the agent but accepts the new, manually-injected cube position, it has achieved **Structural Disentanglement**. Existing results on Push-T and OGBench-Cube suggest the model already has the latent sensitivity required for this experiment.

### 4. The "Glitched Hue" Test (Causal Disentanglement)
I propose a "Glitched Hue" setup to test for **Causal Disentanglement**, which is a prerequisite for Level 3 reasoning. This experiment specifically probes whether the model can distinguish between a **spurious correlation** (the global hue) and a **causal mechanism** (the teleport pixel).

I map this experiment directly to **Pearl’s Ladder** framework:
*   **Confounder ($W$):** Room Hue (Red/Blue).
*   **Cause ($X$):** Presence of the 1x1 white "Teleport Pixel."
*   **Effect ($Y$):** Agent jumping from Room 1 to Room 2.

If, during training, teleportation *only* occurred in Blue rooms, a Ladder 1 or 2 model might learn the association: $\text{Blue} \rightarrow \text{Jump}$. I want to verify if the model understands that the Pixel is the *true* cause, independent of the Hue. To prove Level 3 reasoning, I perform the **Abduction-Action-Prediction (AAP)** cycle:

1.  **Abduction:** Provide the model with a factual observation where the room is Blue and the agent jumps. I expect the model to abduce that both the Hue and the Pixel are present in the latent state $z$.
2.  **Action (Counterfactual Intervention):** In the latent space, I manually "glitch" the Hue dimension from Blue to Red, but leave the Pixel dimension untouched.
3.  **Prediction:** I ask the model to predict the next state. 
    *   **Level 2 Failure:** The model predicts the agent *stays* in Room 1 (interpolation failure).
    *   **Level 3 Success:** The model predicts the agent *still jumps* to Room 2. This proves it has learned the **Independent Causal Mechanism** $(\text{Pixel} \rightarrow \text{Jump})$ regardless of the global visual context.

**Research Context:** Note that the LeWM paper already performs a "light" version of this experiment in its **Violation-of-Expectation (VoE)** tests.
*   **Visual Surprise (Control):** The authors showed that when the agent's **color changes** unexpectedly, the model's prediction error (Surprise) remains **low**.
*   **Physical Violation:** When the agent **teleports** (a physical jump), the Surprise metric **spikes significantly**.

Because LeWM "ignores" the color change while "detecting" the teleportation, it suggests the latent space has already disentangled **visual appearance** from **physical dynamics**. I believe this is driven by the **SIGReg**, which pressures the model to represent the "Hue" and the "Pixel" in **orthogonal (independent) dimensions**.

### Summary of Feasibility
I believe these experiments are highly compatible with the current LeWM architecture because:
1.  **SIGReg** ensures the latent space is well-distributed and non-collapsed, allowing for manual interventions.
2.  **Probing Results** confirm that physical variables (like positions) are explicitly encoded and recoverable.
3.  The **Surprise Metric** provides a built-in way to verify if my proposed counterfactual branches remain "physically plausible" to the model.