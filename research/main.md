In the **LeWorldModel (LeWM)** paper, the authors implement the **Violation-of-Expectation (VoE)** paradigm—a method inspired by developmental psychology used to study infant cognition—to evaluate whether the model's latent space has internalized the physical laws of its environment.

In the **Two Room** experiment (and others like Push-T), the mechanism is implemented as follows:

### 1. The "Surprise" Metric
The model’s level of "surprise" is quantified using the **latent prediction error** $L_{\text{pred}}$. 
* Because LeWM is a Joint-Embedding Predictive Architecture (JEPA), it predicts the next latent embedding $\hat{z}_{t+1}$ based on the current embedding $z_t$ and an action $a_t$.
* The "surprise" is the Mean Squared Error (MSE) between this predicted embedding and the actual encoded embedding $z_{t+1}$ of the next observation: 

$$
\text{Surprise} = \| \hat{z}_{t+1} - z_{t+1} \|_2^2
$$

### 2. Test Scenarios in Two Room
To distinguish between **physical understanding** and simple **visual novelty detection**, the authors present the model with three types of sequences:
* **Physically Expected**: Normal trajectories where the agent moves through the door to transition between rooms.
* **Physical Violation (Unphysical)**: Scenarios that violate environmental constraints, such as the agent **teleporting through a wall** instead of using the door.
* **Visual Surprise (Control)**: Scenarios that are visually unusual but physically possible, such as the agent **changing color** while moving.


### 3. Implementation Mechanism
The core of the mechanism relies on the **SIGReg (Sketched-Isotropic-Gaussian Regularizer)**. By enforcing that latent embeddings follow a Gaussian distribution and preventing representation collapse, the regularizer ensures that the latent space is well-structured and that prediction errors are meaningful. 

The model is shown these sequences, and its prediction error is monitored. If the model has learned the physical "logic" of the Two Room environment (e.g., that walls are solid and rooms are connected only by doors), it should show:
* **Low Error** for expected trajectories.
* **High Error (Surprise)** for physical violations like teleportation.
* **Lower Error** for visual-only surprises (like color changes), indicating the model recognizes them as physically plausible despite the visual change.

### Key Finding
The authors found that LeWM assigns **significantly higher surprise** to physical violations than to visual ones. This confirms that the model does not just learn visual patterns but develops an **emergent understanding of physical dynamics** and environmental constraints directly from raw pixels.