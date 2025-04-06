# HMN – Hybrid Local–Global Modulated Neuron with Multi-Modal Attention: A Theoretical Framework

## Table of Contents

1. Abstract  
2. Introduction  
  2.1. Motivation and Problem Statement  
  2.2. Biological and Computational Inspirations  
  2.3. Core Contributions and Paper Organization  
3. Background and Related Work  
  3.1. Local Synaptic Plasticity: Hebbian Rules, STDP, and Probabilistic Dynamics  
  3.2. Global Neuromodulation: Reinforcement Signals, Attentional Gain, and Meta-Learning  
  3.3. Attention Mechanisms in Neural Systems: From Biology to Deep Learning  
  3.4. Limitations of Centralized Learning Paradigms and Comparison to Alternative Approaches  
4. Model Architecture and Methods: The Hybrid Modulated Neuron (HMN)  
  4.1. Theoretical Framework and Key Assumptions  
  4.2. Local Processing Unit: Probabilistic Synaptic Dynamics  
    4.2.1. Neuronal Activation with Stochasticity  
    4.2.2. Composite Eligibility Traces: Capturing Multiple Timescales  
    4.2.3. Probabilistic Synaptic State Transitions and Oscillatory Gating  
  4.3. Global Neuromodulatory Integration and Dual Meta-Learning  
    4.3.1. Aggregation of Multi-Factor Neuromodulatory Signals  
    4.3.2. Dual Meta-Learning of Local and Global Learning Rates  
    4.3.3. Phase-Locked Weight Updates for Temporal Coherence  
  4.4. Multi-Modal Attention Mechanism  
    4.4.1. Local Attentional Modulation of Eligibility Traces  
    4.4.2. Global Attentional Gating of Neuromodulatory Signals  
  **Figures:**  
   - **Figure 1:** Conceptual Overview of the HMN Architecture (Mermaid Diagram)  
   - **Figure 2:** Detailed Schematic of the HMN Showing Signal Flow and Modulation Points  
5. Hypothesized Capabilities and Potential Applications  
  5.1. Adaptive Agents in Complex Reinforcement Learning Environments  
  5.2. Robust Continuous and Lifelong Learning Systems  
  5.3. A Computational Model for Adaptive Decision-Making Studies  
6. Discussion  
  6.1. Synergistic Advantages of Integrated Local–Global Dynamics  
  6.2. Comparison with Centralized Backpropagation and Other Bio-Inspired Models  
  6.3. Theoretical Implications for Credit Assignment Mechanisms  
  6.4. Complexity, Parameter Sensitivity, and Scalability Considerations  
7. Conclusion and Future Work  
  7.1. Summary of Theoretical Contributions and Broader Impact  
  7.2. Future Directions: Theoretical Analysis, Simulation, and Neuromorphic Implementation  
8. Acknowledgements  
9. References  
10. Appendix  
  10.1. Example Functional Forms  
  10.2. Considerations for Meta-Learning Gradient Approximation  
  10.3. Algorithm Pseudocode for HMN Update  

---

## 1. Abstract

Biological neural systems exhibit remarkable learning capabilities underpinned by the complex interplay of local synaptic plasticity and global neuromodulatory signals. Standard artificial neural networks—primarily reliant on centralized error backpropagation—often lack the biologically inspired adaptability seen in nature. We introduce the Hybrid Modulated Neuron (HMN), a novel theoretical framework that integrates rapid, local, probabilistic synaptic plasticity with slower, globally synthesized neuromodulatory feedback via dual meta-learning loops. The HMN framework unifies composite eligibility traces capturing multiple timescales, phase-locked weight updates synchronized by global oscillations, and a dual attention mechanism that operates both on local eligibility traces and on global modulatory signals. In this revision, we have clarified key abstract components: the input embedding \( \mathbf{h}_i(t) \) is obtained via a learned embedding layer mapping raw inputs \( x_i(t) \) into a feature space, and the local context vector \( \mathbf{c}_j(t) \) is computed as an exponentially decaying moving average of the neuron’s recent activations. We maintain consistent notation throughout: \( \Delta w^* \) denotes the initial computed update, \( \Delta w^\dagger \) is the phase-modulated update, and \( \Delta w \) is the final probabilistic update. We further discuss the rationale for oscillatory gating, potential biological correlates (e.g., neuromodulatory centers such as the VTA or LC), and strategies for managing model complexity.  
*Keywords:* Synaptic Plasticity, Neuromodulation, Meta-Learning, Attention Mechanisms, Probabilistic Neuron, Local Learning Rules, Credit Assignment, Computational Neuroscience, Reinforcement Learning.

---

## 2. Introduction

### 2.1. Motivation and Problem Statement

Biological neural systems excel at continuous learning and adaptive behavior through a combination of local synaptic modifications (e.g., Hebbian learning, STDP) and global neuromodulatory signals that coordinate learning across distributed networks (Chklovskii et al., 2004; Poo et al., 2016). In contrast, conventional artificial neural networks rely predominantly on centralized error backpropagation, a mechanism that raises questions regarding biological plausibility and decentralization in dynamic environments (Lillicrap et al., 2016; Bengio et al., 2015). This paper addresses two central questions:

- **How can computationally tractable models integrate fast, local synaptic learning with slower, multifaceted global neuromodulatory signals to achieve robust adaptive behavior?**
- **Can meta-learning and attention mechanisms, applied at both local and global levels, dynamically balance these influences to solve the spatio-temporal credit assignment problem in a biologically inspired manner?**

### 2.2. Biological and Computational Inspirations

The HMN model draws inspiration from several key phenomena and paradigms:

1. **Local Synaptic Plasticity:** Hebbian learning (“cells that fire together, wire together”), STDP, and eligibility traces allow for immediate as well as delayed synaptic modifications (Hebb, 1949; Markram et al., 1997; Izhikevich, 2007).
2. **Global Neuromodulation:** Neuromodulatory signals (e.g., dopamine, acetylcholine, norepinephrine) modulate synaptic plasticity across large neural ensembles based on reward, novelty, or uncertainty (Schultz, 1998; Doya, 2002; Yu & Dayan, 2005).
3. **Meta-Learning:** The capacity to “learn how to learn” motivates the dynamic adjustment of learning parameters—a phenomenon supported by both biological evidence and recent computational studies (Finn et al., 2017; Schmidhuber, 1992).
4. **Attention Mechanisms:** Both biological and deep learning attention systems enable the selective processing of relevant information. HMN innovates by applying attention to both synaptic eligibility and neuromodulatory signals.
5. **Neural Oscillations:** Oscillatory activity (e.g., theta, gamma rhythms) may serve as a gating mechanism for plasticity, contributing to temporal binding and optimal update timing (Buzsáki & Draguhn, 2004).

### 2.3. Core Contributions and Paper Organization

This paper’s primary contributions are as follows:

1. **Unified HMN Framework:** We propose a comprehensive theoretical formulation that unifies local probabilistic plasticity, global neuromodulation, dual meta-learning, and dual attention mechanisms within a single neuron model.
2. **Multi-Timescale Integration:** Our framework explicitly models fast and slow eligibility traces coordinated by oscillatory gating to capture multiple timescales of plasticity.
3. **Dual Adaptive Learning Rates:** HMN incorporates dual meta-learning loops to dynamically adjust local (\(\eta_{\text{local}}\)) and global (\(\eta_{\text{global}}\)) learning rates, enabling a balance between rapid adaptation and long-term stability.
4. **Dual Attention Mechanisms:** A two-tiered attention process refines local eligibility traces via a context vector (with \( \mathbf{h}_i(t) \) derived from a learned embedding layer and \( \mathbf{c}*j(t) \) as a decayed moving average of \( z_j(t) \)) and simultaneously gates global neuromodulatory signals using a global context \( C*{\text{global}}(t) \) (e.g., average network activity).
5. **Clarification of Abstract Components and Notation:** We offer concrete examples for previously abstract variables and maintain consistent notation throughout—using \( \Delta w^* \) for the initial computed update, \( \Delta w^\dagger \) for the phase-modulated update, and \( \Delta w \) for the final probabilistic weight change.
6. **Foundational Basis for Future Work:** By addressing theoretical and practical challenges—including parameter sensitivity and computational cost—the HMN framework lays the groundwork for future empirical validation and neuromorphic implementation.

The remainder of the paper is organized as follows. Section 3 reviews relevant background and related work while explicitly contrasting HMN with existing models. Section 4 details the HMN formulation and key assumptions. Section 5 maps individual mechanisms to their functional benefits and potential applications. Section 6 discusses theoretical implications, limitations, and computational challenges, with a focus on future ablation studies and self-regulatory mechanisms. Section 7 concludes with future research directions. Sections 8–10 provide acknowledgements, references, and supplementary materials.

---

## 3. Background and Related Work

### 3.1. Local Synaptic Plasticity: Hebbian Rules, STDP, and Probabilistic Dynamics

Hebbian learning (Hebb, 1949) and spike-timing-dependent plasticity (STDP) (Markram et al., 1997; Bi & Poo, 1998) are well-established mechanisms that drive synaptic modifications based on the correlation of neuronal activity. Eligibility traces, which mark synapses for later modification, are an essential element bridging immediate activity with delayed reinforcement signals (Sutton & Barto, 1998; Izhikevich, 2007). HMN builds on these ideas by employing composite eligibility traces that capture both rapid and prolonged dynamics in a probabilistic update framework.

### 3.2. Global Neuromodulation: Reinforcement Signals, Attentional Gain, and Meta-Learning

Neuromodulatory systems (e.g., those driven by dopamine) are known to modulate synaptic plasticity on a network-wide scale (Schultz, 1998; Doya, 2002). While prior models typically employ a single three-factor rule, HMN aggregates multiple neuromodulatory signals into a composite global factor \( G(t) \) (see Eq. 6) and refines this signal through a dedicated global attention mechanism. Additionally, HMN employs dual meta-learning to adapt both local and global learning rates dynamically, a capability absent in simpler models.

### 3.3. Attention Mechanisms in Neural Systems: From Biology to Deep Learning

Attention mechanisms in deep learning (Bahdanau et al., 2014; Vaswani et al., 2017) typically weight input or output signals. In contrast, HMN applies attention at two levels: locally to modulate eligibility traces based on neuron-specific context, and globally to gate neuromodulatory signals using broader network or task context. This dual application of attention distinguishes HMN from approaches such as Equilibrium Propagation or Feedback Alignment.

### 3.4. Limitations of Centralized Learning Paradigms and Comparison to Alternatives

Centralized error backpropagation (Rumelhart et al., 1986) has achieved remarkable success in artificial networks but is limited by its non-local error signals and weight transport issues. HMN offers a decentralized alternative by combining local processing with global modulatory signals that might be produced by dedicated neuromodulatory centers (e.g., VTA, LC). The dual attention and meta-learning mechanisms potentially yield superior adaptability and spatio-temporal credit assignment compared to existing energy-based or feedback alignment models.

---

## 4. Model Architecture and Methods: The Hybrid Modulated Neuron (HMN)

### 4.1. Theoretical Framework and Key Assumptions

HMN posits that effective learning arises from the dynamic interplay between fast, local synaptic modifications and slower, globally informed neuromodulatory signals. Our framework is built on the following key assumptions:

- **Input Representation:** Each neuron receives input signals \( x_i(t) \) (with delays \( \tau_{ij} \)) and generates continuous activations \( z_j(t) \) (analogous to firing rates). In our formulation, spiking details are abstracted to focus on rate-based dynamics.
- **Global Signals:** Neurons have access to global evaluative signals \( E_k(t) \) (e.g., reward, uncertainty, novelty), a global context \( C_{\text{global}}(t) \) (such as the average network activity or prediction error), and an oscillatory phase \( \Phi(t) \). While the precise biological origin of these signals is abstracted, potential sources include neuromodulatory centers like the VTA for reward or the LC for uncertainty.
- **Meta-Learning Gradients:** The meta-learning mechanism relies on approximations of the gradient \( \nabla_{\eta} L_{\text{meta}} \) (using finite differences, SPSA, or REINFORCE-based methods) to update learning rates in a biologically inspired manner (see Appendix 10.2).

### 4.2. Local Processing Unit: Probabilistic Synaptic Dynamics

#### 4.2.1. Neuronal Activation with Stochasticity

The activation of neuron \( j \) is computed as:
\[
z_j(t) = f\Biggl( \sum_i w_{ij}(t)\, x_i(t-\tau_{ij}) + b_j(t) + \epsilon_j(t) \Biggr), \quad (1)
\]
where:

- \( f(\cdot) \) is a nonlinear function (e.g., ReLU or sigmoid),
- \( w_{ij}(t) \) are synaptic weights,
- \( x_i(t-\tau_{ij}) \) are input signals,
- \( \tau_{ij} \) are transmission delays,
- \( b_j(t) \) is an adaptive bias, and
- \( \epsilon_j(t) \) is a stochastic noise term (e.g., drawn from \( \mathcal{N}(0,\sigma_j^2) \)).

#### 4.2.2. Composite Eligibility Traces: Capturing Multiple Timescales

Each synapse \( (i,j) \) maintains a composite eligibility trace:
\[
e_{ij}(t) = \psi_{\text{fast}}\big(x_i(t), z_j(t), t\big) + \psi_{\text{slow}}\big(x_i(t), z_j(t), t\big), \quad (2)
\]
where \( \psi_{\text{fast}} \) and \( \psi_{\text{slow}} \) capture fast and slow dynamics, respectively—often implemented via differential equations or event-triggered updates with decay constants \( \tau_{\text{fast}} \) and \( \tau_{\text{slow}} \) (see Appendix 10.1).

#### 4.2.3. Probabilistic Synaptic State Transitions and Oscillatory Gating

The weight update is computed in three steps:

1. **Initial Update Calculation:**  
   We compute the preliminary update as:
   \[
   \Delta w^**{ij}(t) = \eta*{\text{local}}(t)\, \tilde{e}*{ij}(t) + \eta*{\text{global}}(t)\, G'(t)\, \tilde{e}*{ij}(t), \quad (3)
   \]
   where \( \tilde{e}*{ij}(t) \) is the eligibility trace modulated by local attention (see Section 4.4.1) and \( G'(t) \) is the global modulatory signal after global attention gating (Section 4.4.2).

2. **Phase Modulation:**  
   The update is then modulated by a global oscillatory phase:
   \[
   \Delta w^\dagger_{ij}(t) = \Delta w^**{ij}(t) \cdot \max\Bigl(0, \cos\bigl(\Phi(t) - \phi*{ij}\bigr)\Bigr), \quad (4)
   \]
   where \( \Phi(t) = \Omega t + \phi_0 \) is the global oscillatory phase and \( \phi_{ij} \) is a synapse-specific phase offset. The \( \max(0, \cdot) \) operation acts as a gating function, ensuring that updates occur only during the “active” phase of the oscillation. This choice is motivated by biological observations that plasticity is often restricted to certain phases of network oscillations.

3. **Probabilistic Update Application:**  
   Finally, the weight change is applied probabilistically:
   \[
   \Delta w_{ij}(t) = \begin{cases} \Delta w^\dagger_{ij}(t) & \text{with probability } P_{ij}(t) \\ 0 & \text{otherwise,} \end{cases} \quad (5)
   \]
   with
   \[
   P_{ij}(t) = \sigma\Bigl( \beta_p \Bigl( \big|\Delta w^\dagger_{ij}(t)\big| - \theta_p \Bigr) \Bigr),
   \]
   where \( \sigma(\cdot) \) is the logistic sigmoid, \( \beta_p \) a sensitivity parameter, and \( \theta_p \) a threshold.

### 4.3. Global Neuromodulatory Integration and Dual Meta-Learning

#### 4.3.1. Aggregation of Multi-Factor Neuromodulatory Signals

The global modulatory signal is given by:
\[
G(t) = \mathcal{M}\Bigl( w_{\text{reward}}\, E_{\text{reward}}(t),\; w_{\text{uncert}}\, E_{\text{uncertainty}}(t),\; w_{\text{novel}}\, E_{\text{novelty}}(t), \dots \Bigr), \quad (6)
\]
where each \( E_k(t) \) is computed independently (for example, as a temporal-difference error or prediction variance), and \( \mathcal{M}(\cdot) \) is a weighted sum operator.

#### 4.3.2. Dual Meta-Learning of Local and Global Learning Rates

The learning rates are updated using a meta-objective \( L_{\text{meta}}(t) \):
\[
\begin{aligned}
\eta_{\text{local}}(t+1) &= \eta_{\text{local}}(t) - \alpha_{\text{meta},1}\, \nabla_{\eta_{\text{local}}} L_{\text{meta}}(t), \\
\eta_{\text{global}}(t+1) &= \eta_{\text{global}}(t) - \alpha_{\text{meta},2}\, \nabla_{\eta_{\text{global}}} L_{\text{meta}}(t),
\end{aligned} \quad (7)
\]
where \( \alpha_{\text{meta},1} \) and \( \alpha_{\text{meta},2} \) are meta-learning rates. Approximations of \( \nabla_{\eta} L_{\text{meta}} \) (via finite differences, SPSA, or REINFORCE-based techniques) allow online adaptation in a biologically inspired fashion (see Appendix 10.2).

#### 4.3.3. Phase-Locked Weight Updates for Temporal Coherence

Phase-locking (as in Eq. 4) ensures that synaptic updates are confined to specific oscillatory windows. This mechanism is hypothesized to align synaptic plasticity with optimal network states for encoding or consolidation, reflecting experimental evidence that plasticity is phase-dependent. We note that while neuromodulation and phase-locking are individually supported by biological data, integrating them here abstracts potential distributed processing across local microcircuits.

### 4.4. Multi-Modal Attention Mechanism

#### 4.4.1. Local Attentional Modulation of Eligibility Traces

Local attention refines the composite eligibility trace:
\[
\tilde{e}*{ij}(t) = \alpha*{ij}(t)\, e_{ij}(t), \quad (8)
\]
with attention weights computed as:
\[
\alpha_{ij}(t) = \frac{\exp\Bigl(\beta_a\, s_{ij}(t)\Bigr)}{\sum_l \exp\Bigl(\beta_a\, s_{lj}(t)\Bigr)}, \quad (9)
\]
where the similarity score \( s_{ij}(t) \) is defined as:
\[
s_{ij}(t) = g\bigl( \mathbf{h}_i(t), \mathbf{c}_j(t) \bigr).
\]
Here:

- \( \mathbf{h}*i(t) \) is a feature embedding of input \( x_i(t-\tau*{ij}) \) obtained via a learned embedding layer,
- \( \mathbf{c}_j(t) \) is a local context vector computed as an exponentially decaying moving average of \( z_j(t) \), and
- \( g(\cdot,\cdot) \) is a similarity function (e.g., cosine similarity).

Providing this concrete formulation strengthens the interpretation of the local attention mechanism.

#### 4.4.2. Global Attentional Gating of Neuromodulatory Signals

Global attention modulates the aggregated neuromodulatory signal:
\[
G'(t) = \sum_k \gamma_k(t)\, w_k\, E_k(t), \quad (10)
\]
with gating weights given by:
\[
\gamma_k(t) = \frac{\exp\Bigl(\beta_g\, h\bigl(E_k(t), C_{\text{global}}(t)\bigr)\Bigr)}{\sum_m \exp\Bigl(\beta_g\, h\bigl(E_m(t), C_{\text{global}}(t)\bigr)\Bigr)}, \quad (11)
\]
where:

- \( C_{\text{global}}(t) \) is a global context variable (for example, the network-wide average firing rate or prediction error variance),
- \( h(\cdot,\cdot) \) is a function evaluating the relevance between each neuromodulatory signal and the global context.
A concrete default example might be setting \( C_{\text{global}}(t) \) as the arithmetic mean \( \frac{1}{N}\sum_k z_k(t) \) and letting \( h \) be a dot product if \( E_k(t) \) and \( C_{\text{global}}(t) \) are represented as vectors.

---

## 5. Hypothesized Capabilities and Potential Applications

### 5.1. Adaptive Agents in Complex Reinforcement Learning Environments

The fast local plasticity term \( \eta_{\text{local}}\, \tilde{e}*{ij}(t) \) enables rapid adaptation to new stimuli, while the globally modulated component \( \eta*{\text{global}}\, G'(t)\, \tilde{e}_{ij}(t) \) integrates evaluative signals (reward, uncertainty, novelty) for long-term learning. Combined with the probabilistic updates (Eq. 5) and dual attention filtering, HMN is designed to robustly balance exploration and exploitation in dynamic environments.

### 5.2. Robust Continuous and Lifelong Learning Systems

The dual meta-learning mechanism (Eq. 7) allows dynamic adjustment of local and global learning rates, which may mitigate catastrophic forgetting by balancing plasticity and stability. Furthermore, the multi-modal attention ensures that synaptic updates are context-dependent, a critical requirement for lifelong learning under changing conditions.

### 5.3. A Computational Model for Adaptive Decision-Making Studies

By linking synaptic-level processes (via eligibility traces and phase-locking) with network-level modulation (via global neuromodulation and meta-learning), HMN offers a framework for exploring complex credit assignment in temporally extended tasks. The dual attention mechanisms allow selective prioritization of both input signals and evaluative feedback, thereby offering a nuanced substrate for modeling adaptive decision-making.

---

## 6. Discussion

### 6.1. Synergistic Advantages of Integrated Local–Global Dynamics

We hypothesize that the synergy among local attention, phase-locking, global neuromodulation, and dual meta-learning creates a robust system that:

- **Rapidly Adapts** to new input patterns via fast local plasticity.
- **Efficiently Assigns Credit** by filtering eligibility traces (local attention) and temporally aligning updates (phase-locking).
- **Maintains Long-Term Stability** through global evaluative signals and adaptive learning rates.
- **Mitigates Noise** by employing probabilistic updates, which may also encourage beneficial exploration.

### 6.2. Comparison with Centralized Backpropagation and Other Bio-Inspired Models

While sharing the three-factor learning principle with earlier models (e.g., Izhikevich, 2007), HMN uniquely integrates:

- **Probabilistic Updates:** Adding stochasticity that can prevent overfitting.
- **Dual Meta-Learning:** Dynamically adjusting both local and global learning rates.
- **Composite Eligibility Traces and Dual Attention:** Providing a finer, context-dependent modulation.
- **Phase-Locking:** Aligning updates with optimal oscillatory phases.
These combined features may offer superior adaptability and spatio-temporal credit assignment relative to centralized backpropagation or simpler bio-inspired alternatives.

### 6.3. Theoretical Implications for Credit Assignment Mechanisms

HMN posits that effective credit assignment arises through:

- **Synaptic Tagging:** Marking synapses with eligibility traces.
- **Selective Gating:** Refining these tags using local attention.
- **Temporal Alignment:** Restricting updates to optimal oscillatory windows.
- **Outcome Evaluation:** Integrating global evaluative feedback.
- **Adaptive Regulation:** Adjusting the balance between rapid adaptation and long-term stability through meta-learning.
This layered approach may yield insights into solving the credit assignment problem across multiple timescales.

### 6.4. Complexity, Parameter Sensitivity, and Scalability Considerations

The HMN framework introduces numerous hyperparameters (e.g., \( \tau_{\text{fast}}, \tau_{\text{slow}}, \beta_p, \beta_a, \beta_g, \Omega, \phi_{ij} \)) and interacting mechanisms. We acknowledge that:

- **Theoretical Analysis:** Rigorous stability and convergence analyses for such a nonlinear, stochastic system are nontrivial.
- **Practical Tuning:** Extensive hyperparameter optimization (possibly via Bayesian or evolutionary methods) may be required, and some parameters might benefit from self-regulatory meta-learning mechanisms.
- **Computational Overhead:** Increased complexity must be balanced against potential benefits.
Future work will include systematic ablation studies to evaluate each component’s contribution and methods to reduce tuning complexity.

---

## 7. Conclusion and Future Work

### 7.1. Summary of Theoretical Contributions and Broader Impact

We have presented the Hybrid Modulated Neuron (HMN), a novel theoretical framework that unifies local probabilistic synaptic plasticity, global neuromodulation, dual meta-learning, dual attention mechanisms, and phase-locked weight updates. This revised framework clarifies abstract components, maintains consistent notation, and more carefully distinguishes between "biologically inspired" and "biologically plausible" elements. By integrating these diverse mechanisms, HMN lays a promising foundation for precise spatio-temporal credit assignment and robust adaptation in dynamic learning scenarios.

### 7.2. Future Directions: Theoretical Analysis, Simulation, and Neuromorphic Implementation

Future research should focus on:

1. **Empirical Validation:** Conducting simulation studies on dynamic reinforcement learning, continual learning, and temporal credit assignment tasks.
2. **Theoretical Analysis:** Developing rigorous analyses of simplified HMN variants to investigate stability, convergence, and component interactions.
3. **Hyperparameter Optimization:** Exploring systematic ablation studies and methods to co-adapt structural parameters (e.g., \( \phi_{ij}, \beta_a, \beta_g \)).
4. **Spiking Network Extensions:** Adapting HMN to spiking neural networks to improve biological realism.
5. **Neuromorphic Prototyping:** Implementing HMN on neuromorphic hardware to assess real-time performance and scalability.
6. **Distributed Architectures:** Investigating whether the integrated mechanisms are best implemented across microcircuits rather than within single neurons.

---

## 8. Acknowledgements

We thank our colleagues in computational neuroscience and machine learning for their invaluable discussions and insights. This work was supported by [Funding Agency Name(s)].

---

## 9. References

*Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.*  
*Bengio, Y. (2014). Towards biologically plausible deep learning. arXiv preprint arXiv:1407.1148.*  
*Bengio, Y., Lee, D. H., Bornschein, J., & Lin, Z. (2015). Towards biologically plausible deep learning. arXiv preprint arXiv:1502.04156.*  
*Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. Journal of Neuroscience, 18(24), 10464-10472.*  
*Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. Science, 304(5679), 1926-1929.*  
*Chklovskii, D. B., Mel, B. W., & Svoboda, K. (2004). Cortical rewiring and information storage. Nature, 431(7010), 782-788.*  
*Crick, F. (1989). The recent excitement about neural networks. Nature, 337(6203), 129-132.*  
*Doya, K. (2002). Metalearning and neuromodulation. Neural Networks, 15(4-6), 495-506.*  
*Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. In Proceedings of the 34th International Conference on Machine Learning (pp. 1126-1135).*  
*Hebb, D. O. (1949). The Organization of Behavior: A Neuropsychological Theory. Wiley.*  
*Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP and dopamine signaling. Cerebral Cortex, 17(10), 2443-2452.*  
*Lillicrap, T. P., Cownden, D., Tweed, D. B., & Akerman, C. J. (2016). Random synaptic feedback weights support error backpropagation for deep learning. Nature Communications, 7, 13276.*  
*Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. (1997). Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs. Science, 275(5297), 213-215.*  
*Moran, J., & Desimone, R. (1985). Selective attention gates visual processing in the extrastriate cortex. Science, 229(4715), 782-784.*  
*Poo, M., et al. (2016). What is memory? The present state of the engram. Biological Psychiatry, 80(4), 344-352.*  
*Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533-536.*  
*Schmidhuber, J. (1992). Learning to control fast-weight memories: an alternative to dynamic recurrent networks. In Advances in Neural Information Processing Systems (pp. 1-9).*  
*Schultz, W. (1998). Predictive reward signal of dopamine neurons. Journal of Neurophysiology, 80(1), 1-27.*  
*Scellier, B., & Bengio, Y. (2017). Equilibrium propagation: Bridging the gap between energy-based models and backpropagation. Frontiers in Computational Neuroscience, 11, 24.*  
*Sutton, R. S., & Barto, A. G. (1998). Reinforcement Learning: An Introduction. MIT Press.*  
*Vaswani, A., et al. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (pp. 5998-6008).*  
*Yu, A. J., & Dayan, P. (2005). Uncertainty, neuromodulation, and attention. Neuron, 46(4), 681-692.*

---

## 10. Appendix

### 10.1. Example Functional Forms

- **Activation Function \( f(\cdot) \):**  
  Example: ReLU: \( f(x) = \max(0, x) \) or Sigmoid: \( f(x) = \frac{1}{1 + e^{-x}} \).

- **Fast Eligibility Trace \( \psi_{\text{fast}} \):**  
  Example:  
  \(\psi_{\text{fast}}(x_i, z_j, t) = x_i(t) \cdot z_j(t) \cdot \exp\left(-\frac{t-t_{\text{last}}}{\tau_{\text{fast}}}\right).\)

- **Slow Eligibility Trace \( \psi_{\text{slow}} \):**  
  Example:  
  \(\psi_{\text{slow}}(x_i, z_j, t) = x_i(t) \cdot z_j(t) \cdot \exp\left(-\frac{t-t_{\text{last}}}{\tau_{\text{slow}}}\right).\)

- **Aggregation Function \( \mathcal{M}(\cdot) \):**  
  Example: Weighted sum:  
  \(\mathcal{M}(E_1, E_2, \dots) = \sum_k w_k E_k.\)

- **Similarity Function \( g(\cdot,\cdot) \):**  
  Example: Cosine similarity:  
  \(g(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}.\)

### 10.2. Considerations for Meta-Learning Gradient Approximation

Due to the complexity of computing \( \nabla_{\eta} L_{\text{meta}} \) exactly, practical implementations may rely on approximations such as finite differences, simultaneous perturbation stochastic approximation (SPSA), or REINFORCE-based techniques when \( L_{\text{meta}} \) is framed as a reinforcement learning objective. These methods offer a plausible route for online, biologically inspired meta-gradient estimation.

### 10.3. Algorithm Pseudocode for HMN Update

```python
# Pseudocode for HMN Weight Update at time t

for each synapse (i, j):
    # 1. Compute delayed input and activation
    x_delayed = get_input(i, t - tau_ij)  # Obtain input x_i(t) as a rate or continuous value
    z_j = f(sum_over_i(w_ij * x_delayed) + b_j + noise())  # Compute neuron j activation

    # 2. Compute eligibility traces
    psi_fast = compute_psi_fast(x_delayed, z_j, t)
    psi_slow = compute_psi_slow(x_delayed, z_j, t)
    e_ij = psi_fast + psi_slow

    # 3. Apply local attention using context vector c_j
    # h_i: feature embedding of input x_i(t); c_j: exponentially decayed average of z_j.
    s_ij = similarity(h_i, c_j)
    alpha_ij = softmax(beta_a * s_ij)  # Computed over all inputs for neuron j
    e_tilde = alpha_ij * e_ij  # Effective eligibility trace

    # 4. Compute global modulatory signal from multiple factors
    E_reward = get_reward_signal(t)
    E_uncert = get_uncertainty_signal(t)
    E_novel = get_novelty_signal(t)
    G = M(w_reward * E_reward, w_uncert * E_uncert, w_novel * E_novel, ...)

    # 5. Apply global attention using global context C_global
    for each modulatory component E_k:
        gamma_k = softmax(beta_g * h(E_k, C_global))
    G_prime = sum_over_k(gamma_k * w_k * E_k)

    # 6. Compute potential weight update using dual meta-learning rates
    delta_w_star = eta_local * e_tilde + eta_global * G_prime * e_tilde

    # 7. Apply phase-locking with a global oscillator
    phi = global_oscillator(t)  # e.g., Phi(t) = Omega * t + phi0
    delta_w_dagger = delta_w_star * max(0, cos(phi - phi_ij))  # phi_ij: learned phase offset

    # 8. Determine probabilistic update
    p_update = sigmoid(beta_p * (abs(delta_w_dagger) - theta_p))
    if random() < p_update:
        delta_w = delta_w_dagger
    else:
        delta_w = 0

    # 9. Update synaptic weight
    w_ij = w_ij + delta_w

# 10. Update meta-learning rates using approximated gradients from L_meta
eta_local = eta_local - alpha_meta1 * grad(L_meta, eta_local)
eta_global = eta_global - alpha_meta2 * grad(L_meta, eta_global)
```

*Notes:*  

- \( \Delta w^*_{ij} \) denotes the initial computed update (Eq. 3).  
- \( \Delta w^\dagger_{ij} \) represents the phase-modulated update (Eq. 4).  
- \( \Delta w_{ij} \) is the final probabilistic weight update (Eq. 5).  
- Functions such as `grad(L_meta, eta)` approximate the gradient of the meta-objective with respect to the learning rate.
