# Side Information Experiment: Specification


This document addresses the problem of **strategic classification with side information under non-uniform preferences**. Our experimental implementation is organized across four dedicated files, each described in detail below.

## Checklist of Requirements

- **Data Generation**
  - Base joint distribution over $(X, Y, Z, U)$ with $Y, Z, U \in \{\pm1\}$, $X \mid Y, Z, U \sim \mathcal{N}(\mu_{yzu}, \Sigma_{yzu})$.
  - `BaseDistribution` class specifies seed, PMF, conditional Gaussian params.
  - `PerturbedDistribution` subclass perturbs $\mathrm{corr}(Y, U)$, $\mathrm{corr}(Y, Z)$, supports named perturbation levels, observability $p$, and sampling with masked $Z$.
- **Models**
  - Two classifiers (Vanilla, Strategic): support missing $Z$, implement `fit` and `eval`, store weights/metrics.
  - Loss: logistic; metrics: logistic loss and accuracy ($\text{prediction} = \mathrm{sign}(\text{score})$).
- **Experiment**
  - `Experiment` class to run fits, store results, sweep $p$ and perturbation levels.
- **Visualization**
  - Per-$p$ plots across correlation axes, coloring for which model wins (loss and accuracy).

---

## Notation and Core Definitions

- **Variables:** $Y, Z, U \in \{ -1, +1\}$, $X \in \mathbb{R}^2$
- **Joint pmf:** $p(y, z, u) = \Pr(Y=y, Z=z, U=u)$, 8 entries summing to 1.
- **Conditional:** $X \mid (Y=y, Z=z, U=u) \sim \mathcal{N}(\mu_{yzu}, \Sigma_{yzu})$
- **Correlations:**
  $$
  \mathrm{corr}(Y, Z) = \mathbb{E}[YZ] = \sum_{y, z, u} yz\;p(y, z, u) \in [-1, 1]
  $$
  $$
  \mathrm{corr}(Y, U) = \mathbb{E}[Y U] = \sum_{y, z, u} yu\; p(y, z, u) \in [-1, 1]
  $$

---

## 1. Data Generation

### BaseDistribution

- **Class:** `BaseDistribution`
- **Inputs:** `seed: int`
- **Attributes:**
  `pmf` (shape (8,)), `mus` (dict (y,z,u)), `Sigmas` (dict), `corr_YZ`, `corr_YU`
- **Method:**  
  `compute_corr()` → computes correlaton $(Y,Z)$ and $(Y,U)$.

**Construction:**  
Seed is used for reproducibility. `pmf` obtained by sampling 8 positive iid numbers and normalizing. Each $\mu_{yzu} \in \mathbb{R}^2$ is drawn randomly (e.g., small Gaussian). Each $\Sigma_{yzu}$ is diagonal or stable SPD with jitter.

---

### PerturbedDistribution

- Inherits `BaseDistribution`
- **Inputs:**
  `perturb_level_YU`, `perturb_level_YZ` in named levels,
  $p \in [0, 1]$ ($Z$’s observability),
  `seed` + forwarded args.
- **Purpose:**
  Modify marginal correlation strength $(Y,U)$, $(Y,Z)$ while preserving remaining conditionals.

**Perturbation Mechanism:**
- For pair $(A, B) \in \{(Y,U), (Y,Z)\}$:
  1. Compute $p_{A,B}(a, b) = \sum_c p(a, b, c)$, $c$ is other var.
  2. Let $S_+ = \{(a, b): ab = +1\}$, $S_- = \{(a, b): ab = -1\}$, totals $P_+ = \sum_{S_+} p_{A,B}$, $P_- = \sum_{S_-} p_{A,B}$.
  3. One step: transfer mass $t = P_- / 3$ from $S_-$ to $S_+$, distributed proportionally to weights.
  4. Repeat step $S$ times $(S \in \{0,1,2,3\})$ per named perturbation. Negative $S$ for decrease.
  5. Full joint: $p'(a,b,c) = p'_{A,B}(a,b) p(c|a, b)$, with $p(c | a, b) = p(a, b, c) / p_{A,B}(a, b)$ where $p_{A,B}(a, b) > 0$.

- Apply YU and YZ sequentially.
- Numerical stability: clip negatives, renormalize, all probabilities $\geq 0$, sum to 1.

**Observability and Sampling:**  
- `sample(n, mask=True)`: with probability $p$, keep $z$ observed, else set to missing.

**Edge Cases:**  
If $P_-=0$ or $P_+=0$, skip transfers. If $n_{\text{vis}}=0$, skip regularizer.

---

## 2. Models

- **Feature Vectors:**
  - $\phi_f(x, z) = [1, x_1, x_2, z]^T$
  - $\phi_g(x) = [1, x_1, x_2]^T$
- **Linear Scores:**
  - $f(x, z) = w_f^T \phi_f(x, z)$
  - $g(x) = w_g^T \phi_g(x)$
- **Loss:**
  - $\ell(y, s) = \log(1 + \exp(-y s))$
- **API Contract:**
  - `fit(X, Y, Z, U)`, `eval(X, Y, Z, U)`
  - Attributes: `w_f`, `w_g`, training history

### Vanilla Classifier

- If $z$ observed: score $s = f(x, z)$
- If $z$ missing: $s = g(x)$
- Optimize average logistic loss over all samples jointly in $w_f, w_g$
- Prediction: $\mathrm{sign}(s)$

### Strategic Classifier

- If $z$ missing: $s = g(x)$
- If $z$ observed: depends on $u$:
  - If $u=+1$: $s = \mathrm{sm\_max}_\tau(f(x, z), g(x))$
  - If $u=-1$: $s = \mathrm{sm\_min}_\tau(f(x, z), g(x)) = -\mathrm{sm\_max}_\tau(-f, -g)$
- **Smooth Max:**
  $$
  \mathrm{sm\_max}_\tau(a, b) = \frac{1}{\tau} \log\left( e^{\tau a} + e^{\tau b} \right)
  $$
- **Objective:**
  $$
  \mathcal{L} = \frac{1}{n} \sum_{i=1}^n \ell(y_i, s_i) + \lambda \cdot \frac{1}{n_{\text{vis}}} \sum_{i \in \mathcal{V}} (f(x_i, z_i) - g(x_i))^2
  $$
  where $n_{\text{vis}}$ is number of visible $z$, $\mathcal{V}$ those samples (skip regularizer if $n_{\text{vis}}=0$).

- Stabilize optimization by normalizing features, weight decay.

---
## 3. Experiment Orchestration

- **`Experiment` Class**
  - **Inputs**
    - A `PerturbedDistribution` instance (or initialization parameters)
    - Classifier models (Vanilla and Strategic) provided as classes or factories
    - Training and testing set sizes
    - Optimizer selection and relevant hyperparameters
    - Parameter sweep settings (e.g., observability `p`, perturbation levels for `Y,U` and `Y,Z`)
  
  - **Core Methods**
    - `run_single(config)`
        Executes a single experiment configuration and returns a structured result dictionary including:
          - Distribution summary: numeric correlations (`corr_YU`, `corr_YZ`), `p`, perturbation levels
          - Model outputs: learned weights (`w_f`, `w_g`) for both classifiers
          - Training metrics and results: loss curves, test/train accuracy, convergence diagnostics

    - `run_sweep()`
        Iterates over all combinations in the parameter grid, initiating independent runs with progress tracking.
        Collects outputs into a unified summary DataFrame. Intermediate results are checkpointed for robustness.

    - `save_csv()`
        Saves the results DataFrame from `run_sweep()` to a CSV file for downstream analysis or record-keeping.



- **Sweep Parameters**
  - Observability: `p ∈ {0, 0.25, 0.5, 0.75, 1}`
  - Perturbation levels for both `YU` and `YZ`, chosen independently from the defined set
  - All combinations are evaluated systematically, with progress indicators and result checkpointing for reliability.


---

## 4. Visualization

- **Per-$p$ Plots:**
  - Scatter plot of $(\mathrm{corr}(Y, U), \mathrm{corr}(Y, Z))$ per experiment/config and save it.
  - Color rule (loss-based):
    - If Vanilla test loss < Strategic by >1%: Blue (Vanilla wins)
    - If Strategic test loss < Vanilla by >1%: Red (Strategic wins)
    - Otherwise: Grey (within 1%)


- **Tools:**
  Matplotlib/Seaborn static or Plotly for interactive, with legends, labels, and caption

---

## Suggested File Layout

- `data_generation.py` — `BaseDistribution`, `PerturbedDistribution`, sampling
- `models.py` — `VanillaClassifier`, `StrategicClassifier` (with shared base)
- `experiment.py` — `Experiment` runner, sweep orchestration, saving
- `visualization.py` — plotting helpers

---

## Appendix: Formulas

- **Joint Reconstruction:**
  $$
  p'(a, b, c) = p'_{A,B}(a, b) \cdot p(c|a, b)
  $$
- **Logistic Loss:**
  $$
  \ell(y, s) = \log(1+\exp(-y s))
  $$
- **Smooth Max:**
  $$
  \mathrm{sm\_max}_\tau(a, b) = \frac{1}{\tau}\log\left( e^{\tau a} + e^{\tau b} \right)
  $$

---

