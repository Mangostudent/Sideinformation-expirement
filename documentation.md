# Detailed Documentation for `data_generation.py` and `models.py`

## Overview
This document provides a detailed explanation of every object, method, attribute, function, and variable in the `data_generation.py` and `models.py` files. It also explains any unfamiliar syntax or concepts used in the code.

---

## `data_generation.py`

### Classes

#### 1. `BaseDistribution`
- **Purpose**: Represents a base probability distribution over 8 combinations of `(Y, Z, U)`.
- **Attributes**:
  - `COMBINATIONS`: A hardcoded list of all 8 possible combinations of `(Y, Z, U)` values, each in `{-1, +1}`.
  - `pmf`: A probability mass function (PMF) represented as a NumPy array of shape `(8,)`. It is initialized randomly and normalized to sum to 1.
  - `mus`: A dictionary mapping each combination to a 2D mean vector (NumPy array of shape `(2,)`).
  - `sigmas`: A dictionary mapping each combination to a 2x2 positive definite covariance matrix (NumPy array of shape `(2, 2)`).
- **Methods**:
  - `__init__(seed: int = 0)`: Initializes the PMF, `mus`, and `sigmas` using a random seed for reproducibility.
  - `compute_corr() -> Tuple[float, float]`: Computes and returns the correlations `E[YZ]` and `E[YU]` using the PMF and combinations.

#### 2. `PerturbedDistribution` (inherits `BaseDistribution`)
- **Purpose**: Extends `BaseDistribution` to allow perturbations in the correlations between `(Y, Z)` and `(Y, U)`.
- **Attributes**:
  - `S_YU`: An integer representing the perturbation level for `(Y, U)` correlation.
  - `S_YZ`: An integer representing the perturbation level for `(Y, Z)` correlation.
  - `p`: A float representing the visibility parameter for `Z`.
  - `corr_YZ`, `corr_YU`: The computed correlations after applying perturbations.
- **Methods**:
  - `__init__(perturb_level_YU: int, perturb_level_YZ: int, p: float)`: Initializes the object, applies perturbations to the PMF, and computes the correlations.
  - `sample(n: int = 100) -> Dict[str, np.ndarray]`: Samples `n` points from the perturbed distribution and returns a dictionary with keys `X`, `Y`, `Z`, and `U`.
  - `_apply_perturbations(pmf: np.ndarray) -> np.ndarray`: Modifies the PMF to change correlations as specified by `S_YU` and `S_YZ`.
  - `_joint_to_table(pmf: np.ndarray) -> Dict[Tuple[int, int, int], float]`: Converts the PMF to a dictionary mapping `(Y, Z, U)` to probabilities.
  - `_table_to_pmf(table: Dict[Tuple[int, int, int], float]) -> np.ndarray`: Converts a dictionary back to a PMF (NumPy array).

---

## `models.py`

### Classes

#### 1. `BaseClassifier`
- **Purpose**: A base class for classifiers in the Side Information Experiment.
- **Attributes**:
  - `w_f`, `w_g`: Optional NumPy arrays representing weights for feature maps `Phi_f` and `Phi_g`.
  - `history`: A dictionary to store training history (e.g., loss values).
- **Methods**:
  - `__init__()`: Initializes the weights and history.
  - `_phi_f(X: np.ndarray, Z: np.ndarray) -> np.ndarray`: Computes the feature map `Phi_f` for observed `Z` values.
  - `_phi_g(X: np.ndarray) -> np.ndarray`: Computes the feature map `Phi_g` for missing `Z` values.

#### 2. `VanillaClassifier` (inherits `BaseClassifier`)
- **Purpose**: Implements a simple classifier using gradient-based optimization.
- **Attributes**:
  - `lr`: Learning rate for gradient updates.
  - `epochs`: Number of training epochs.
  - `weight_decay`: Regularization parameter for weight decay.
- **Methods**:
  - `__init__(lr: float, epochs: int, weight_decay: float)`: Initializes the classifier with hyperparameters.
  - `fit(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, U: np.ndarray) -> Self`: Trains the classifier using gradient-based optimization.
  - `eval(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, U: np.ndarray) -> Dict[str, float]`: Evaluates the classifier and returns metrics (loss and accuracy).

#### 3. `StrategicClassifier` (inherits `BaseClassifier`)
- **Purpose**: Implements a strategic classifier that accounts for smooth-max operations.
- **Attributes**:
  - `tau`: Temperature parameter for smooth-max.
  - `lam`: Regularization parameter for correlation penalties.
- **Methods**:
  - `__init__(lr: float, epochs: int, tau: float, lam: float, weight_decay: float)`: Initializes the classifier with hyperparameters.
  - `fit(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, U: np.ndarray) -> Self`: Trains the classifier using gradient-based optimization with smooth-max.
  - `eval(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, U: np.ndarray) -> Dict[str, float]`: Evaluates the classifier and returns metrics (loss and accuracy).

### Functions

#### 1. `logistic_loss(y: np.ndarray, s: np.ndarray) -> np.ndarray`
- **Purpose**: Computes the logistic loss for given labels `y` and scores `s`.
- **Details**: Uses a numerically stable implementation with `np.logaddexp`.

#### 2. `sm_max(a: float, b: float, tau: float = 5.0) -> float`
- **Purpose**: Computes a numerically stable smooth maximum of `a` and `b`.
- **Details**: Uses a temperature parameter `tau` to control smoothness.

#### 3. `_stable_sigmoid(x: np.ndarray) -> np.ndarray`
- **Purpose**: Computes a numerically stable sigmoid function.
- **Details**: Handles large positive and negative values of `x` to avoid overflow.

#### 4. `_clip_grad(grad: np.ndarray, max_norm: float = 5.0) -> np.ndarray`
- **Purpose**: Clips the gradient vector to have an L2 norm at most `max_norm`.
- **Details**: Prevents exploding gradients during training.

---

## Example Usage

### `data_generation.py`
- Example usage is provided in the `if __name__ == "__main__"` block.
- Demonstrates how to create and sample from `PerturbedDistribution` with different parameters.

### `models.py`
- Example usage is provided in the `if __name__ == "__main__"` block.
- Demonstrates how to train and evaluate `VanillaClassifier` and `StrategicClassifier` using data generated from `PerturbedDistribution`.

---

## Notes on Syntax and Concepts

### 1. `np.random.RandomState`
- Used to create a random number generator with a fixed seed for reproducibility.

### 2. `np.logaddexp`
- Computes `log(exp(x1) + exp(x2))` in a numerically stable way.

### 3. `np.nan_to_num`
- Replaces NaN, positive infinity, and negative infinity with specified finite values.

### 4. `np.linalg.norm`
- Computes the L2 norm (Euclidean distance) of a vector.

### 5. `np.exp`
- Computes the exponential of all elements in an array.
- Used with care to avoid overflow by shifting inputs.

---

This concludes the detailed documentation for `data_generation.py` and `models.py`. Let me know if you need further clarifications or additional examples!
