
"""Models for Side Information Experiment.

VanillaClassifier and StrategicClassifier implementing fit and eval.
"""
from typing import Optional, Dict, List
import numpy as np


def logistic_loss(y, s):
    # numerically stable logistic loss: log(1 + exp(-y*s))
    # use logaddexp for stability: log(1 + exp(-y*s)) = logaddexp(0, -y*s)
    return np.logaddexp(0, -y * s)


def _stable_sigmoid(x):
    # numerically stable sigmoid
    # for x >= 0: 1/(1+exp(-x)); for x < 0: exp(x)/(1+exp(x))
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    neg = ~pos
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def _clip_grad(grad: np.ndarray, max_norm: float = 5.0) -> np.ndarray:
    """Clip gradient vector to have L2 norm at most max_norm."""
    norm = np.linalg.norm(grad)
    if norm > max_norm and norm > 0:
        return grad * (max_norm / norm)
    return grad


def sm_max(a, b, tau=5.0):
    # numerically stable smooth max
    m = np.maximum(a, b)
    return (np.log(np.exp(tau * (a - m)) + np.exp(tau * (b - m))) + tau * m) / tau


class BaseClassifier:
    def __init__(self):
        self.w_f: Optional[np.ndarray] = None
        self.w_g: Optional[np.ndarray] = None
        self.history: Dict[str, List[float]] = {}

    def _phi_f(self, X, Z):
        # X: (n,2), Z: (n,) with possible nan
        n = X.shape[0]
        Zf = np.where(np.isnan(Z), 0.0, Z)
        return np.column_stack([np.ones(n), X[:, 0], X[:, 1], Zf])

    def _phi_g(self, X):
        n = X.shape[0]
        return np.column_stack([np.ones(n), X[:, 0], X[:, 1]])


class VanillaClassifier(BaseClassifier):
    def __init__(self, lr=1.0, epochs=200, weight_decay=1e-3):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay

    def fit(self, X, Y, Z, U):
        n = X.shape[0]
        Phi_f = self._phi_f(X, Z)
        Phi_g = self._phi_g(X)
        # initialize weights
        self.w_f = np.zeros(Phi_f.shape[1])
        self.w_g = np.zeros(Phi_g.shape[1])
        self.history = {'loss': None}

        # simple gradient-based joint optimization
        last_loss = None
        for ep in range(self.epochs):
            mask_obs = ~np.isnan(Z)
            s = np.zeros(n)
            s[mask_obs] = Phi_f[mask_obs].dot(self.w_f)
            s[~mask_obs] = Phi_g[~mask_obs].dot(self.w_g)
            losses = logistic_loss(Y, s)
            loss = losses.mean()
            loss = np.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
            loss = loss + 0.5 * self.weight_decay * (np.sum(self.w_f**2) + np.sum(self.w_g**2))
            last_loss = float(loss)
            # gradients (stable): coeff = -Y * sigmoid(-Y * s)
            coeff = -Y * _stable_sigmoid(-Y * s)
            grad_w_f = np.zeros_like(self.w_f)
            grad_w_g = np.zeros_like(self.w_g)
            if mask_obs.any():
                grad_w_f = (Phi_f[mask_obs].T @ coeff[mask_obs]) / n + self.weight_decay * self.w_f
            if (~mask_obs).any():
                grad_w_g = (Phi_g[~mask_obs].T @ coeff[~mask_obs]) / n + self.weight_decay * self.w_g
            grad_w_f = _clip_grad(grad_w_f, max_norm=5.0)
            grad_w_g = _clip_grad(grad_w_g, max_norm=5.0)
            self.w_f -= self.lr * grad_w_f
            self.w_g -= self.lr * grad_w_g
        self.history['loss'] = last_loss
        return self

    def eval(self, X, Y, Z, U):
        n = X.shape[0]
        Phi_f = self._phi_f(X, Z)
        Phi_g = self._phi_g(X)
        mask_obs = ~np.isnan(Z)
        s = np.zeros(n)
        s[mask_obs] = Phi_f[mask_obs].dot(self.w_f)
        s[~mask_obs] = Phi_g[~mask_obs].dot(self.w_g)
        loss = logistic_loss(Y, s).mean()
        preds = np.sign(s)
        acc = (preds == Y).mean()
        return {'loss': float(loss), 'accuracy': float(acc)}


class StrategicClassifier(BaseClassifier):
    def __init__(self, lr=1.0, epochs=200, tau=5.0, lam=1.0, weight_decay=1e-3):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.tau = tau
        self.lam = lam
        self.weight_decay = weight_decay

    def fit(self, X, Y, Z, U):
        n = X.shape[0]
        Phi_f = self._phi_f(X, Z)
        Phi_g = self._phi_g(X)
        self.w_f = np.zeros(Phi_f.shape[1])
        self.w_g = np.zeros(Phi_g.shape[1])
        self.history = {'loss': None}

        mask_obs = ~np.isnan(Z)
        n_vis = mask_obs.sum()

        last_loss = None
        for ep in range(self.epochs):
            s_f = Phi_f.dot(self.w_f)
            s_g = Phi_g.dot(self.w_g)
            s = np.zeros(n)
            s[~mask_obs] = s_g[~mask_obs]
            if mask_obs.any():
                idx = np.where(mask_obs)[0]
                for i in idx:
                    if U[i] == 1:
                        s[i] = sm_max(s_f[i], s_g[i], tau=self.tau)
                    else:
                        s[i] = -sm_max(-s_f[i], -s_g[i], tau=self.tau)
            losses = logistic_loss(Y, s)
            reg = 0.0
            if n_vis > 0:
                diff = s_f[mask_obs] - s_g[mask_obs]
                # guard against huge values
                diff = np.nan_to_num(diff, nan=0.0, posinf=1e6, neginf=-1e6)
                reg = (diff**2).mean()
            loss = losses.mean()
            loss = np.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
            loss = loss + self.lam * reg + 0.5 * self.weight_decay * (np.sum(self.w_f**2) + np.sum(self.w_g**2))
            last_loss = float(loss)
            coeff = -Y * _stable_sigmoid(-Y * s)
            grad_w_f = np.zeros_like(self.w_f)
            grad_w_g = np.zeros_like(self.w_g)
            if (~mask_obs).any():
                grad_w_g += (Phi_g[~mask_obs].T @ coeff[~mask_obs]) / n
            if mask_obs.any():
                idx = np.where(mask_obs)[0]
                for i in idx:
                    sf = s_f[i]
                    sg = s_g[i]
                    # compute softmax-like weights in a numerically stable way
                    a = self.tau * sf
                    b = self.tau * sg
                    m = max(a, b)
                    wa = np.exp(a - m)
                    wb = np.exp(b - m)
                    denom = wa + wb
                    # derivatives of smooth-max wrt sf and sg
                    dsf = (wa / denom)
                    dsg = (wb / denom)
                    if U[i] != 1:
                        dsf = -dsf
                        dsg = -dsg
                    grad_w_f += (Phi_f[i] * (coeff[i] * dsf)) / n
                    grad_w_g += (Phi_g[i] * (coeff[i] * dsg)) / n
                if n_vis > 0:
                    diff = s_f[mask_obs] - s_g[mask_obs]
                    grad_w_f[0:Phi_f.shape[1]] += (2.0 / n) * (Phi_f[mask_obs].T @ diff) * self.lam
                    grad_w_g[0:Phi_g.shape[1]] += (-2.0 / n) * (Phi_g[mask_obs].T @ diff) * self.lam
            grad_w_f += self.weight_decay * self.w_f
            grad_w_g += self.weight_decay * self.w_g
            # clip gradients to avoid explosion
            grad_w_f = _clip_grad(grad_w_f, max_norm=5.0)
            grad_w_g = _clip_grad(grad_w_g, max_norm=5.0)
            self.w_f -= self.lr * grad_w_f
            self.w_g -= self.lr * grad_w_g
        self.history['loss'] = last_loss
        return self

    def eval(self, X, Y, Z, U):
        n = X.shape[0]
        Phi_f = self._phi_f(X, Z)
        Phi_g = self._phi_g(X)
        s_f = Phi_f.dot(self.w_f)
        s_g = Phi_g.dot(self.w_g)
        mask_obs = ~np.isnan(Z)
        s = np.zeros(n)
        s[~mask_obs] = s_g[~mask_obs]
        if mask_obs.any():
            idx = np.where(mask_obs)[0]
            for i in idx:
                if U[i] == 1:
                    s[i] = sm_max(s_f[i], s_g[i], tau=self.tau)
                else:
                    s[i] = -sm_max(-s_f[i], -s_g[i], tau=self.tau)
        loss = logistic_loss(Y, s).mean()
        preds = np.sign(s)
        acc = (preds == Y).mean()
        return {'loss': float(loss), 'accuracy': float(acc)}

"""
Example usage of VanillaClassifier and StrategicClassifier with data generated from data_generation.py
"""
from data_generation import PerturbedDistribution


if __name__ == "__main__":
    from data_generation import PerturbedDistribution
    # Set specific parameters for reproducibility
    params = {'perturb_level_YU': 1, 'perturb_level_YZ': 0, 'p': 0.2}
    dist = PerturbedDistribution(**params,seed=50)

    # Generate train and test data using the distribution's sample method
    train = dist.sample(n=200)
    test = dist.sample(n=100)

    # Train and evaluate VanillaClassifier
    van = VanillaClassifier()
    van.fit(train['X'], train['Y'], train['Z'], train['U'])
    van_train = van.eval(train['X'], train['Y'], train['Z'], train['U'])
    van_test = van.eval(test['X'], test['Y'], test['Z'], test['U'])
    print("VanillaClassifier train:", van_train)
    print("VanillaClassifier test:", van_test)

    # Train and evaluate StrategicClassifier
    strat = StrategicClassifier()
    strat.fit(train['X'], train['Y'], train['Z'], train['U'])
    strat_train = strat.eval(train['X'], train['Y'], train['Z'], train['U'])
    strat_test = strat.eval(test['X'], test['Y'], test['Z'], test['U'])
    print("StrategicClassifier train:", strat_train)
    print("StrategicClassifier test:", strat_test)