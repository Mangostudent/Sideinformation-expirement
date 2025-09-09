"""Models for Side Information Experiment.

VanillaClassifier and StrategicClassifier implementing fit and eval.
"""
from typing import Optional, Dict, List
import numpy as np

import numpy as np

def logistic_loss(y, s):
    "Numerically stable logistic loss: log(1+exp(-y*s))"
    return np.logaddexp(0, -y * s)

def _stable_sigmoid(x):
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    neg = ~pos
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out

def _clip_grad(grad, max_norm=5.0):
    norm = np.linalg.norm(grad)
    if norm > max_norm and norm > 0:
        return grad * (max_norm / norm)
    return grad

def sm_max(a, b, tau=5.0):
    """Smooth max, differentiable: approaches max(a,b) as tau -> infty"""
    m = np.maximum(a, b)
    return (np.log(np.exp(tau*(a-m)) + np.exp(tau*(b-m))) + tau*m) / tau

def sm_min(a, b, tau=5.0):
    """Smooth min, differentiable: approaches min(a,b) as tau -> infty"""
    return -sm_max(-a, -b, tau)

class BaseClassifier:
    def __init__(self):
        self.w_f = None
        self.w_g = None
        self.history = {}

    def _phi_f(self, X, Z):
        n = X.shape[0]
        Zf = np.where(np.isnan(Z), 0.0, Z)
        return np.column_stack([np.ones(n), X[:, 0], X[:, 1], Zf])

    def _phi_g(self, X):
        n = X.shape[0]
        return np.column_stack([np.ones(n), X[:, 0], X[:, 1]])

# ----- VANILLA -----
class VanillaClassifier(BaseClassifier):
    def __init__(self, lr=1.0, epochs=200, weight_decay=1e-3):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay

    def fit(self, X, Y, Z, U=None):
        if not hasattr(self, 'weight_decay'):
            raise AttributeError("VanillaClassifier: 'weight_decay' attribute not set. Did you forget to call __init__?")
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
            # Gradients (stable): coeff = -Y * sigmoid(-Y * s)
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

    def eval(self, X, Y, Z, U=None):
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

# ----- STRATEGIC -----
class StrategicClassifier(BaseClassifier):
    def __init__(self, lr=1.0, epochs=200, tau=5.0, lam=1.0, weight_decay=1e-3):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.tau = tau
        self.lam = lam
        self.weight_decay = weight_decay

    def fit(self, X, Y, Z, U):
        if not hasattr(self, 'weight_decay'):
            raise AttributeError("StrategicClassifier: 'weight_decay' attribute not set. Did you forget to call __init__?")
        n = X.shape[0]
        Phi_f = self._phi_f(X, Z)
        Phi_g = self._phi_g(X)
        self.w_f = np.zeros(Phi_f.shape[1])
        self.w_g = np.zeros(Phi_g.shape[1])
        self.history = {'loss': None}

        mask_obs = ~np.isnan(Z)
        n_vis = mask_obs.sum()
        idx_vis = np.where(mask_obs)[0]
        idx_hid = np.where(~mask_obs)[0]

        last_loss = None
        for ep in range(self.epochs):
            s_f = Phi_f.dot(self.w_f)
            s_g = Phi_g.dot(self.w_g)
            s = np.zeros(n)
            s[~mask_obs] = s_g[~mask_obs]
            if mask_obs.any():
                for i in idx_vis:
                    if U[i] == 1:
                        s[i] = sm_max(s_f[i], s_g[i], tau=self.tau)
                    else:
                        s[i] = -sm_max(-s_f[i], -s_g[i], tau=self.tau)
            losses = logistic_loss(Y, s)
            reg = 0.0
            if n_vis > 0:
                diff = s_f[mask_obs] - s_g[mask_obs]
                diff = np.nan_to_num(diff, nan=0.0, posinf=1e6, neginf=-1e6)
                reg = (diff**2).mean()
            loss = losses.mean()
            loss = np.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
            loss = loss + self.lam * reg + 0.5 * self.weight_decay * (np.sum(self.w_f**2) + np.sum(self.w_g**2))
            last_loss = float(loss)
            # Gradient computation
            coeff = -Y * _stable_sigmoid(-Y * s)
            grad_w_f = np.zeros_like(self.w_f)
            grad_w_g = np.zeros_like(self.w_g)
            # For hidden Z: simple case, straight through g
            if len(idx_hid) > 0:
                grad_w_g += (Phi_g[idx_hid].T @ coeff[idx_hid]) / n

            # For visible Z: max/min (smooth) and regularizer
            for i in idx_vis:
                sf = s_f[i]
                sg = s_g[i]
                # Smooth max or min derivative weights
                if U[i] == 1:
                    a = self.tau * sf
                    b = self.tau * sg
                    m = max(a, b)
                    wa = np.exp(a - m)
                    wb = np.exp(b - m)
                    denom = wa + wb
                    dsf = (wa / denom)
                    dsg = (wb / denom)
                else:
                    a = -self.tau * sf
                    b = -self.tau * sg
                    m = max(a, b)
                    wa = np.exp(a - m)
                    wb = np.exp(b - m)
                    denom = wa + wb
                    dsf = - (wa / denom)
                    dsg = - (wb / denom)

                grad_w_f += (Phi_f[i] * (coeff[i] * dsf)) / n
                grad_w_g += (Phi_g[i] * (coeff[i] * dsg)) / n
                # Add reg gradients for regularizer to both:
                grad_w_f += (2.0 / n) * self.lam * (sf - sg) * Phi_f[i]
                grad_w_g += (-2.0 / n) * self.lam * (sf - sg) * Phi_g[i]

            # Optional: gradient clipping
            grad_w_f = _clip_grad(grad_w_f, max_norm=5.0)
            grad_w_g = _clip_grad(grad_w_g, max_norm=5.0)
            # Weight update
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
        s = np.zeros(n)
        mask_obs = ~np.isnan(Z)
        idx_vis = np.where(mask_obs)[0]
        idx_hid = np.where(~mask_obs)[0]
        for i in idx_vis:
            if U[i] == 1:
                s[i] = max(s_f[i], s_g[i])
            else:
                s[i] = min(s_f[i], s_g[i])
        if len(idx_hid) > 0:
            s[idx_hid] = s_g[idx_hid]
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
    params = {'perturb_level_YU': 1, 'perturb_level_YZ': 0.75, 'p': 1}
    dist = PerturbedDistribution(**params,seed=50)

    # Generate train and test data using the distribution's sample method
    train = dist.sample(n=200)
    test = dist.sample(n=100)

    # Train and evaluate VanillaClassifier
    van = VanillaClassifier()
    van.fit(train['X'], train['Y'], train['Z'], train['U'])
    van_train_loss = van.history['loss']
    van_test = van.eval(test['X'], test['Y'], test['Z'], test['U'])
    print("VanillaClassifier train loss:", van_train_loss)
    print("VanillaClassifier test:", van_test)

    # Train and evaluate StrategicClassifier
    strat = StrategicClassifier()
    strat.fit(train['X'], train['Y'], train['Z'], train['U'])
    strat_train_loss = strat.history['loss']
    strat_test = strat.eval(test['X'], test['Y'], test['Z'], test['U'])
    print("StrategicClassifier train loss:", strat_train_loss)
    print("StrategicClassifier test:", strat_test)