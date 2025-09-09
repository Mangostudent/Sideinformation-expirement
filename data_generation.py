"""
# Detailed Documentation for `data_generation.py`
#
# Overview:
#   Implements the `BaseDistribution` and `PerturbedDistribution` classes for the Side Information Experiment.
#   This documentation explains every variable, attribute, method, and object in a hierarchical manner.
#
# 1. BaseDistribution Class
#    - COMBINATIONS: List of all 8 (Y, Z, U) combinations, each in {-1, +1}.
#    - pmf: Probability mass function (np.ndarray, shape (8,)), random, normalized.
#    - mus: Dict mapping each combination to a 2D mean vector (np.ndarray, shape (2,)).
#    - sigmas: Dict mapping each combination to a 2x2 positive definite covariance matrix (np.ndarray, shape (2,2)).
#    - __init__(seed): Initializes all above attributes using the given seed for reproducibility.
#    - compute_corr(): Returns (E[YZ], E[YU]) using the pmf and combinations.
#
# 2. PerturbedDistribution Class (inherits BaseDistribution)
#    - S_YU: int, perturbation level for (Y, U) correlation.
#    - S_YZ: int, perturbation level for (Y, Z) correlation.
#    - __init__(perturb_level_YU, perturb_level_YZ, p): Calls BaseDistribution, applies perturbations.
#    - _apply_perturbations(pmf): Modifies pmf to change correlations as specified.
#    - _joint_to_table(pmf): Converts pmf to dict mapping (Y, Z, U) to probability.
#    - _table_to_pmf(table): Converts dict back to pmf (np.ndarray).
#
# Example usage is provided at the end of this file for PerturbedDistribution.
"""
from typing import Dict, Tuple
import numpy as np



class BaseDistribution:
    # Hardcoded 8 combinations of (y, z, u)
    COMBINATIONS = [
        (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
        (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)
    ]

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        # Random PMF
        raw = self.rng.rand(8)
        self.pmf = raw / raw.sum()
        # Random mus and sigmas
        self.mus = {}
        self.sigmas = {}
        for k in self.COMBINATIONS:
            # mean: sample from 2D standard normal
            mu = self.rng.randn(2)
            # covariance: sample random 2x2 matrix, make SPD
            A = self.rng.randn(2, 2)
            Sigma = np.dot(A, A.T) + np.eye(2) * 0.1  # add jitter for SPD
            self.mus[k] = mu
            self.sigmas[k] = Sigma

    def compute_corr(self):
        # compute correlations E[YZ], E[YU]
        corr_YZ = 0.0
        corr_YU = 0.0
        for prob, (y, z, u) in zip(self.pmf, self.COMBINATIONS):
            corr_YZ += prob * y * z
            corr_YU += prob * y * u
        return float(corr_YZ), float(corr_YU)

    # No sampling implemented



class PerturbedDistribution(BaseDistribution):

    def sample(self, n=100):
        """Sample (X, Y, Z, U) from the perturbed distribution using internal RNG.
        Args:
            n: number of samples
        Returns:
            dict with keys 'X', 'Y', 'Z', 'U'
        """
        idx = self.rng.choice(8, size=n, p=self.pmf)
        X = np.zeros((n, 2), dtype=float)
        Y = np.zeros(n, dtype=int)
        Z = np.zeros(n, dtype=float)
        U = np.zeros(n, dtype=int)
        keys = list(self.COMBINATIONS)
        for i, k in enumerate(keys):
            mask = idx == i
            cnt = mask.sum()
            if cnt == 0:
                continue
            mu = self.mus[k]
            Sigma = self.sigmas[k]
            X[mask] = self.rng.multivariate_normal(mu, Sigma, size=cnt)
            Y[mask] = k[0]
            Z[mask] = k[1]
            U[mask] = k[2]
        vis = self.rng.rand(n) < self.p
        Z_masked = np.where(vis, Z, np.nan)
        return {"X": X, "Y": Y, "Z": Z_masked, "U": U}
    """Perturbs correlations between (Y,U) and (Y,Z) using the algorithm in instructions.

    perturb_level_YU and perturb_level_YZ may be integers in -3..3 or named levels mapping to that.
    """

    def __init__(self, perturb_level_YU=0, perturb_level_YZ=0, p: float = 1.0):
        super().__init__()
        self.p = float(p)
        self.S_YU = int(perturb_level_YU)
        self.S_YZ = int(perturb_level_YZ)
        self.pmf = self._apply_perturbations(self.pmf)
        self.corr_YZ, self.corr_YU = self.compute_corr()

    def _joint_to_table(self, pmf):
        # return dict mapping (y,z,u) -> prob
        d = {}
        for p, k in zip(pmf, BaseDistribution.COMBINATIONS):
            d[k] = float(p)
        return d

    def _table_to_pmf(self, table):
        arr = np.array([table[k] for k in BaseDistribution.COMBINATIONS], dtype=float)
        # clip and renormalize
        arr = np.clip(arr, 0.0, None)
        s = arr.sum()
        if s <= 0:
            # fallback to uniform
            arr = np.ones_like(arr) / arr.size
        else:
            arr = arr / s
        return arr

    def _apply_perturbations(self, pmf):
        table = self._joint_to_table(pmf)

        def perturb_pair(table, A, B, S):
            # A and B are 'Y','Z' or 'U'
            # compute marginal p_AB(a,b)
            states = list(BaseDistribution.COMBINATIONS)
            pAB = {}
            # c is the remaining variable
            vars = ['Y', 'Z', 'U']
            other = [v for v in vars if v not in (A, B)][0]
            # sum over c
            for a in [-1, 1]:
                for b in [-1, 1]:
                    s = 0.0
                    for (y, z, u) in states:
                        vals = {'Y': y, 'Z': z, 'U': u}
                        if vals[A] == a and vals[B] == b:
                            s += table[(y, z, u)]
                    pAB[(a, b)] = s

            S_steps = abs(S)
            direction = 1 if S >= 0 else -1
            for _ in range(S_steps):
                S_plus = [(a, b) for (a, b) in pAB if a * b == 1]
                S_minus = [(a, b) for (a, b) in pAB if a * b == -1]
                P_plus = sum(pAB[k] for k in S_plus)
                P_minus = sum(pAB[k] for k in S_minus)
                if P_minus == 0 and direction == 1:
                    break
                if P_plus == 0 and direction == -1:
                    break
                # amount to transfer
                t = (P_minus / 3.0) if direction == 1 else (P_plus / 3.0)
                if t <= 0:
                    break
                # transfer mass from S_minus to S_plus (or reverse)
                if direction == 1:
                    src = S_minus
                    dst = S_plus
                else:
                    src = S_plus
                    dst = S_minus

                # distribute proportionally by current pAB weight
                src_total = sum(pAB[k] for k in src)
                if src_total <= 0:
                    break
                for k in src:
                    frac = pAB[k] / src_total if src_total > 0 else 0
                    delta = t * frac
                    pAB[k] = max(pAB[k] - delta, 0.0)
                # add to destinations proportionally to their weights
                dst_total = sum(pAB[k] for k in dst)
                # if dst_total == 0 distribute evenly
                if dst_total <= 0:
                    per = t / len(dst)
                    for k in dst:
                        pAB[k] += per
                else:
                    for k in dst:
                        frac = pAB[k] / dst_total
                        pAB[k] += t * frac

            # reconstruct full joint p'(a,b,c) = p'_AB(a,b) * p(c|a,b)
            new_table = {}
            for (y, z, u) in states:
                vals = {'Y': y, 'Z': z, 'U': u}
                a = vals[A]
                b = vals[B]
                # get original conditional p(c | a, b)
                # compute p_AB_orig
                p_ab_orig = 0.0
                for (yy, zz, uu) in states:
                    v = {'Y': yy, 'Z': zz, 'U': uu}
                    if v[A] == a and v[B] == b:
                        p_ab_orig += table[(yy, zz, uu)]
                if p_ab_orig > 0:
                    # p(c | a,b) = p(a,b,c)/p_ab_orig
                    p_c_given_ab = table[(y, z, u)] / p_ab_orig
                else:
                    # fallback: uniform over c
                    p_c_given_ab = 0.5
                new_table[(y, z, u)] = pAB[(a, b)] * p_c_given_ab
            return new_table

        # apply YU then YZ sequentially
        table1 = perturb_pair(table, 'Y', 'U', self.S_YU)
        table2 = perturb_pair(table1, 'Y', 'Z', self.S_YZ)
        return self._table_to_pmf(table2)


# Example usage for testing


# Example usage for testing PerturbedDistribution with multiple parameter values

if __name__ == "__main__":
    param_sets = [
        {'perturb_level_YU': 0, 'perturb_level_YZ': 0, 'p': 1.0},
        {'perturb_level_YU': 2, 'perturb_level_YZ': -2, 'p': 0.8},
        {'perturb_level_YU': -3, 'perturb_level_YZ': 3, 'p': 0.5},
    ]
    for i, params in enumerate(param_sets):
        print(f"\n--- PerturbedDistribution Example {i+1} ---")
        pdist = PerturbedDistribution(**params)
        print(f"Parameters: {params}")
        print("PMF:", pdist.pmf)
        print("mus:")
        for k, mu in pdist.mus.items():
            print(f"  {k}: {mu}")
        print("sigmas:")
        for k, sigma in pdist.sigmas.items():
            print(f"  {k}:\n{sigma}")
        corr_YZ, corr_YU = pdist.compute_corr()
        print("Correlation E[YZ]:", corr_YZ)
        print("Correlation E[YU]:", corr_YU)
        # Show a few samples
        samples = pdist.sample(n=5)
        print("Sampled points (first 5):")
        for j in range(5):
            print(f"  X: {samples['X'][j]}, Y: {samples['Y'][j]}, Z: {samples['Z'][j]}, U: {samples['U'][j]}")