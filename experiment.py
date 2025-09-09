"""Experiment orchestration for the Side Information Experiment."""
from typing import Any, Dict, List, Tuple
import itertools
import pandas as pd
import numpy as np
from data_generation import PerturbedDistribution
from models import VanillaClassifier, StrategicClassifier


class Experiment:
    def __init__(self, dist_params: Dict[str, Any], seed: int = 0):
        self.dist_params = dist_params.copy()
        self.seed = seed
        self.results: List[Dict[str, Any]] = []

    def run_single(self, perturb_YU, perturb_YZ, p_vis, train_n=1000, test_n=1000, model_hparams=None):
        params = dict(self.dist_params)
        params.update({'perturb_level_YU': perturb_YU, 'perturb_level_YZ': perturb_YZ, 'p': p_vis, 'seed': self.seed})
        dist = PerturbedDistribution(perturb_level_YU=perturb_YU, perturb_level_YZ=perturb_YZ, p=p_vis)

        # sample train/test
        train = dist.sample(train_n, mask_z_p=p_vis, seed=self.seed + 1)
        test = dist.sample(test_n, mask_z_p=p_vis, seed=self.seed + 2)

        # instantiate models
        if model_hparams is None:
            model_hparams = {}
        van = VanillaClassifier(**model_hparams.get('vanilla', {}))
        strat = StrategicClassifier(**model_hparams.get('strategic', {}))

        van.fit(train['X'], train['Y'], train['Z'], train['U'])
        strat.fit(train['X'], train['Y'], train['Z'], train['U'])

        van_train = van.eval(train['X'], train['Y'], train['Z'], train['U'])
        van_test = van.eval(test['X'], test['Y'], test['Z'], test['U'])
        strat_train = strat.eval(train['X'], train['Y'], train['Z'], train['U'])
        strat_test = strat.eval(test['X'], test['Y'], test['Z'], test['U'])

        res = {
            'perturb_YU': perturb_YU,
            'perturb_YZ': perturb_YZ,
            'p': p_vis,
            'corr_YU': dist.corr_YU,
            'corr_YZ': dist.corr_YZ,
            'van_w_f': van.w_f.tolist(),
            'van_w_g': van.w_g.tolist(),
            'strat_w_f': strat.w_f.tolist(),
            'strat_w_g': strat.w_g.tolist(),
            'van_train_loss': van_train['loss'],
            'van_test_loss': van_test['loss'],
            'van_train_acc': van_train['accuracy'],
            'van_test_acc': van_test['accuracy'],
            'strat_train_loss': strat_train['loss'],
            'strat_test_loss': strat_test['loss'],
            'strat_train_acc': strat_train['accuracy'],
            'strat_test_acc': strat_test['accuracy'],
        }

        self.results.append(res)
        return res

    def run_sweep(self, p_list=None, perturb_levels=None, train_n=1000, test_n=1000, model_hparams=None):
        if p_list is None:
            p_list = [0.0,0.25,0.5,0.75,1.0]
        if perturb_levels is None:
            perturb_levels = [2]

        total = len(p_list) * len(perturb_levels) * len(perturb_levels)
        i = 0
        for p_vis in p_list:
            for pyu in perturb_levels:
                for pyz in perturb_levels:
                    i += 1
                    print(f"Running {i}/{total}: p={p_vis}, YU={pyu}, YZ={pyz}")
                    self.run_single(pyu, pyz, p_vis, train_n=train_n, test_n=test_n, model_hparams=model_hparams)
                    # checkpoint every run
                    self.save_csv('experiment_results_checkpoint.csv')

        return self.get_results_df()

    def get_results_df(self):
        return pd.DataFrame(self.results)


    def save_csv(self, path: str = 'experiment_results.csv'):
        df = self.get_results_df()
        df.to_csv(path, index=False)
        return path


if __name__ == "__main__":
    # Minimal run for inspection: p=1.0, perturbation levels=2
    dist_params = {}
    exp = Experiment(dist_params, seed=42)
    # Use small train/test size for quick check
    result = exp.run_single(perturb_YU=2, perturb_YZ=2, p_vis=1.0, train_n=20, test_n=10)
    print("Single run result:")
    for k, v in result.items():
        print(f"{k}: {v}")
