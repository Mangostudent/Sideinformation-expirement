"""Experiment orchestration for the Side Information Experiment."""
from typing import Any, Dict, List
import itertools
import pandas as pd
from data_generation import PerturbedDistribution
from models import VanillaClassifier, StrategicClassifier


class Experiment:
    def __init__(self, param_space: Dict[str, list], seed=0):
        """
        param_space: dict of parameter names to lists of values (the parameter grid)
        seed: random seed for reproducibility
        """
        self.param_space = param_space
        self.seed = seed
        self.results: List[Dict[str, Any]] = []

    def single_run(self, params: Dict[str, Any], model_hparams=None):
        """
        params: dict of parameter values for this run (e.g., {'perturb_level_YU': 2, ...})
        model_hparams: dict of model hyperparameters
        """
        dist = PerturbedDistribution(
            perturb_level_YU=params['perturb_level_YU'],
            perturb_level_YZ=params['perturb_level_YZ'],
            p=params['p'],
            seed=self.seed
        )
        train_n = params.get('train_n', 1000)
        test_n = params.get('test_n', 1000)
        train = dist.sample(train_n)
        test = dist.sample(test_n)
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
            **params,
            'corr_YU': dist.corr_YU,
            'corr_YZ': dist.corr_YZ,
            'van_w_f': van.w_f.tolist(),
            'van_w_g': van.w_g.tolist(),
            'strat_w_f': strat.w_f.tolist(),
            'strat_w_g': strat.w_g.tolist(),
            'van_train_loss': van_train['loss'],
            'van_test_loss': van_test['loss'],
            'van_test_acc': van_test['accuracy'],
            'strat_train_loss': strat_train['loss'],
            'strat_test_loss': strat_test['loss'],
            'strat_test_acc': strat_test['accuracy'],
        }
        self.results.append(res)
        return res

    def run_sweep(self, model_hparams=None):
        keys = list(self.param_space.keys())
        values = [self.param_space[k] for k in keys]
        combos = list(itertools.product(*values))
        total = len(combos)
        results = []
        for idx, combo in enumerate(combos, 1):
            print(f"Sweep progress: {idx}/{total} ({idx/total:.1%})", end='\r')
            params = dict(zip(keys, combo))
            res = self.single_run(params, model_hparams=model_hparams)
            results.append(res)
        print()  # for clean newline after sweep
        return results

    def get_results_df(self):
        return pd.DataFrame(self.results)

    def save_csv(self, path: str = 'experiment_results.csv'):
        df = self.get_results_df()
        # Reorder columns: p, correlations, then rest
        col_order = []
        # Always put 'p' first if present
        if 'p' in df.columns:
            col_order.append('p')
        # Then correlations if present
        for c in ['corr_YZ', 'corr_YU']:
            if c in df.columns: 
                col_order.append(c)
        # Then other parameter columns (excluding p and correlations)
        param_cols = [c for c in ['perturb_level_YU', 'perturb_level_YZ', 'train_n', 'test_n'] if c in df.columns]
        col_order.extend(param_cols)
        # Then all other columns (results, weights, etc.)
        rest = [c for c in df.columns if c not in col_order]
        col_order.extend(rest)
        # Reorder and sort by p, then correlations, then params
        df = df[col_order]
        sort_keys = [k for k in ['p', 'corr_YZ', 'corr_YU', 'perturb_level_YU', 'perturb_level_YZ', 'train_n', 'test_n'] if k in df.columns]
        df = df.sort_values(by=sort_keys).reset_index(drop=True)
        df.to_csv(path, index=False)
        return path


if __name__ == "__main__":
    
    param_space = {'perturb_level_YU': [-3,-2,-1,0,1,2,3], 'perturb_level_YZ': [-3,-2,-1,0,1,2,3], 'p': [0,0.25,0.5,0.75,1], 'train_n': [2500], 'test_n': [500]}
    exp = Experiment(param_space, seed=50)

    exp.run_sweep()
    exp.save_csv()




