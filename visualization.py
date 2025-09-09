"""Visualization helpers for experiment results."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_correlation_space(csv_path='experiment_results.csv'):
    results = pd.read_csv(csv_path)
    p_values = sorted(results['p'].unique())
    # Define color map: 0=Vanilla, 1=Strategic, 2=Tie
    color_map = {0: 'tab:blue', 1: 'tab:orange', 2: 'gray'}
    label_map = {0: 'Vanilla', 1: 'Strategic', 2: 'Tie'}
    for metric, label, vcol, scol in [
        ('test_loss', 'Model with Lower Test Loss', 'van_test_loss', 'strat_test_loss'),
        ('test_acc', 'Model with Higher Test Accuracy', 'van_test_acc', 'strat_test_acc')
    ]:
        for p in p_values:
            dfp = results[results['p'] == p]
            if len(dfp) == 0:
                continue
            # Compute margin and winner
            if metric == 'test_loss':
                margin = (dfp[scol] - dfp[vcol]) / np.maximum(dfp[vcol], dfp[scol])
                winner = np.where(margin > 0.05, 0, np.where(margin < -0.05, 1, 2))
            else:
                margin = (dfp[vcol] - dfp[scol]) / np.maximum(dfp[vcol], dfp[scol])
                winner = np.where(margin > 0.05, 0, np.where(margin < -0.05, 1, 2))
            colors = [color_map[w] for w in winner]
            plt.figure(figsize=(7, 6))
            plt.scatter(dfp['corr_YZ'], dfp['corr_YU'], c=colors, alpha=0.8, edgecolor='k')
            # Custom legend
            for i in [0, 1, 2]:
                plt.scatter([], [], c=color_map[i], label=label_map[i])
            plt.xlabel('Correlation E[YZ]')
            plt.ylabel('Correlation E[YU]')
            plt.title(f'{label} (p={p})')
            plt.legend(title='Winner')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f'correlation_space_{metric}_p{p}.png')
            plt.close()


if __name__ == "__main__":
    plot_correlation_space()
