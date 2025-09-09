"""Visualization helpers for experiment results."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_corr_scatter(df: pd.DataFrame, p_value: float, outpath: str = None):
    # filter for p
    sub = df[np.isclose(df['p'], p_value)]
    x = sub['corr_YU'].astype(float)
    y = sub['corr_YZ'].astype(float)

    # color rule based on test loss
    van = sub['van_test_loss'].astype(float)
    strat = sub['strat_test_loss'].astype(float)
    color = []
    for v, s in zip(van, strat):
        if v < s * 0.99:
            color.append('blue')
        elif s < v * 0.99:
            color.append('red')
        else:
            color.append('grey')

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, c=color, alpha=0.8)
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.xlabel('corr(Y,U)')
    plt.ylabel('corr(Y,Z)')
    plt.title(f'Corr scatter p={p_value}')
    if outpath:
        plt.savefig(outpath, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_winners(results_path: str):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Load experiment results
    results = pd.read_csv(results_path)

    # Unique p values
    p_values = sorted(results['p'].unique())

    for metric, label, vcol, scol in [
        ('test_loss', 'Model with Lower Test Loss', 'van_test_loss', 'strat_test_loss'),
        ('test_acc', 'Model with Higher Test Accuracy', 'van_test_acc', 'strat_test_acc')
    ]:
        for p in p_values:
            dfp = results[results['p'] == p]
            # Determine winner: 0=vanilla, 1=strategic
            if metric == 'test_loss':
                winner = np.where(dfp[vcol] < dfp[scol], 0, 1)
            else:
                winner = np.where(dfp[vcol] > dfp[scol], 0, 1)
            colors = np.array(['tab:blue', 'tab:orange'])[winner]
            labels = np.array(['Vanilla', 'Strategic'])[winner]
            plt.figure(figsize=(7, 6))
            scatter = plt.scatter(dfp['corr_YZ'], dfp['corr_YU'], c=colors, label=None, alpha=0.8, edgecolor='k')
            # Custom legend
            for i, name in enumerate(['Vanilla', 'Strategic']):
                plt.scatter([], [], c=['tab:blue', 'tab:orange'][i], label=name)
            plt.xlabel('Correlation E[YZ]')
            plt.ylabel('Correlation E[YU]')
            plt.title(f'{label} (p={p})')
            plt.legend(title='Winner')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f'correlation_space_{metric}_p{p}.png')
            plt.close()
