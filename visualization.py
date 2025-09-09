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
