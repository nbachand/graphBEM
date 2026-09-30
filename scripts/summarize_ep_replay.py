"""Summarize the matched replay and its numerical/initialization sensitivities."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scripts.compare_ep_replay import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'analysis/energyplus_replay')
    args = parser.parse_args()
    frames, summaries = {}, []
    for name in ['coarse', 'refined', 'matched', 'convergence']:
        folder = args.input / name
        frame = pd.read_csv(folder / 'timeseries.csv.gz', parse_dates=['time'])
        frames[name] = frame
        start = frame.time.min() - pd.Timedelta(minutes=15)
        scored = frame[frame.time > start + pd.Timedelta(days=7)]
        summary = metrics(scored, ['mode', 'category', 'side'])
        # Also refresh older output tables to exclude prescribed temperatures.
        summary.to_csv(folder / 'metrics.csv', index=False)
        metrics(scored, ['mode', 'category', 'surface_id', 'side']).to_csv(folder / 'surface_metrics.csv', index=False)
        summaries.append(summary.assign(run=name))
    pd.concat(summaries).to_csv(args.input / 'summary.csv', index=False)
    matched = frames['matched']
    start = matched.time.min() - pd.Timedelta(minutes=15)
    sensitivity = [metrics(matched[matched.time > start + pd.Timedelta(days=days)],
                           ['mode', 'category', 'side']).assign(excluded_days=days)
                   for days in [7, 14, 21]]
    pd.concat(sensitivity).to_csv(args.input / 'initialization_sensitivity.csv', index=False)
    keys = ['time', 'surface_id', 'mode', 'category', 'side', 'area']
    joined = matched.merge(frames['convergence'], on=keys, suffixes=('_base', '_fine'), validate='one_to_one')
    joined = joined[joined.time > start + pd.Timedelta(days=7)]
    rows = []
    for key, group in joined.groupby(['mode', 'category', 'side']):
        for q in ['Ts', 'cond']:
            if q == 'Ts' and (key[0] == 'dirichlet' or (key[1] == 'floor' and key[2] == 'outside')):
                continue
            delta = group[q + '_model_fine'] - group[q + '_model_base']
            rows.append(dict(zip(['mode', 'category', 'side'], key)) | dict(quantity=q,
                refinement_change_rmse=float(np.sqrt(np.average(delta**2, weights=group.area)))))
    pd.DataFrame(rows).to_csv(args.input / 'refinement_changes.csv', index=False)

    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True, constrained_layout=True)
    window = matched[(matched.time > start + pd.Timedelta(days=14)) &
                     (matched.time <= start + pd.Timedelta(days=17)) & (matched['mode'] == 'dirichlet')]
    for col, (sid, title) in enumerate([(1, 'Exterior wall'), (6, 'Roof'), (5, 'Floor'), (3, 'Partition')]):
        for row, side in enumerate(['inside', 'outside']):
            f = window[(window.surface_id == sid) & (window.side == side)]
            ax = axes[row, col]
            ax.plot(f.time, f.cond_ep, color='black', linewidth=1.8, label='EnergyPlus')
            ax.plot(f.time, f.cond_model, color='#2389c9', linewidth=1.2, linestyle='--', label='GraphBEM')
            ax.set_title(f'{title}, {side} face')
            ax.set_ylabel('Conduction (W/m²)')
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle('Conduction replay with EnergyPlus surface temperatures prescribed\n'
                 'One surface of each type; positive flux points from the construction toward each face')
    fig.savefig(args.input / 'comparison.png', dpi=170)
    plt.close(fig)
    print(pd.concat(summaries).query("run == 'matched'").to_string(index=False))


if __name__ == '__main__':
    main()
