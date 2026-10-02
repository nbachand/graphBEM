"""Summarize saved convection runs, flux errors and numerical refinement."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis/convection_models'
sys.path.insert(0, str(ROOT))
from scripts.compare_ep_free_running import score


def main():
    rows, flux_rows = [], []
    for folder in sorted(OUT.iterdir()):
        if not (folder/'metadata.json').exists():
            continue
        climate, variant = folder.name.rsplit('_', 1)
        if variant in ['1', '2', '3']:
            climate = climate.removesuffix('_fixed')
            variant = 'fixed_'+variant
        meta = json.loads((folder/'metadata.json').read_text())
        for excluded in [0, 7, 14]:
            rooms = pd.read_csv(folder/f'room_metrics_exclude_{excluded}d.csv')
            rows.append(dict(climate=climate, variant=variant, excluded_days=excluded,
                room_rmse_K=np.sqrt(np.mean(rooms.rmse**2)), room_bias_K=rooms.bias.mean(),
                max_energy_residual_W=meta['max_energy_residual_W'], warmup_days=meta['warmup_days']))
            flux = pd.read_csv(folder/f'surface_metrics_exclude_{excluded}d.csv')
            flux_rows.append(flux.assign(climate=climate, variant=variant, excluded_days=excluded))
    baseline = ROOT/'analysis/energyplus_free_running/base'
    meta = json.loads((baseline/'metadata.json').read_text())
    for excluded in [0, 7, 14]:
        rooms = pd.read_csv(baseline/f'room_metrics_exclude_{excluded}d.csv')
        rows.append(dict(climate='burbank', variant='legacy', excluded_days=excluded,
            room_rmse_K=np.sqrt(np.mean(rooms.rmse**2)), room_bias_K=rooms.bias.mean(),
            max_energy_residual_W=meta['max_energy_residual_W'],warmup_days=meta['warmup_days']))
        flux_rows.append(pd.read_csv(baseline/f'surface_metrics_exclude_{excluded}d.csv').assign(
            climate='burbank', variant='legacy', excluded_days=excluded))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT/'summary.csv', index=False)
    pd.concat(flux_rows).to_csv(OUT/'surface_summary.csv', index=False)
    coarse = pd.read_csv(OUT/'burbank_variable/room_timeseries.csv')
    fine = pd.read_csv(OUT/'burbank_refined/room_timeseries.csv')
    paired = coarse.merge(fine, on=['time', 'zone'], suffixes=('_coarse', '_fine'), validate='one_to_one')
    refinement = []
    for zone, frame in paired.groupby('zone'):
        error = frame.T_model_fine-frame.T_model_coarse
        refinement.append(dict(zone=zone, rms_change_K=np.sqrt(np.mean(error**2)), max_change_K=abs(error).max()))
    pd.DataFrame(refinement).to_csv(OUT/'refinement.csv', index=False)
    surface_coarse = pd.read_csv(OUT/'burbank_variable/surface_timeseries.csv.gz')
    surface_fine = pd.read_csv(OUT/'burbank_refined/surface_timeseries.csv.gz')
    joined = surface_coarse.merge(surface_fine, on=['time', 'surface_id', 'side', 'category', 'area'],
                                  suffixes=('_coarse', '_fine'), validate='one_to_one')
    refinement = []
    for (category, side), group in joined.groupby(['category', 'side']):
        for q in ['Ts', 'cond', 'conv', 'rad']:
            error = group[q+'_model_fine']-group[q+'_model_coarse']
            refinement.append(dict(category=category, side=side, quantity=q,
                rms_change=np.sqrt(np.average(error**2, weights=group.area)), max_change=abs(error).max()))
    pd.DataFrame(refinement).to_csv(OUT/'surface_refinement.csv', index=False)
    events = pd.read_csv(OUT/'wet_boundary_events.csv')
    dry_metrics = []
    for climate in ['burbank', 'palm_springs', 'arcata']:
        frame = pd.read_csv(OUT/f'{climate}_variable/surface_timeseries.csv.gz')
        frame = frame[(frame.side == 'outside') & frame.category.isin(['wall', 'roof'])]
        wet = events[events.climate == climate][['surface_id', 'time']].assign(wet=True)
        frame = frame.merge(wet, on=['surface_id', 'time'], how='left', validate='many_to_one')
        cutoff = pd.to_datetime(frame.time).min().normalize()+pd.Timedelta(days=7)
        frame = frame[(pd.to_datetime(frame.time) > cutoff) & frame.wet.isna()]
        dry_metrics.append(score(frame, ['category', 'side'], ['Ts', 'cond', 'conv', 'rad']).assign(climate=climate))
    pd.concat(dry_metrics).to_csv(OUT/'dry_surface_summary.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    order = ['legacy', 'fixed_1', 'fixed_2', 'fixed_3', 'variable']
    labels = ['Previous model', 'Fixed natural 1', 'Fixed natural 2', 'Fixed natural 3', 'TARP + DOE-2']
    for ax, climate in zip(axes, ['burbank', 'palm_springs', 'arcata']):
        subset = summary[(summary.climate == climate) & (summary.excluded_days == 7)].set_index('variant').loc[order]
        ax.barh(labels, subset.room_rmse_K, color=['.65', '#adc5dd', '#82a6ce', '#587fb0', '#217a69'])
        for i, val in enumerate(subset.room_rmse_K):
            ax.text(val+.015, i, f'{val:.3f}', va='center', fontsize=9)
        ax.set_title(climate.replace('_', ' ').title())
        ax.set_xlabel('Room temperature RMSE (°C)')
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(subset.room_rmse_K)*1.25)
    axes[0].invert_yaxis()
    fig.suptitle('Matched free-running comparison, August 8–31\nFixed cases retain DOE-2 wind; updated cases use the EnergyPlus sky split', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT/'comparison.png', dpi=160)
    plt.close(fig)
    print(summary[summary.excluded_days == 7].to_string(index=False))


if __name__ == '__main__':
    main()
