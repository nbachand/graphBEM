"""Combine saved diagnostic stages and plot before/after EnergyPlus errors."""
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis/physics_fixes'
STAGES = [
    ('01_radiation', '4c86e3e'),
    ('02_partition', 'deb6fc6'),
    ('03_volumes', '6066f03'),
    ('04_geometry', '0c96bb8'),
    ('05_coupling', '269cf80'),
    ('06_final', '002b349'),
]


def main():
    original = pd.read_csv(ROOT/'analysis/energyplus_wall_fix/metrics.csv')
    frames = [original[original.solver == 'before'].assign(stage='original'),
              original[original.solver == 'after'].assign(stage='wall_only')]
    for stage, _ in STAGES:
        frames.append(pd.read_csv(OUT/stage/'metrics.csv').assign(stage=stage))
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUT/'stages.csv', index=False)
    provenance = {stage: subprocess.check_output(['git', 'rev-parse', ref], cwd=ROOT, text=True).strip()
                  for stage, ref in STAGES}
    provenance['wall_only'] = subprocess.check_output(['git','rev-parse','11be057'],cwd=ROOT,text=True).strip()
    provenance['original'] = subprocess.check_output(['git','rev-parse','ca77fca'],cwd=ROOT,text=True).strip()
    (OUT/'stages.json').write_text(json.dumps(provenance,indent=2)+'\n')
    rows = [
        ('Indoor air temperature', 'OD', 'front', 'Ta', 'K'),
        ('Exterior wall surface temperature', 'OD', 'back', 'Ts', 'K'),
        ('Interior wall-face conduction', 'OD', 'front', 'cond', 'W/m2'),
        ('Exterior wall-face conduction', 'OD', 'back', 'cond', 'W/m2'),
        ('Exterior roof-face conduction', 'RF', 'back', 'cond', 'W/m2'),
        ('Exterior roof convection', 'RF', 'back', 'conv', 'W/m2'),
        ('Exterior roof radiation', 'RF', 'back', 'rad', 'W/m2'),
    ]
    records = []
    for label, boundary, side, quantity, unit in rows:
        values = combined[(combined.boundary == boundary) & (combined.side == side) & (combined.quantity == quantity)].set_index('stage').rmse
        records.append(dict(quantity=label, unit=unit, original=values['original'],
                            wall_only=values['wall_only'], all_fixes=values['06_final']))
    summary = pd.DataFrame(records)
    summary.to_csv(OUT/'summary.csv', index=False)
    print(summary.round(3).to_string(index=False))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    groups = [(rows[:2], ['Indoor air', 'Exterior wall surface']),
              (rows[2:5], ['Wall inside', 'Wall outside', 'Roof outside']),
              (rows[5:], ['Convection', 'Radiation'])]
    for ax, (selected, labels) in zip(axes, groups):
        subset = summary[summary.quantity.isin([row[0] for row in selected])]
        positions = np.arange(len(subset))
        for i, (column, label, color) in enumerate([('original','Original','#a5a5a5'),
                                                  ('wall_only','Wall fix only','#dd9c4a'),
                                                  ('all_fixes','All fixes','#3279aa')]):
            ax.bar(positions+(i-1)*.24, subset[column], width=.24, label=label, color=color)
        ax.set_xticks(positions,labels)
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y',alpha=.2)
    axes[0].set(title='Temperature RMSE', ylabel='K')
    axes[1].set(title='Conduction RMSE', ylabel='W/m²')
    axes[2].set(title='Exterior roof flux RMSE', ylabel='W/m²')
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Burbank diagnostic · 30 s timestep · same short run, no matched warm-up')
    fig.savefig(OUT/'comparison.png',dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
