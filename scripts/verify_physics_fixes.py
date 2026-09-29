"""Final EnergyPlus comparison, whole-building energy budget and timestep study.

Run from the repository root:
    python scripts/verify_physics_fixes.py --ep-case /path/to/Burbank/case

The same 41.658-hour forcing is interpolated onto each time grid. The initial
room temperatures and ground temperature are held fixed across refinements.
"""
import argparse
import hashlib
import subprocess
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_wall_solver import weather_sample, run_graph, read_ep, compare
from model.BuildingSimulation import BuildingSimulation
from model.WallSimulation import WallSimulation


def energy_budget(sim):
    """All wall storage + free room air, excluding prescribed boundary air."""
    storage = np.zeros(sim.N-1)
    boundary_input = np.zeros(sim.N-1)
    internal_radiation = np.zeros(sim.N-1)
    for _, _, edge in sim.bG.G.edges(data=True):
        w = edge['wall']
        area = w.Af * edge['weight']
        profiles = edge['T_profs']
        storage += area * (w.capacity @ np.diff(profiles[1:-1], axis=1)) / sim.delt
        for side, j in [('front', 0), ('back', -1)]:
            node = getattr(edge['nodes'], side)
            rad = getattr(edge['radEApplied'], side)[1:]
            if node in sim.Tbound:
                air = sim.bG.G.nodes[node]['Tints'][:-1]
                conv = getattr(edge['hCalced'], side)[1:] * (air-profiles[j, 1:])
                boundary_input += area*(rad+conv)
            else:
                internal_radiation += area*rad
    for node, data in sim.bG.G.nodes(data=True):
        if node in sim.Tbound:
            continue
        room = data['room']
        storage += room.rho*room.Cp*room.V*np.diff(data['Tints'])/sim.delt
        boundary_input += room.Eint
        if data['vent'].ventType == 'HWP4':
            boundary_input += room.rho*room.Cp*data['Vnvs'][1:]*(sim.Tbound['OD'][1:]-data['Tints'][:-1])
        elif data['vent'].ventType is not None:
            raise ValueError('Budget currently supports inactive or HWP4 ventilation')
    residual = storage-boundary_input
    return dict(max_abs_residual_W=float(np.max(np.abs(residual))),
                rms_residual_W=float(np.sqrt(np.mean(residual**2))),
                integrated_residual_J=float(np.sum(residual)*sim.delt),
                max_abs_internal_radiation_W=float(np.max(np.abs(internal_radiation))),
                max_abs_external_power_W=float(np.max(np.abs(boundary_input))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ep-case', type=Path, required=True)
    args = parser.parse_args()
    out = ROOT/'analysis/physics_fixes'
    out.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [ROOT/'myBuilding.py', *sorted((ROOT/'model').glob('*.py'))]}
    base = weather_sample()
    initial_air = float(base.temp_air.mean()+273.15)
    original_initialize = BuildingSimulation.initialize
    def initialize(sim, graph, **kwargs):
        sim.Tbound['FL'][:] = initial_air-4.25
        for _, data in graph.G.nodes(data=True):
            data['room_kwargs']['T0'] = initial_air
        original_initialize(sim, graph, **kwargs)
    ep = read_ep(args.ep_case, base.index[0].year)
    temperatures, surfaces, budgets, metrics = {}, {}, {}, []
    for dt in [30, 15, 7.5]:
        # Numeric input columns only; time range and interpolation fixed to baseline.
        sample = base.resample(pd.Timedelta(seconds=dt)).interpolate()
        with patch.object(BuildingSimulation, 'initialize', initialize):
            graph, _, sim = run_graph(sample, WallSimulation, return_sim=True)
        budgets[str(dt)] = energy_budget(sim)
        assert budgets[str(dt)]['max_abs_residual_W'] < 1e-4, budgets[str(dt)]
        stride = int(30/dt)
        temperatures[dt] = np.array([sim.bG.G.nodes[r]['Tints'][::stride] for r in ['CR','SS','DR','CV']])
        surfaces[dt] = np.array([edge['T_profs'][[0,-1], ::stride]
                                for _,_,edge in sim.bG.G.edges(data=True)])
        assert np.isfinite(temperatures[dt]).all() and np.isfinite(surfaces[dt]).all()
        metrics.append(compare(graph, ep, sample).assign(timestep_seconds=dt))
    convergence = []
    for coarse, fine in [(30,15), (15,7.5)]:
        air_delta = temperatures[coarse]-temperatures[fine]
        surface_delta = surfaces[coarse]-surfaces[fine]
        convergence.append(dict(coarse_seconds=coarse, fine_seconds=fine,
            air_rms_difference_K=float(np.sqrt(np.mean(air_delta**2))),
            air_max_difference_K=float(np.max(np.abs(air_delta))),
            surface_rms_difference_K=float(np.sqrt(np.mean(surface_delta**2))),
            surface_max_difference_K=float(np.max(np.abs(surface_delta)))))
    assert convergence[1]['air_rms_difference_K'] < convergence[0]['air_rms_difference_K']
    assert convergence[1]['surface_rms_difference_K'] < convergence[0]['surface_rms_difference_K']
    report = dict(budgets=budgets, convergence=convergence, initial_air_K=initial_air,
                  model_revision=revision, model_hashes=hashes,
                  note='Current corrected model; same initial and ground temperatures on every grid.')
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.concat(metrics).to_csv(out/'final_comparison.csv',index=False)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
