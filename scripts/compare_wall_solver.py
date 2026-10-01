"""Reproduce the Burbank diagnostic, optionally comparing a saved older solver.

Run from the repository root with the graphBEM environment:
    python scripts/compare_wall_solver.py --ep-case /path/to/Burbank/case \
        --baseline-wall /path/to/old/WallSimulation.py

The case must contain surface_data.csv and results/eplusout.sql. This is the
notebook's short diagnostic run, not a validation with matched warm-up/forcing.
Fluxes are W/m2, positive toward each surface. GraphBEM end-of-step samples
are averaged into 15-minute intervals. EnergyPlus CSV labels denote interval
starts (the first SQL interval ends at 00:15, labelled 00:00 in the CSV).
Orientations are area-averaged within each zone; zones have equal weight in
reported RMSE. Both series use local standard clock time and the weather year.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from model import WallSimulation as ws
from model.BuildingSimulation import BuildingSimulation
from myBuilding import runMyBEM
from runMyBuildingMC import getConstructions
from energyPlus.weather.weather import process_epw_file, getShortwaveRadiation, getLongwaveRadiation

WEATHER = 'CA_BURBANK-GLNDLE-PASAD-AP_722880S_CZ2022'
QUANTITIES = ['Ts', 'Ta', 'h', 'cond', 'conv', 'rad']


def weather_sample():
    data, meta = process_epw_file(str(ROOT / 'energyPlus/weather/CAClimateZones' / WEATHER / (WEATHER+'.epw')))
    data['Latitude'] = 34.2
    for prefix, tilt, azimuths in [('Horizontal', 0, [0]), ('Vertical', 90, range(0, 360, 5))]:
        data[prefix+' Shortwave Radiation'] = getShortwaveRadiation(data, meta, tilts=[tilt], azimuths=azimuths)['poa_global']
        data[prefix+' Sky Longwave Radiation'], data[prefix+' Surfaces Longwave Radiation'] = getLongwaveRadiation(data, tilt, R_obs=0)
    return data[data.index.month == 8].iloc[1:].resample('30s').interpolate().iloc[:5000]


def run_graph(sample, wall_class, return_sim=False):
    if not hasattr(wall_class, 'convection_metadata'):
        # Saved historical solvers predate explicit model configuration.
        class LegacyAdapter(wall_class):
            def __init__(self, **kwargs):
                extra = {key: kwargs.pop(key) for key in
                         ('convection', 'tilt', 'azimuth', 'wind_exposure') if key in kwargs}
                super().__init__(**kwargs)
                self.__dict__.update(extra)
            def initialize(self, *args, windDirection=None, **kwargs):
                return super().initialize(*args, **kwargs)
            def convection_metadata(self):
                return {'model': 'historical solver'}
        wall_class = LegacyAdapter
    captured = []
    original_run = BuildingSimulation.run
    def capture(sim):
        original_run(sim)
        captured.append(sim)
    materials = getConstructions('My', constructionFile=str(ROOT/'energyPlus/My_Constructions.csv'),
                                 materialFile=str(ROOT/'energyPlus/ASHRAE_2005_HOF_Materials.csv'))
    with patch.object(ws, 'WallSimulation', wall_class), patch.object(BuildingSimulation, 'run', capture):
        runMyBEM(sample, materials, -4.25, 2, 2, 1.64, .75, makePlots=False, interior_convection="legacy", exterior_convection="legacy")
    sim = captured[0]
    frames, balances = [], []
    for a, b, d in sim.bG.G.edges(data=True):
        w, profiles = d['wall'], d['T_profs']
        surface_fluxes = []
        for side, j, node in [('front', 0, d['nodes'].front), ('back', -1, d['nodes'].back)]:
            air = sim.bG.G.nodes[node]['Tints'][:-1]  # air actually used by wall solve
            h = getattr(d['hCalced'], side)[1:]
            rad = getattr(d['radEApplied'], side)[1:]
            if hasattr(w, 'surface_conductance'):
                g = getattr(w.surface_conductance, side)
            else:
                g = w.kfs[j] / w.delx
            cell = 1 if j == 0 else -2
            cond = g * (profiles[cell, 1:] - profiles[j, 1:])
            conv = h * (air - profiles[j, 1:])
            residual = cond + conv + rad
            surface_fluxes.append(conv + rad)
            balances.append(dict(zone=a, boundary=b, side=side,
                                 surface_rms=np.sqrt(np.mean(residual**2)),
                                 surface_max_abs=np.max(np.abs(residual))))
            if b in ['OD', 'RF', 'FL']:
                frames.append(pd.DataFrame(dict(time=sample.index[1:].tz_localize(None), zone=a, boundary=b,
                                               side=side, Ts=profiles[j, 1:], Ta=air, h=h,
                                               rad=rad, conv=conv, cond=cond)))
        if hasattr(w, 'capacity'):
            storage = w.capacity @ np.diff(profiles[1:-1], axis=1) / sim.delt
            residual = storage - sum(surface_fluxes)
            balances.append(dict(zone=a, boundary=b, side='storage',
                                 surface_rms=np.sqrt(np.mean(residual**2)),
                                 surface_max_abs=np.max(np.abs(residual))))
    result = (pd.concat(frames, ignore_index=True), pd.DataFrame(balances))
    return (*result, sim) if return_sim else result


def read_ep(case, year):
    e = pd.read_csv(case/'surface_data.csv')
    with sqlite3.connect((case/'results/eplusout.sql').resolve().as_uri()+'?mode=ro', uri=True) as db:
        surfaces = pd.read_sql('select SurfaceName, Area, ConstructionIndex from Surfaces', db)
        constructions = pd.read_sql('select ConstructionIndex, OutsideAbsorpSolar from Constructions', db)
    e = e.merge(surfaces, left_on='space_names', right_on='SurfaceName', validate='many_to_one')
    e = e.merge(constructions, on='ConstructionIndex', validate='many_to_one')
    e['time'] = pd.to_datetime(e.datetimes).map(lambda t: t.replace(year=year))
    e['boundary'] = np.where(e.direction == 'UP', 'RF', np.where(e.direction == 'DOWN', 'FL', 'OD'))
    e = e[e.is_exterior | (e.boundary == 'FL')].copy()
    e.zone = e.zone.map({'corner_ventilation': 'CR', 'single_sided_ventilation': 'SS',
                         'dual_room_ventilation': 'DR', 'cross_ventilation': 'CV'})
    if e.zone.isna().any():
        raise ValueError('Unmapped EnergyPlus zones')
    return e


def compare(graph, ep, sample):
    metrics = []
    for side, face in [('front', 'Inside'), ('back', 'Outside')]:
        e = ep.copy()
        e['Ts'] = e[f'Surface {face} Face Temperature'] + 273.15
        e['h'] = e[f'Surface {face} Face Convection Heat Transfer Coefficient']
        e['cond'] = e[f'Surface {face} Face Conduction Heat Transfer Rate per Area']
        if side == 'front':
            e['Ta'] = e['Surface Inside Face Adjacent Air Temperature'] + 273.15
            e['conv'] = e.h * (e.Ta - e.Ts)
            e['rad'] = sum(e[f'Surface Inside Face {q} Heat Gain Rate per Area'] for q in
                           ['Net Surface Thermal Radiation', 'Solar Radiation', 'Internal Gains Radiation'])
        else:
            e = e[e.boundary != 'FL'].copy()
            e['Ta'] = e['Surface Outside Face Outdoor Air Drybulb Temperature'] + 273.15
            e['conv'] = e['Surface Outside Face Convection Heat Gain Rate per Area']
            e['rad'] = (e.OutsideAbsorpSolar * e['Surface Outside Face Incident Solar Radiation Rate per Area']
                        + e['Surface Outside Face Net Thermal Radiation Heat Gain Rate per Area'])
        for q in QUANTITIES:
            e[q] *= e.Area
        e = e.groupby(['time', 'zone', 'boundary'])[QUANTITIES+['Area']].sum()
        e[QUANTITIES] = e[QUANTITIES].div(e.Area, axis=0)
        g = graph[graph.side == side].copy()
        g['time'] = (g.time-pd.Timedelta(seconds=1)).dt.floor('15min')
        g = g.groupby(['time', 'zone', 'boundary'])[QUANTITIES].mean()
        joined = g.join(e[QUANTITIES], how='inner', lsuffix='_g', rsuffix='_e').reset_index()
        # Exclude the incomplete final averaging interval.
        end = sample.index[-1].tz_localize(None).floor('15min')
        joined = joined[joined.time < end]
        if joined.empty:
            raise ValueError('No overlapping comparison intervals')
        for boundary, subset in joined.groupby('boundary'):
            for q in QUANTITIES:
                error = subset[q+'_g'] - subset[q+'_e']
                metrics.append(dict(boundary=boundary, side=side, quantity=q,
                                    rmse=np.sqrt(np.mean(error**2)), bias=error.mean(),
                                    graph_mean=subset[q+'_g'].mean(), ep_mean=subset[q+'_e'].mean(),
                                    n=len(subset)))
    return pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ep-case', type=Path, required=True)
    parser.add_argument('--baseline-wall', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'analysis/energyplus_wall_fix')
    args = parser.parse_args()
    sample = weather_sample()
    ep = read_ep(args.ep_case, sample.index[0].year)
    solvers = {'after': ws.WallSimulation}
    if args.baseline_wall:
        spec = importlib.util.spec_from_file_location('baseline_wall', args.baseline_wall)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        solvers = {'before': module.WallSimulation, **solvers}
    metrics, balances = [], []
    for label, solver in solvers.items():
        graph, balance = run_graph(sample, solver)
        metrics.append(compare(graph, ep, sample).assign(solver=label))
        balances.append(balance.assign(solver=label))
    args.output.mkdir(parents=True, exist_ok=True)
    pd.concat(metrics).to_csv(args.output/'metrics.csv', index=False)
    pd.concat(balances).to_csv(args.output/'balances.csv', index=False)
    metadata = dict(weather=WEATHER, start=str(sample.index[0]), end=str(sample.index[-1]),
                    timestep_seconds=30, samples=len(sample), ep_case=str(args.ep_case),
                    baseline_wall=str(args.baseline_wall) if args.baseline_wall else None,
                    baseline_sha256=hashlib.sha256(args.baseline_wall.read_bytes()).hexdigest() if args.baseline_wall else None,
                    solver_sha256=hashlib.sha256((ROOT/'model/WallSimulation.py').read_bytes()).hexdigest(),
                    model_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    model_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [ROOT/'myBuilding.py', *sorted((ROOT/'model').glob('*.py'))]},
                    note='Short diagnostic without matched warm-up; full 15-minute intervals only. '
                         'Flux RMSE in W/m2; temperature RMSE in K; h RMSE in W/(m2 K). '
                         'Conduction positive toward each surface; convection/radiation positive into surface.')
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(pd.concat(metrics).pivot(index=['boundary', 'side', 'quantity'], columns='solver', values='rmse').round(3))
    print('Maximum balance residuals (W/m2):')
    print(pd.concat(balances).groupby(['solver', 'side']).surface_max_abs.max())


if __name__ == '__main__':
    main()
