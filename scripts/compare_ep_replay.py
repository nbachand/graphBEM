"""Controlled conduction comparison using EnergyPlus SQL boundary histories.

Each physical opaque surface is simulated once, with exact construction layers.
Robin replay supplies EP air temperatures, convection coefficients and net
radiation. Dirichlet replay instead supplies both EP surface temperatures.
Neither is an independent test of room air, convection or radiation predictions.
SQL zone-timestep values are treated as endpoint samples. Robin inputs can be
held or linearly interpolated; Dirichlet temperatures use linear interpolation.
Endpoint predictions are compared with the corresponding EP zone-timestep value.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from model.WallSimulation import WallSimulation
from model.utils import WallSides


def read_case(case):
    path = (case / 'results/eplusout.sql').resolve()
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        surfaces = pd.read_sql('SELECT * FROM Surfaces', db).set_index('SurfaceIndex')
        constructions = pd.read_sql('SELECT * FROM Constructions', db).set_index('ConstructionIndex')
        layers = pd.read_sql('SELECT * FROM ConstructionLayers', db)
        materials = pd.read_sql('SELECT * FROM Materials', db).set_index('MaterialIndex')
        times = pd.read_sql('SELECT * FROM Time WHERE WarmupFlag=0 AND IntervalType=-1', db).set_index('TimeIndex')
        raw = pd.read_sql("""SELECT r.TimeIndex, d.KeyValue, d.Name, r.Value
            FROM ReportData r JOIN ReportDataDictionary d USING(ReportDataDictionaryIndex)
            JOIN Time t USING(TimeIndex)
            WHERE t.WarmupFlag=0 AND t.IntervalType=-1
            AND d.ReportingFrequency='Zone Timestep' AND d.Name LIKE 'Surface %'""", db)
    if times.EnvironmentPeriodIndex.nunique() != 1 or not (times.Interval == 15).all():
        raise ValueError('Expected a single run period with 15-minute zone timesteps')
    times['timestamp'] = pd.to_datetime(dict(year=times.Year, month=times.Month, day=times.Day)) + pd.to_timedelta(times.Hour * 60 + times.Minute, unit='min')
    if not (times.timestamp.diff().dropna() == pd.Timedelta(minutes=15)).all():
        raise ValueError('Noncontiguous zone-timestep history')
    data = {key: group.pivot(index='TimeIndex', columns='Name', values='Value').reindex(times.index)
            for key, group in raw.groupby('KeyValue')}
    builds = {}
    for cid, group in layers.groupby('ConstructionIndex'):
        # EnergyPlus layer 1 is outside; WallSimulation front is inside.
        mat = materials.loc[group.sort_values('LayerIndex', ascending=False).MaterialIndex].copy()
        mat['key'] = np.where(mat.ROnly.astype(bool), 'Material:AirGap', 'Material')
        mat = mat.rename(columns={'SpecHeat': 'Specific_Heat', 'Resistance': 'Thermal_Resistance'})
        mat.loc[mat.ROnly.astype(bool), 'Thickness'] = np.nan
        builds[cid] = mat
    return surfaces, constructions, builds, times, data


def face_data(frame, face, alpha=.7, ground=False):
    def get(label):
        value = frame[f'Surface {face} Face {label}'].to_numpy()
        if not np.isfinite(value).all():
            raise ValueError(f'Missing {face} {label}')
        return value
    ts = get('Temperature') + 273.15
    h = get('Convection Heat Transfer Coefficient')
    cond = get('Conduction Heat Transfer Rate per Area')
    conv = get('Convection Heat Gain Rate per Area')
    if face == 'Inside':
        air = get('Adjacent Air Temperature') + 273.15
        rad = sum(get(q + ' Heat Gain Rate per Area') for q in
                  ['Net Surface Thermal Radiation', 'Solar Radiation', 'Internal Gains Radiation'])
    else:
        air = get('Outdoor Air Drybulb Temperature') + 273.15
        rad = (np.zeros(len(frame)) if ground else
               alpha * get('Incident Solar Radiation Rate per Area') + get('Net Thermal Radiation Heat Gain Rate per Area'))
    return np.column_stack([ts, air, h, rad, cond, conv])


def make_wall(layers, cells, dt):
    wall = WallSimulation(X=1, Y=1, material_df=layers, h=WallSides(2., 2.),
                          roughness=WallSides(0., 0.), absorptivity=.7,
                          n=cells, delt=dt, implicit=True)
    wall.initialize(dt, 298.15, 298.15)
    return wall


def advance(wall, current, previous, mode, ground, substeps):
    """Advance production wall solver; return endpoint Ts and conductive flux."""
    for sub in range(substeps):
        if mode == 'dirichlet':
            air = previous[:, 0] + (current[:, 0] - previous[:, 0]) * (sub + 1) / substeps
            h = np.full(2, 1e8)
            rad = np.zeros(2)
        else:
            forcing = (previous + (current - previous) * (sub + 1) / substeps
                       if mode == 'robin_linear' else current)
            air, h, rad = (forcing[:, col].copy() for col in [1, 2, 3])
            if ground:
                air[1], h[1], rad[1] = forcing[1, 0], 1e8, 0.
        wall.h = WallSides(*h)
        wall.Erad = WallSides(*rad)
        wall.timeStep(*air)
    ts = wall.T_prof[[0, -1]]
    cond = np.array([wall.surface_conductance.front, wall.surface_conductance.back]) * (wall.T[[0, -1]] - ts)
    return np.column_stack([ts, cond])


def run_surface(layers, history, mode, ground, cells, dt):
    wall = make_wall(layers, cells, dt)
    substeps = round(900 / dt)
    # A periodic replay removes arbitrary initial state; it does not reproduce
    # EP's unreported warm-up state. Also exclude the first week from metrics.
    first_day = history[:96]
    for day in range(1, 101):
        before = wall.T.copy()
        previous = first_day[-1]
        for current in first_day:
            advance(wall, current, previous, mode, ground, substeps)
            previous = current
        change = float(np.max(np.abs(wall.T - before)))
        if change < 1e-5:
            break
    else:
        raise RuntimeError('First-day periodic spin-up did not converge')
    prediction = []
    previous = first_day[-1]
    for current in history:
        prediction.append(advance(wall, current, previous, mode, ground, substeps))
        previous = current
    return np.array(prediction), dict(spinup_days=day, spinup_change_K=change,
                                     cells=wall.n, resistance=wall.total_resistance,
                                     capacity_J_m2K=float(wall.capacity.sum()))


def metrics(frame, grouping):
    rows = []
    for keys, group in frame.groupby(grouping):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for q, unit in [('Ts', 'K'), ('cond', 'W/m2')]:
            # Prescribed temperatures are boundary inputs, not predictions.
            if q == 'Ts' and (group['mode'].iloc[0] == 'dirichlet' or
                             (group.category.iloc[0] == 'floor' and group.side.iloc[0] == 'outside')):
                continue
            error = group[q + '_model'] - group[q + '_ep']
            rows.append(dict(zip(grouping, keys)) | dict(quantity=q, unit=unit,
                rmse=float(np.sqrt(np.average(error**2, weights=group.area))),
                bias=float(np.average(error, weights=group.area)),
                max_abs=float(abs(error).max()), samples=len(group)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ep-case', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/energyplus_replay')
    parser.add_argument('--dt', type=float, default=60)
    parser.add_argument('--cells', type=int, default=36)
    parser.add_argument('--exclude-days', type=int, default=7)
    parser.add_argument('--modes', nargs='+', choices=['robin', 'robin_linear', 'dirichlet'], default=['robin_linear', 'dirichlet'])
    parser.add_argument('--surface-ids', nargs='+', type=int, help='Optional subset for refinement checks')
    args = parser.parse_args()
    if args.dt <= 0 or not np.isclose(900 / args.dt, round(900 / args.dt)):
        parser.error('--dt must divide 900 seconds')
    surfaces, constructions, builds, times, data = read_case(args.ep_case)
    records, audit, metadata = [], [], []
    for sid, surface in surfaces.iterrows():
        if args.surface_ids and sid not in args.surface_ids:
            continue
        pair = int(surface.ExtBoundCond)
        if pair > 0 and sid > pair:
            continue  # The same physical interzone partition is reported twice.
        if pair < -1:
            raise ValueError('Unsupported boundary condition')
        construction = constructions.loc[surface.ConstructionIndex]
        front = face_data(data[surface.SurfaceName], 'Inside')
        if pair > 0:
            back = face_data(data[surfaces.loc[pair].SurfaceName], 'Inside')
            if int(surfaces.loc[pair].ExtBoundCond) != sid:
                raise ValueError('Nonreciprocal interzone pairing')
            np.testing.assert_allclose(front[:, 0], data[surfaces.loc[pair].SurfaceName]['Surface Outside Face Temperature'] + 273.15, atol=.02, rtol=0)
        else:
            back = face_data(data[surface.SurfaceName], 'Outside', construction.OutsideAbsorpSolar, ground=pair == -1)
        history = np.stack([front, back], axis=1)
        category = 'partition' if pair > 0 else surface.ClassName.lower()
        for side, f in [('inside', front), ('outside', back)]:
            if side == 'outside' and pair == -1:
                continue
            residual = f[:, 4] + f[:, 5] + f[:, 3]
            conv_error = f[:, 2] * (f[:, 1] - f[:, 0]) - f[:, 5]
            audit.append(dict(surface_id=sid, side=side, ep_balance_max_W_m2=float(abs(residual).max()),
                              ep_convection_reconstruction_max_W_m2=float(abs(conv_error).max())))
        for mode in args.modes:
            prediction, info = run_surface(builds[surface.ConstructionIndex], history, mode,
                                            pair == -1, args.cells, args.dt)
            np.testing.assert_allclose(info['resistance'], 1 / construction.Uvalue, rtol=1e-10)
            metadata.append(dict(surface_id=sid, surface=surface.SurfaceName, mode=mode,
                                 category=category, area=surface.Area, **info))
            for j, side in enumerate(['inside', 'outside']):
                records.append(pd.DataFrame(dict(time=times.timestamp.to_numpy(), surface_id=sid,
                    mode=mode, category=category, side=side, area=surface.Area,
                    Ts_model=prediction[:, j, 0], Ts_ep=history[:, j, 0],
                    cond_model=prediction[:, j, 1], cond_ep=history[:, j, 4])))
            print(f'{sid:2d} {category:9s} {mode:9s}: spin-up {info["spinup_days"]} days', flush=True)
    all_results = pd.concat(records, ignore_index=True)
    cutoff = times.timestamp.iloc[0] - pd.Timedelta(minutes=15) + pd.Timedelta(days=args.exclude_days)
    scored = all_results[all_results.time > cutoff]
    if scored.empty:
        raise ValueError('No samples after excluded initialization period')
    args.output.mkdir(parents=True, exist_ok=True)
    summary = metrics(scored, ['mode', 'category', 'side'])
    summary.to_csv(args.output / 'metrics.csv', index=False)
    metrics(scored, ['mode', 'category', 'surface_id', 'side']).to_csv(args.output / 'surface_metrics.csv', index=False)
    pd.DataFrame(audit).to_csv(args.output / 'ep_balance_audit.csv', index=False)
    all_results.to_csv(args.output / 'timeseries.csv.gz', index=False, compression='gzip')
    details = dict(ep_case=str(args.ep_case.resolve()), timestep_seconds=args.dt,
        target_cells=args.cells, excluded_days=args.exclude_days, score_start=str(cutoff),
        end=str(times.timestamp.iloc[-1]), physical_surfaces=len(set(all_results.surface_id)),
        matched=['construction materials and inside/outside layer order', 'ground face temperature',
                 'each wall orientation separately', 'SQL zone-timestep timestamps',
                 'paired interzone faces; no duplicate partitions', 'surface-area weighting'],
        limitations=['Boundary replay, not a free-running building validation.',
            'Robin replay prescribes EP air temperatures, h and net radiation; these are not validated outputs.',
            'Dirichlet replay prescribes EP surface temperatures; only conduction is a prediction.',
            'Sub-timestep boundary histories and EP warm-up states are unavailable.',
            'robin holds inputs; robin_linear and dirichlet interpolate endpoint inputs linearly.',
            'Endpoint predictions compared with reported zone-timestep values; no hourly records included.'],
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [args.ep_case / 'results/eplusout.sql', Path(__file__), ROOT / 'model/WallSimulation.py']},
        surfaces=metadata)
    (args.output / 'metadata.json').write_text(json.dumps(details, indent=2) + '\n')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
