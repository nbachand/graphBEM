"""Free-running four-zone GraphBEM benchmark using EnergyPlus case geometry.

Only geometry, materials and exterior forcing enter the simulation. EnergyPlus
room/surface temperatures and heat flows are read solely for comparison. This
uses production WallSimulation, RoomSimulation and Radiation physics with a
streaming driver to retain individual exterior orientations and bound memory.
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
from model.RoomSimulation import RoomSimulation
from model.Radiation import Radiation, incident_longwave
from model.utils import WallSides
from scripts.compare_ep_replay import read_case, face_data

SIGMA = 5.67e-8
# EnergyPlus 22.2 SQL uses zero-based roughness enums; IDF Smooth is 4.
# DOE-2 material multipliers, ordered VeryRough through VerySmooth.
ROUGHNESS_MULTIPLIERS = {0: 2.17, 1: 1.67, 2: 1.52, 3: 1.13, 4: 1.11, 5: 1.0}


def read_zones(case, times):
    with sqlite3.connect((case / 'results/eplusout.sql').resolve().as_uri() + '?mode=ro', uri=True) as db:
        zones = pd.read_sql('SELECT * FROM Zones', db).set_index('ZoneIndex')
        raw = pd.read_sql("""SELECT r.TimeIndex,d.KeyValue,r.Value FROM ReportData r
            JOIN ReportDataDictionary d USING(ReportDataDictionaryIndex)
            WHERE d.Name='Zone Mean Air Temperature' AND d.ReportingFrequency='Zone Timestep'""", db)
    return zones, raw.pivot(index='TimeIndex', columns='KeyValue', values='Value').reindex(times.index)


def weather_ir(epw, times):
    """EPW hour 1 is the 01:00 endpoint, in local standard time (no DST)."""
    raw = pd.read_csv(epw, skiprows=8, header=None)
    stamp = pd.to_datetime(dict(year=np.full(len(raw), int(times.Year.iloc[0])),
                               month=raw[1], day=raw[2])) + pd.to_timedelta(raw[3], unit='h')
    ir = raw[12].to_numpy(float)
    if not np.all((ir > 0) & (ir < 9999)):
        raise ValueError('This benchmark requires valid EPW horizontal infrared')
    result = np.interp(times.timestamp.astype('int64'), stamp.astype('int64'), ir)
    # This case initializes the first hour from hour 24 of its first day.
    # Its SQL dry-bulb samples (20.575,20.15,19.725,19.3 C) verify that choice.
    start = times.timestamp.iloc[0].normalize()
    last = float(ir[np.asarray(stamp == start+pd.Timedelta(days=1))][0])
    first = float(ir[np.asarray(stamp == start+pd.Timedelta(hours=1))][0])
    fraction = (times.timestamp-start).dt.total_seconds().to_numpy()/3600
    mask = fraction <= 1
    result[mask] = last+(first-last)*fraction[mask]
    return result


def radiation_operator(rad):
    """Precompute the exact linear map used by Radiation.timeStep (T**4 -> q)."""
    names = list(rad.G)
    lap = np.zeros((len(names), len(names)))
    for a, b, edge in rad.G.edges(data=True):
        i, j = names.index(a), names.index(b)
        g = 1 / edge['radianceResistance']
        lap[i, i] -= g
        lap[j, j] -= g
        lap[i, j] += g
        lap[j, i] += g
    area = np.array([rad.G.nodes[n]['A'] for n in names])
    return (lap @ np.linalg.solve(rad.A.to_numpy(), np.eye(len(names)))) / area[:, None] * rad.sigma


class FreeBuilding:
    def __init__(self, zones, surfaces, constructions, builds, dt=60, cells=18,
                 interior="legacy", exterior="legacy", h_natural=2., sky="isotropic"):
        self.dt = dt
        self.models = dict(interior=interior, exterior=exterior, h_natural=h_natural, sky=sky)
        self.rooms = {}
        self.walls = []
        self.enclosures = []
        self.audit = []
        for zid, zone in zones.iterrows():
            if zone.Multiplier != 1 or zone.ListMultiplier != 1 or zone.ExtWindowArea != 0:
                raise ValueError('Only unmultiplied opaque zones are supported')
            room = RoomSimulation(T0=298.15, V=zone.Volume, Eint=0)
            room.initialize(dt)
            self.rooms[zid] = room
        for sid, surface in surfaces.iterrows():
            pair = int(surface.ExtBoundCond)
            if pair > 0 and sid > pair:
                continue
            if pair < -1 or surface.ClassName not in ['Wall', 'Floor', 'Roof']:
                raise ValueError('Unsupported surface')
            front = int(surface.ZoneIndex)
            back = int(surfaces.loc[pair].ZoneIndex) if pair > 0 else None
            if pair > 0:
                paired = surfaces.loc[pair]
                assert paired.ExtBoundCond == sid
                np.testing.assert_allclose(paired.Area, surface.Area)
                np.testing.assert_allclose(builds[paired.ConstructionIndex].Thermal_Resistance.sum(),
                                           builds[surface.ConstructionIndex].Thermal_Resistance.sum())
            zone = zones.loc[front]
            if surface.ClassName in ['Floor', 'Roof']:
                x, y = zone.MaximumX-zone.MinimumX, zone.MaximumY-zone.MinimumY
            else:
                x, y = surface.Area / zone.CeilingHeight, zone.CeilingHeight
            np.testing.assert_allclose(x*y, surface.Area, atol=1e-8)
            construction = constructions.loc[surface.ConstructionIndex]
            roughness = (ROUGHNESS_MULTIPLIERS[int(builds[surface.ConstructionIndex].iloc[-1].Roughness)]
                         if pair == 0 else 0.)
            wall = WallSimulation(X=x, Y=y, material_df=builds[surface.ConstructionIndex],
                h=WallSides(h_natural, 1e8 if pair == -1 else h_natural),
                convection=WallSides(interior, "fixed" if pair == -1 else interior if pair > 0 else exterior),
                tilt=WallSides(180-surface.Tilt, surface.Tilt),
                azimuth=WallSides((surface.Azimuth+180)%360, surface.Azimuth),
                roughness=WallSides(0., roughness),
                absorptivity=construction.OutsideAbsorpSolar, n=cells, delt=dt, implicit=True)
            wall.initialize(dt, 298.15, 291.15 if pair == -1 else 298.15, windDirection=0.)
            np.testing.assert_allclose(wall.total_resistance, 1/construction.Uvalue, rtol=1e-10)
            category = 'partition' if pair > 0 else surface.ClassName.lower()
            self.walls.append(dict(sid=sid, pair=pair, front=front, back=back, wall=wall,
                                   surface=surface, category=category,
                                   exterior_emissivity=construction.OutsideAbsorpThermal))
            self.audit.append(dict(surface_id=sid, paired_surface=pair, front_zone=front,
                back_zone=back, category=category, area_m2=surface.Area, azimuth=surface.Azimuth,
                tilt=surface.Tilt, cells=wall.n, resistance_m2K_W=wall.total_resistance,
                capacity_J_m2K=float(wall.capacity.sum()), roughness_multiplier=roughness))
        for zid, zone in zones.iterrows():
            adjacency, refs = {}, {}
            for entry in self.walls:
                if zid not in [entry['front'], entry['back']]:
                    continue
                side = 0 if zid == entry['front'] else 1
                key = {'roof':'RF', 'floor':'FL'}.get(entry['category'], str(entry['sid']))
                if key in adjacency:
                    raise ValueError('Expected one floor and roof per zone')
                # Radiation identifies the room-side face from the neighbor key.
                nodes = WallSides(zid, key) if side == 0 else WallSides(key, zid)
                adjacency[key] = dict(wall=entry['wall'], weight=1., nodes=nodes)
                refs[key] = (self.walls.index(entry), side)
            eps = constructions.loc[surfaces.loc[surfaces.ZoneIndex == zid, 'ConstructionIndex'], 'InsideAbsorpThermal']
            if eps.nunique() != 1:
                raise ValueError('Current room benchmark requires uniform interior emissivity')
            rad = Radiation(solveType='room', storyHeight=zone.CeilingHeight, emissivity=eps.iloc[0])
            rad.initialize(adjacency)
            self.enclosures.append((radiation_operator(rad), [refs[n] for n in rad.G]))
        self.max_energy_residual = 0.

    def state(self):
        return np.r_[[r.Tint for r in self.rooms.values()], *[e['wall'].T for e in self.walls]]

    def energy(self):
        return sum(r.rho*r.Cp*r.V*r.Tint for r in self.rooms.values()) + sum(
            e['wall'].Af * (e['wall'].capacity @ e['wall'].T) for e in self.walls)

    def step(self, forcing):
        before = self.energy()
        rad = np.zeros((len(self.walls), 2))
        for operator, refs in self.enclosures:
            temperatures = np.array([self.walls[i]['wall'].T_prof[0 if side == 0 else -1] for i, side in refs])
            fluxes = operator @ temperatures**4
            for (i, side), flux in zip(refs, fluxes):
                rad[i, side] = flux
        external_power = 0.
        room_power = dict.fromkeys(self.rooms, 0.)
        result = np.zeros((len(self.walls), 2, 4))
        for i, entry in enumerate(self.walls):
            wall, pair = entry['wall'], entry['pair']
            front_air = self.rooms[entry['front']].Tint
            if pair > 0:
                back_air = self.rooms[entry['back']].Tint
            elif pair == -1:
                back_air = 291.15
            else:
                back_air, wind, sw, ir = forcing[i, :4]
                if forcing.shape[1] > 4:
                    wall.windDirection = forcing[i, 4]
                wall.windSpeed = wind
                longwave = incident_longwave(ir, back_air, entry["surface"].Tilt, self.models["sky"])
                rad[i, 1] = wall.absorptivity*sw + entry['exterior_emissivity']*(longwave-SIGMA*wall.T_prof[-1]**4)
            wall.Erad = WallSides(*rad[i])
            power = wall.timeStep(front_air, back_air)
            room_power[entry['front']] += power.front
            if pair > 0:
                room_power[entry['back']] += power.back
            ts = wall.T_prof[[0, -1]]
            cond = np.array([wall.surface_conductance.front, wall.surface_conductance.back]) * (wall.T[[0, -1]]-ts)
            if pair <= 0:
                # Conductive boundary flux avoids cancellation in h*(Ts-Tground)
                # for the nearly prescribed ground surface (h=1e8).
                external_power -= wall.Af*cond[1]
            conv = -np.array([power.front, power.back])/wall.Af
            result[i] = np.column_stack([ts, cond, conv, rad[i]])
        for zid, room in self.rooms.items():
            room.timeStep(room_power[zid], 0.)
        residual = (self.energy()-before)/self.dt-external_power
        self.max_energy_residual = max(self.max_energy_residual, abs(residual))
        return result, np.array([r.Tint for r in self.rooms.values()])


def run_intervals(building, forcing, previous, record=True):
    surfaces, rooms = [], []
    substeps = round(900/building.dt)
    for current in forcing:
        for sub in range(substeps):
            fraction = (sub+1)/substeps
            sample = (1-fraction)*previous + fraction*current
            if sample.shape[1] > 4:
                sample[:, 4] = current[:, 4]  # EP interval direction; never interpolate across north.
            endpoint = building.step(sample)
        if record:
            surfaces.append(endpoint[0])
            rooms.append(endpoint[1])
        previous = current
    return np.asarray(surfaces), np.asarray(rooms)


def build_forcing(building, data, ir):
    directional = building.models["exterior"] in {"doe2", "doe2_fixed_natural"}
    forcing = np.zeros((len(ir), len(building.walls), 5 if directional else 4))
    for i, entry in enumerate(building.walls):
        if entry['pair'] != 0:
            continue
        frame = data[entry['surface'].SurfaceName]
        forcing[:, i, 0] = frame['Surface Outside Face Outdoor Air Drybulb Temperature']+273.15
        forcing[:, i, 1] = frame['Surface Outside Face Outdoor Air Wind Speed']
        forcing[:, i, 2] = frame['Surface Outside Face Incident Solar Radiation Rate per Area']
        forcing[:, i, 3] = ir
        if directional:
            forcing[:, i, 4] = frame["Surface Outside Face Outdoor Air Wind Direction"]
    if not np.isfinite(forcing).all():
        raise ValueError('Nonfinite exterior forcing')
    return forcing


def score(frame, keys, quantities):
    rows = []
    for group_keys, group in frame.groupby(keys):
        group_keys = group_keys if isinstance(group_keys, tuple) else (group_keys,)
        for q in quantities:
            if q in ['Ts', 'conv', 'rad'] and 'category' in group and group.category.iloc[0] == 'floor' and group.side.iloc[0] == 'outside':
                continue
            error = group[q+'_model']-group[q+'_ep']
            weights = group.area if 'area' in group else np.ones(len(group))
            rows.append(dict(zip(keys, group_keys)) | dict(quantity=q,
                rmse=float(np.sqrt(np.average(error**2, weights=weights))),
                bias=float(np.average(error, weights=weights)), max_abs=float(abs(error).max())))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ep-case', type=Path, required=True)
    parser.add_argument('--epw', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'analysis/convection_models/burbank_variable')
    parser.add_argument('--dt', type=float, default=60)
    parser.add_argument('--cells', type=int, default=18)
    parser.add_argument('--days', type=int, default=31)
    parser.add_argument('--warmup-tolerance', type=float, default=.001)
    parser.add_argument('--interior', choices=['legacy', 'fixed', 'tarp'], default='tarp')
    parser.add_argument('--exterior', choices=['legacy', 'fixed', 'doe2', 'doe2_fixed_natural'], default='doe2')
    parser.add_argument('--h-natural', type=float, default=2.)
    parser.add_argument('--sky', choices=['isotropic', 'energyplus'], default='energyplus')
    args = parser.parse_args()
    if args.dt <= 0 or not np.isclose(900/args.dt, round(900/args.dt)) or not 1 <= args.days <= 31 or args.warmup_tolerance <= 0:
        parser.error('dt must divide 900; days must be 1..31; warmup tolerance must be positive')
    surfaces, constructions, builds, times, data = read_case(args.ep_case)
    zones, ep_rooms = read_zones(args.ep_case, times)
    building = FreeBuilding(zones, surfaces, constructions, builds, args.dt, args.cells,
                            args.interior, args.exterior, args.h_natural, args.sky)
    forcing = build_forcing(building, data, weather_ir(args.epw, times))
    warmup = []
    for day in range(1, 101):
        before = building.state()
        run_intervals(building, forcing[:96], forcing[95], record=False)
        change = float(abs(building.state()-before).max())
        warmup.append(dict(day=day, max_state_change_K=change))
        print(f'Warm-up day {day}: {change:.6f} K', flush=True)
        if day >= 6 and change < args.warmup_tolerance:
            break
    else:
        raise RuntimeError('Warm-up did not converge')
    predictions, room_predictions = [], []
    previous = forcing[95]
    for day in range(args.days):
        start = day*96
        surface_prediction, room_prediction = run_intervals(building, forcing[start:start+96], previous)
        predictions.append(surface_prediction)
        room_predictions.append(room_prediction)
        previous = forcing[start+95]
        print(f'Run day {day+1}/{args.days}', flush=True)
    prediction = np.concatenate(predictions)
    room_prediction = np.concatenate(room_predictions)
    n = len(prediction)
    records = []
    for i, entry in enumerate(building.walls):
        front = face_data(data[entry['surface'].SurfaceName], 'Inside')
        back = (face_data(data[surfaces.loc[entry['pair']].SurfaceName], 'Inside') if entry['pair'] > 0 else
                face_data(data[entry['surface'].SurfaceName], 'Outside', entry['wall'].absorptivity, entry['pair'] == -1))
        for side_index, (side, ep) in enumerate([('inside', front), ('outside', back)]):
            frame = pd.DataFrame(dict(time=times.timestamp.iloc[:n].to_numpy(), surface_id=entry['sid'],
                side=side, category=entry['category'], area=entry['wall'].Af))
            for col, (q, ep_col) in enumerate([('Ts', 0), ('cond', 4), ('conv', 5), ('rad', 3)]):
                frame[q+'_model'] = prediction[:, i, side_index, col]
                frame[q+'_ep'] = ep[:n, ep_col]
            records.append(frame)
    results = pd.concat(records, ignore_index=True)
    rooms = pd.concat([pd.DataFrame(dict(time=times.timestamp.iloc[:n].to_numpy(), zone=zid,
        T_model=room_prediction[:, i], T_ep=ep_rooms[zone.ZoneName].iloc[:n].to_numpy()+273.15))
        for i, (zid, zone) in enumerate(zones.iterrows())], ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output/'surface_timeseries.csv.gz', index=False)
    rooms.to_csv(args.output/'room_timeseries.csv', index=False)
    pd.DataFrame(building.audit).to_csv(args.output/'geometry.csv', index=False)
    zones.to_csv(args.output/'zones.csv')
    pd.DataFrame(warmup).to_csv(args.output/'warmup.csv', index=False)
    start = times.timestamp.iloc[0]-pd.Timedelta(minutes=15)
    sensitivity = []
    for excluded in [0, 7, 14]:
        if excluded >= args.days:
            continue
        cutoff = start+pd.Timedelta(days=excluded)
        rm = score(rooms[rooms.time > cutoff], ['zone'], ['T'])
        sm = score(results[results.time > cutoff], ['category', 'side'], ['Ts', 'cond', 'conv', 'rad'])
        rm.to_csv(args.output/f'room_metrics_exclude_{excluded}d.csv', index=False)
        sm.to_csv(args.output/f'surface_metrics_exclude_{excluded}d.csv', index=False)
        sensitivity.extend((rm.assign(excluded_days=excluded)).to_dict('records'))
    pd.DataFrame(sensitivity).to_csv(args.output/'initialization_sensitivity.csv', index=False)
    metadata = dict(timestep_seconds=args.dt, target_cells=args.cells, days=args.days,
        warmup_days=len(warmup), warmup_tolerance_K=args.warmup_tolerance,
        max_energy_residual_W=building.max_energy_residual,
        ground_temperature_C=18, interior_emissivity=.9, exterior_emissivity=.9,
        convection_models=building.models,
        face_convection={str(e["sid"]):e["wall"].convection_metadata() for e in building.walls},
        roughness='Imported exterior material roughness; this case is Smooth, multiplier 1.11',
        forcing='EP local outdoor temperature, local wind and incident solar; EPW horizontal infrared. Linear interpolation between zone-timestep endpoints. Local standard time.',
        wind_direction='EP zone-timestep endpoint, held within the interval',
        time_comparison='Endpoint samples; no indoor temperatures or net heat flows used as forcing.',
        warmup='Repeat first day to periodic state, at least 6 days. EP warm-up state unavailable; its 6..25 day settings and convergence criteria differ.',
        limitations=['Coefficients use previous GraphBEM surface state and current air; no EP thermal predictions prescribed',
                     'GraphBEM enclosure radiation omits wall-to-wall exchange and approximates split-wall view factors',
                     'Sky model selected independently; surrounding ground radiates at outdoor air temperature',
                     'Incident sunlight supplied by EP, so solar transposition is not independently tested',
                     'Constant GraphBEM air density and heat capacity retained',
                     'First midnight starts from repeated first-day final weather sample'],
        hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                [args.ep_case/'results/eplusout.sql', args.epw, Path(__file__), ROOT/'model/WallSimulation.py',
                 ROOT/'model/Radiation.py', ROOT/'model/RoomSimulation.py', ROOT/'model/Convection.py']})
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    plot(rooms, results, args.output)
    cutoff = start+pd.Timedelta(days=min(7, args.days-1))
    print(score(rooms[rooms.time > cutoff], ['zone'], ['T']).to_string(index=False), flush=True)


def plot(rooms, surfaces, output):
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, (zid, frame) in zip(axes, rooms.groupby('zone')):
        ax.plot(frame.time, frame.T_ep-273.15, label='EnergyPlus', lw=1)
        ax.plot(frame.time, frame.T_model-273.15, label='GraphBEM', lw=1)
        ax.set_ylabel(f'Zone {zid}\n°C')
    axes[0].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output/'room_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(15, 10), sharex=True)
    for col, (category, sid) in enumerate([('wall', 1), ('roof', 6), ('floor', 5), ('partition', 3)]):
        frame = surfaces[(surfaces.surface_id == sid) & (surfaces.side == 'inside')].copy()
        begin = frame.time.min().normalize()+pd.Timedelta(days=14)
        if begin > frame.time.max():
            begin = frame.time.min().normalize()
        frame = frame[(frame.time >= begin) & (frame.time < begin+pd.Timedelta(days=3))]
        for row, q in enumerate(['Ts', 'cond', 'conv', 'rad']):
            offset = 273.15 if q == 'Ts' else 0
            axes[row, col].plot(frame.time, frame[q+'_ep']-offset, label='EP')
            axes[row, col].plot(frame.time, frame[q+'_model']-offset, label='GraphBEM')
            axes[row, col].set_ylabel('°C' if q == 'Ts' else 'W/m²')
            axes[row, col].set_title(f'{category} inside {q}')
            axes[row, col].tick_params(axis='x', labelrotation=30)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output/'surface_comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
