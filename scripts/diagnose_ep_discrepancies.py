"""Attribute matched-building differences without changing production physics.

--audit compares constitutive laws at identical EP temperatures/weather.
--variant runs a free building with selected EP correlation choices. No EP
thermal prediction is prescribed in these counterfactual building runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.compare_ep_replay import read_case, face_data
from scripts.compare_ep_free_running import (FreeBuilding, read_zones, weather_ir,
                                             build_forcing, run_intervals, score, SIGMA)
from model.WallSimulation import convectionDOE2

OUT = ROOT/'analysis/energyplus_diagnosis'
BASE = ROOT/'analysis/energyplus_free_running/base'
EP_SIGMA = 5.6697e-8


def natural(ts, ta, cosine):
    """EnergyPlus CalcASHRAETARPNatural; cosine points out of the face."""
    delta = np.asarray(ts)-np.asarray(ta)
    magnitude = abs(delta)**(1/3)
    if abs(cosine) < 1e-10:
        return 1.31*magnitude
    # Source uses 7.238; the 22.2 Engineering Reference prints 7.283.
    return np.where(delta*cosine > 0, 9.482/(7.238-abs(cosine)),
                    1.810/(1.382+abs(cosine)))*magnitude


def forced(wind, direction, azimuth, cosine):
    angle = abs((np.asarray(direction)-azimuth+180) % 360-180)
    windward = (abs(cosine) >= .98) | (angle <= 90.001)
    return np.where(windward, 3.26*np.asarray(wind)**.89, 3.55*np.asarray(wind)**.617)


def exterior_h(ts, ta, wind, direction, surface, roughness):
    cosine = np.cos(np.radians(surface.Tilt))
    hn = natural(ts, ta, cosine)
    hf = forced(wind, direction, surface.Azimuth, cosine)
    return hn+roughness*(np.sqrt(hn**2+hf**2)-hn)


def context():
    meta = json.loads((BASE/'metadata.json').read_text())
    paths = [Path(p) for p in meta['hashes']]
    sql = next(p for p in paths if p.suffix == '.sql')
    epw = next(p for p in paths if p.suffix == '.epw')
    surfaces, constructions, builds, times, data = read_case(sql.parent.parent)
    zones, ep_rooms = read_zones(sql.parent.parent, times)
    ir = weather_ir(epw, times)
    return zones, surfaces, constructions, builds, times, data, ep_rooms, ir


def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def audit(ctx):
    zones, surfaces, constructions, builds, times, data, ep_rooms, ir = ctx
    building = FreeBuilding(zones, surfaces, constructions, builds)
    # This audit deliberately supplies EP surface states to evaluate formulas,
    # not to simulate the building. Actual free-running errors remain separate.
    keep = np.arange(len(times)) >= 7*96
    rows, inside, decomposition = [], [], []
    free = pd.read_csv(ROOT/'analysis/energyplus_free_running/refined/surface_timeseries.csv.gz')
    for i, entry in enumerate(building.walls):
        s = entry['surface']; sid=entry['sid']; pair=entry['pair']
        f = data[s.SurfaceName]
        if pair != 0:
            continue
        ts, ta, hep, radep, _, convep = face_data(f, 'Outside').T
        wind = f['Surface Outside Face Outdoor Air Wind Speed'].to_numpy()
        direction = f['Surface Outside Face Outdoor Air Wind Direction'].to_numpy()
        previous_ts = np.r_[ts[0], ts[:-1]]
        roughness = entry['wall'].roughness.back
        cosine = np.cos(np.radians(s.Tilt))
        hn = natural(ts, ta, cosine)
        hf = forced(wind, direction, s.Azimuth, cosine)
        h_graph = convectionDOE2(2., wind, roughness)
        h_updated_wind = 2.+roughness*(np.sqrt(4+hf**2)-2.)
        h_updated_natural = (1-roughness)*hn+roughness*np.sqrt(hn**2+(2.62*wind**.7535)**2)
        h_ep_law = exterior_h(ts, ta, wind, direction, s, roughness)
        h_ep_lag = exterior_h(previous_ts, ta, wind, direction, s, roughness)
        for name, h in [('graph', h_graph), ('updated_wind_only',h_updated_wind),
                        ('updated_natural_only',h_updated_natural), ('ep_law_current_Ts',h_ep_law),
                        ('ep_law_previous_Ts',h_ep_lag)]:
            rows.append(pd.DataFrame(dict(surface_id=sid, category=entry['category'],area=s.Area,
                law=name, h_error=(h-hep)[keep], flux_error=(h*(ta-ts)-convep)[keep],
                h_model=h[keep], h_ep=hep[keep])))
        sky_view = (1+cosine)/2
        sky_temp = (ir/EP_SIGMA)**.25
        sw = .7*f['Surface Outside Face Incident Solar Radiation Rate per Area'].to_numpy()
        lw_ep = radep-sw
        lw_graph = .9*(sky_view*ir+(1-sky_view)*SIGMA*ta**4-SIGMA*ts**4)
        beta = np.sqrt(sky_view)
        lw_ep_nonlinear = .9*EP_SIGMA*(sky_view*beta*sky_temp**4+(1-sky_view*beta)*ta**4-ts**4)
        # EP initializes exterior radiative coefficients before solving the
        # current outside temperature, then reports h_r*(Tenv-Tcurrent).
        hr_sky = .9*EP_SIGMA*sky_view*beta*(previous_ts**2+sky_temp**2)*(previous_ts+sky_temp)
        hr_air_ground = .9*EP_SIGMA*(1-sky_view*beta)*(previous_ts**2+ta**2)*(previous_ts+ta)
        lw_ep_lag = hr_sky*(sky_temp-ts)+hr_air_ground*(ta-ts)
        actual = free[(free.surface_id==sid)&(free.side=='outside')]
        actual_error = actual.rad_model.to_numpy()-actual.rad_ep.to_numpy()
        components = {
            'free_running_total':actual_error,
            'graph_law_at_ep_Ts':lw_graph-lw_ep,
            'sky_split_difference':lw_graph-lw_ep_nonlinear,
            'ep_coefficient_lag_difference':lw_ep_nonlinear-lw_ep,
            'ep_lagged_formula_reconstruction':lw_ep_lag-lw_ep,
            'model_temperature_and_30s_lag':actual_error-(lw_graph-lw_ep),
        }
        for name, values in components.items():
            decomposition.append(pd.DataFrame(dict(surface_id=sid, category=entry['category'],area=s.Area,
                component=name, error=values[keep])))
    for sid, s in surfaces.iterrows():
        f = data[s.SurfaceName]
        ts, ta, hep, radep, _, convep = face_data(f, 'Inside').T
        category = 'partition' if s.ExtBoundCond>0 else s.ClassName.lower()
        h = np.maximum(.1, natural(ts, ta, -np.cos(np.radians(s.Tilt))))
        for name, prediction in [('graph',np.full(len(ts),2.)),('ep_tarp',h)]:
            inside.append(pd.DataFrame(dict(surface_id=sid,category=category,area=s.Area,
                law=name,h_error=(prediction-hep)[keep],flux_error=(prediction*(ta-ts)-convep)[keep],
                h_model=prediction[keep],h_ep=hep[keep])))
    def aggregate(frames, grouping, errors, averages=()):
        frame=pd.concat(frames)
        results=[]
        for keys, group in frame.groupby(grouping):
            record=dict(zip(grouping,keys))
            for col in errors:
                record[col+'_rmse']=float(np.sqrt(np.average(group[col]**2,weights=group.area)))
                record[col+'_bias']=float(np.average(group[col],weights=group.area))
            for col in averages:
                record[col+'_mean']=float(np.average(group[col],weights=group.area))
            results.append(record)
        return pd.DataFrame(results)
    aggregate(rows,['category','law'],['h_error','flux_error'],['h_model','h_ep']).to_csv(OUT/'exterior_same_state.csv',index=False)
    aggregate(inside,['category','law'],['h_error','flux_error'],['h_model','h_ep']).to_csv(OUT/'interior_same_state.csv',index=False)
    aggregate(decomposition,['category','component'],['error']).to_csv(OUT/'radiation_decomposition.csv',index=False)
    rad_rows=[]
    for operator, refs in building.enclosures:
        temperatures=[]; targets=[]
        for i, side in refs:
            entry=building.walls[i]
            sid=entry['sid'] if side==0 else entry['pair']
            f=data[surfaces.loc[sid].SurfaceName]
            temperatures.append(f['Surface Inside Face Temperature'].to_numpy()+273.15)
            targets.append(f['Surface Inside Face Net Surface Thermal Radiation Heat Gain Rate per Area'].to_numpy())
        prediction=operator @ np.array(temperatures)**4
        for k,(i,side) in enumerate(refs):
            entry=building.walls[i]
            rad_rows.append(pd.DataFrame(dict(category=entry['category'],law='graph_enclosure_at_ep_Ts',
                area=entry['wall'].Af,error=(prediction[k]-targets[k])[keep])))
    aggregate(rad_rows,['category','law'],['error']).to_csv(OUT/'enclosure_same_state.csv',index=False)
    print(pd.read_csv(OUT/'exterior_same_state.csv').to_string(index=False))
    print(pd.read_csv(OUT/'radiation_decomposition.csv').to_string(index=False))


class CounterfactualBuilding(FreeBuilding):
    def __init__(self, *args, variant, directions, **kwargs):
        super().__init__(*args, **kwargs)
        self.variant=variant
        self.directions=directions
        self.interval=0
        self.substep=0
        self.roughness=[e['wall'].roughness.back for e in self.walls]

    def step(self, forcing):
        f=forcing.copy()
        use_exterior=self.variant in ['exterior_convection','exterior_and_sky','all_convection_and_sky']
        use_sky=self.variant in ['sky','exterior_and_sky','all_convection_and_sky']
        use_interior=self.variant=='all_convection_and_sky'
        for i, entry in enumerate(self.walls):
            wall=entry['wall']; s=entry['surface']
            cosine=np.cos(np.radians(s.Tilt))
            if use_interior:
                wall.h.front=float(max(.1,natural(wall.T_prof[0],self.rooms[entry['front']].Tint,-cosine)))
                if entry['pair']>0:
                    wall.h.back=float(max(.1,natural(wall.T_prof[-1],self.rooms[entry['back']].Tint,cosine)))
            if entry['pair']!=0:
                continue
            ta, wind, sw, ir=f[i]
            if use_exterior:
                # Wind direction is held at the EP 15-minute endpoint, avoiding
                # interpolation across north or between windward/leeward classes.
                direction=self.directions[self.interval,i]
                wall.h.back=float(exterior_h(wall.T_prof[-1],ta,wind,direction,s,self.roughness[i]))
                wall.roughness.back=0.
            if use_sky:
                beta=np.sqrt((1+cosine)/2)
                f[i,3]=beta*ir+(1-beta)*SIGMA*ta**4
        return super().step(f)


def variant_run(ctx, variant):
    zones,surfaces,constructions,builds,times,data,ep_rooms,ir=ctx
    building=CounterfactualBuilding(zones,surfaces,constructions,builds,
        dt=60,cells=18,variant=variant,directions=None)
    directions=np.zeros((len(times),len(building.walls)))
    for i,e in enumerate(building.walls):
        if e['pair']==0:
            directions[:,i]=data[e['surface'].SurfaceName]['Surface Outside Face Outdoor Air Wind Direction']
    building.directions=directions
    forcing=build_forcing(building,data,ir)
    def day_run(day,previous,record):
        outputs=[]; room_outputs=[]
        for k in range(day*96,(day+1)*96):
            building.interval=k
            p,r=run_intervals(building,forcing[k:k+1],previous,record=record)
            if record:
                outputs.append(p[0]); room_outputs.append(r[0])
            previous=forcing[k]
        return outputs,room_outputs
    for day in range(1,101):
        state=building.state()
        day_run(0,forcing[95],False)
        change=float(abs(building.state()-state).max())
        if day%5==0:
            print(variant,'warmup',day,change,flush=True)
        if day>=6 and change<.001:
            break
    else:
        raise RuntimeError('Warm-up failed')
    warmup_days=day
    predictions=[]; room_predictions=[]
    for day in range(31):
        p,r=day_run(day,forcing[95] if day==0 else forcing[day*96-1],True)
        predictions.extend(p);room_predictions.extend(r)
        if day%5==0:
            print(variant,'simulation day',day+1,flush=True)
    predictions=np.array(predictions);room_predictions=np.array(room_predictions)
    rooms=pd.concat([pd.DataFrame(dict(time=times.timestamp.to_numpy(),zone=zid,
        T_model=room_predictions[:,j],T_ep=ep_rooms[z.ZoneName].to_numpy()+273.15))
        for j,(zid,z) in enumerate(zones.iterrows())])
    records=[]
    for i,e in enumerate(building.walls):
        front=face_data(data[e['surface'].SurfaceName],'Inside')
        back=(face_data(data[surfaces.loc[e['pair']].SurfaceName],'Inside') if e['pair']>0 else
              face_data(data[e['surface'].SurfaceName],'Outside',e['wall'].absorptivity,e['pair']==-1))
        for side,(label,target) in enumerate([('inside',front),('outside',back)]):
            frame=pd.DataFrame(dict(time=times.timestamp.to_numpy(),surface_id=e['sid'],side=label,
                                    category=e['category'],area=e['wall'].Af))
            for col,(q,target_col) in enumerate([('Ts',0),('cond',4),('conv',5),('rad',3)]):
                frame[q+'_model']=predictions[:,i,side,col]
                frame[q+'_ep']=target[:,target_col]
            records.append(frame)
    results=pd.concat(records)
    path=OUT/variant;path.mkdir(exist_ok=True)
    rooms.to_csv(path/'room_timeseries.csv',index=False)
    results.to_csv(path/'surface_timeseries.csv.gz',index=False)
    for excluded in [0,7,14]:
        cutoff=times.timestamp.iloc[0].normalize()+pd.Timedelta(days=excluded)
        score(rooms[rooms.time>cutoff],['zone'],['T']).to_csv(path/f'room_metrics_exclude_{excluded}d.csv',index=False)
        score(results[results.time>cutoff],['category','side'],['Ts','cond','conv','rad']).to_csv(path/f'surface_metrics_exclude_{excluded}d.csv',index=False)
    metadata=dict(variant=variant,dt=60,cells=18,warmup_days=warmup_days,
        warmup_change_K=change,max_energy_residual_W=building.max_energy_residual,
        independent=True,exterior_h_evaluation='Previous 60s surface state, current weather',
        wind_direction='Current EP 15-minute endpoint; held within interval',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [Path(__file__),ROOT/'scripts/compare_ep_free_running.py',ROOT/'model/WallSimulation.py',ROOT/'model/Radiation.py']})
    (path/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(variant,pd.read_csv(path/'room_metrics_exclude_7d.csv').to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--variant',choices=['exterior_convection','sky','exterior_and_sky','all_convection_and_sky'])
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    ctx=context()
    if args.audit:
        audit(ctx)
    if args.variant:
        variant_run(ctx,args.variant)


if __name__=='__main__':
    main()
