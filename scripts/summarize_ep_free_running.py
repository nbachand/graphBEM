"""Summarize the saved free-running benchmark and its numerical refinement."""
from pathlib import Path
import argparse
import json
import sqlite3
import numpy as np
import pandas as pd


def forcing_audit(root, metadata):
    paths = [Path(p) for p in metadata['hashes']]
    sql = next(p for p in paths if p.suffix == '.sql')
    epw = next(p for p in paths if p.suffix == '.epw')
    with sqlite3.connect(sql.resolve().as_uri()+'?mode=ro', uri=True) as db:
        air = pd.read_sql("""SELECT t.Year,t.Month,t.Day,t.Hour,t.Minute,r.Value FROM ReportData r
            JOIN ReportDataDictionary d USING(ReportDataDictionaryIndex) JOIN Time t USING(TimeIndex)
            WHERE d.Name='Site Outdoor Air Drybulb Temperature' AND t.WarmupFlag=0
            AND t.IntervalType=-1 ORDER BY t.TimeIndex""", db)
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'Nominal%'")]
        counts = {name:db.execute('SELECT count(*) FROM '+name).fetchone()[0] for name in tables}
        ground_range = db.execute("""SELECT min(r.Value),max(r.Value) FROM ReportData r
            JOIN ReportDataDictionary d USING(ReportDataDictionaryIndex)
            JOIN Surfaces s ON s.SurfaceName=d.KeyValue
            WHERE d.Name='Surface Outside Face Temperature' AND s.ExtBoundCond=-1""").fetchone()
    stamp = pd.to_datetime(dict(year=air.Year, month=air.Month, day=air.Day))+pd.to_timedelta(air.Hour*60+air.Minute, unit='min')
    raw = pd.read_csv(epw, skiprows=8, header=None)
    weather_stamp = pd.to_datetime(dict(year=np.full(len(raw), int(air.Year.iloc[0])), month=raw[1], day=raw[2]))+pd.to_timedelta(raw[3], unit='h')
    prediction = np.interp(stamp.astype('int64'), weather_stamp.astype('int64'), raw[6])
    start = stamp.iloc[0].normalize()
    last = raw.loc[weather_stamp == start+pd.Timedelta(days=1), 6].iloc[0]
    first = raw.loc[weather_stamp == start+pd.Timedelta(hours=1), 6].iloc[0]
    fraction = (stamp-start).dt.total_seconds().to_numpy()/3600
    prediction[fraction <= 1] = last+(first-last)*fraction[fraction <= 1]
    error = float(abs(prediction-air.Value).max())
    if error > 1e-6 or any(counts.values()) or not np.allclose(ground_range, [18., 18.]):
        raise ValueError('Source weather, gains or ground assumptions do not match the case')
    report = dict(site_air_epw_interpolation_max_error_K=error,
                  nominal_gain_and_airflow_table_counts=counts, ground_surface_range_C=ground_range)
    (root/'forcing_audit.json').write_text(json.dumps(report, indent=2)+'\n')


def exterior_plot(root):
    from matplotlib import pyplot as plt
    frame = pd.read_csv(root/'refined/surface_timeseries.csv.gz', parse_dates=['time'])
    begin = frame.time.min().normalize()+pd.Timedelta(days=14)
    frame = frame[(frame.time >= begin) & (frame.time < begin+pd.Timedelta(days=3))]
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
    for col, (category, sid) in enumerate([('Wall', 1), ('Roof', 6)]):
        surface = frame[(frame.surface_id == sid) & (frame.side == 'outside')]
        for row, (q, label) in enumerate([('Ts', 'Surface temperature'), ('cond', 'Conduction'),
                                         ('conv', 'Convection'), ('rad', 'Net radiation')]):
            offset = 273.15 if q == 'Ts' else 0
            ax = axes[row, col]
            ax.plot(surface.time, surface[q+'_ep']-offset, label='EnergyPlus')
            ax.plot(surface.time, surface[q+'_model']-offset, label='GraphBEM')
            ax.set_title(f'{category} exterior: {label}')
            ax.set_ylabel('°C' if q == 'Ts' else 'W/m²')
            ax.tick_params(axis='x', labelrotation=30)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(root/'exterior_comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('analysis/energyplus_free_running'))
    args = parser.parse_args()
    rows = []
    for kind, filename, keys, quantities in [
        ('room', 'room_timeseries.csv', ['time', 'zone'], ['T']),
        ('surface', 'surface_timeseries.csv.gz', ['time', 'surface_id', 'side'], ['Ts', 'cond', 'conv', 'rad'])]:
        base, fine = [pd.read_csv(args.root/run/filename, parse_dates=['time']) for run in ['base', 'refined']]
        merged = base.merge(fine, on=keys, suffixes=('_base', '_fine'), validate='one_to_one')
        if len(merged) != len(base) or len(merged) != len(fine):
            raise ValueError('Refinement histories differ in coverage')
        cutoff = merged.time.min().normalize()+pd.Timedelta(days=7)
        merged = merged[merged.time > cutoff]
        grouping = ['zone'] if kind == 'room' else ['category_base', 'side']
        for values, group in merged.groupby(grouping):
            values = values if isinstance(values, tuple) else (values,)
            for q in quantities:
                if kind == 'surface' and values == ('floor', 'outside') and q in ['Ts', 'conv', 'rad']:
                    continue
                error = group[q+'_model_fine']-group[q+'_model_base']
                weight = group.area_base if kind == 'surface' else np.ones(len(group))
                rows.append(dict(kind=kind, group='/'.join(map(str, values)), quantity=q,
                    refinement_rmse=float(np.sqrt(np.average(error**2, weights=weight))),
                    refinement_max_abs=float(abs(error).max())))
    pd.DataFrame(rows).to_csv(args.root/'refinement.csv', index=False)
    summary = []
    for run in ['base', 'refined']:
        for kind in ['room', 'surface']:
            frame = pd.read_csv(args.root/run/f'{kind}_metrics_exclude_7d.csv')
            summary.append(frame.assign(run=run, kind=kind))
    pd.concat(summary, ignore_index=True).to_csv(args.root/'summary.csv', index=False)
    exterior_plot(args.root)
    metadata = json.loads((args.root/'refined/metadata.json').read_text())
    forcing_audit(args.root, metadata)
    sensitivity = []
    for run, roughness in [('legacy_roughness_refined', 1.64), ('refined', 1.11)]:
        path = args.root/run/'room_metrics_exclude_7d.csv'
        if path.exists():
            sensitivity.append(pd.read_csv(path).assign(roughness_multiplier=roughness))
    pd.concat(sensitivity).to_csv(args.root/'roughness_sensitivity.csv', index=False)
    print('Refined run energy residual, W:', metadata['max_energy_residual_W'])
    print(pd.read_csv(args.root/'refined/room_metrics_exclude_7d.csv').to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
