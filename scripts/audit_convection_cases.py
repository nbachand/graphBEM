"""Audit geometry, ground and gains in the three saved EnergyPlus cases."""
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
from scripts.compare_ep_replay import read_case
from scripts.compare_ep_free_running import read_zones


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, required=True)
    root = parser.parse_args().case_root
    records, reference, wet_events = [], None, []
    for climate, key in [('burbank', 'BURBANK'), ('palm_springs', 'PALM-SPRINGS'), ('arcata', 'ARCATA')]:
        case = next(root.glob('*'+key+'*'))
        surfaces, constructions, builds, times, data = read_case(case)
        zones, _ = read_zones(case, times)
        if reference is None:
            reference = surfaces, constructions, builds, zones
        else:
            for left, right in [(surfaces, reference[0]), (constructions, reference[1]), (zones, reference[3])]:
                pd.testing.assert_frame_equal(left, right)
            for key in builds:
                pd.testing.assert_frame_equal(builds[key], reference[2][key])
        ground = np.concatenate([data[s.SurfaceName]['Surface Outside Face Temperature'].to_numpy()
                                 for _, s in surfaces[surfaces.ExtBoundCond == -1].iterrows()])
        np.testing.assert_allclose(ground, 18.)
        outdoor = np.concatenate([data[s.SurfaceName]['Surface Outside Face Outdoor Air Drybulb Temperature'].to_numpy()
                                  for _, s in surfaces[surfaces.ExtBoundCond == 0].iterrows()])
        for sid, surface in surfaces[surfaces.ExtBoundCond == 0].iterrows():
            frame = data[surface.SurfaceName]
            h = frame['Surface Outside Face Convection Heat Transfer Coefficient']
            for index in h.index[h >= 999.]:
                wet_events.append(dict(climate=climate, surface_id=sid,
                    time=times.loc[index, 'timestamp'], ep_h_W_m2K=h.loc[index],
                    drybulb_C=frame.loc[index, 'Surface Outside Face Outdoor Air Drybulb Temperature'],
                    wetbulb_C=frame.loc[index, 'Surface Outside Face Outdoor Air Wetbulb Temperature']))
        sql = case/'results/eplusout.sql'
        with sqlite3.connect(sql.resolve().as_uri()+'?mode=ro', uri=True) as db:
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", db).name
            nominal = {name: int(db.execute('SELECT count(*) FROM '+name).fetchone()[0])
                       for name in tables if name.startswith('Nominal')}
            simulation = pd.read_sql('SELECT * FROM Simulations', db).to_dict('records')
        records.append(dict(case=str(case), identical_geometry_constructions_zones=True,
            quarter_hours=len(times), ground_C=[float(ground.min()), float(ground.max())],
            outdoor_C=[float(outdoor.min()), float(outdoor.max())], nominal_table_counts=nominal,
            simulation=simulation, sql_sha256=hashlib.sha256(sql.read_bytes()).hexdigest()))
    output = ROOT/'analysis/convection_models'
    output.mkdir(parents=True, exist_ok=True)
    (output/'case_audit.json').write_text(json.dumps(records, indent=2)+'\n')
    pd.DataFrame(wet_events, columns=['climate', 'surface_id', 'time', 'ep_h_W_m2K',
                                    'drybulb_C', 'wetbulb_C']).to_csv(output/'wet_boundary_events.csv', index=False)
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
