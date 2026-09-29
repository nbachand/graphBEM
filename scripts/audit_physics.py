"""Reproduce physics-audit counterexamples without changing model behavior.

Run from the repository root in the graphBEM environment:
    python scripts/audit_physics.py

Writes current measurements to analysis/physics_fixes/audit_after.json.
The historical, pre-fix measurements remain in analysis/physics_audit.json. This is an
investigation of existing behavior, not a regression suite that approves it.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from model.Radiation import Radiation
from model.BuildingSimulation import BuildingSimulation
from model.VentilationSimulation import VentilationSimulation
from model import WallSimulation as ws
from model.utils import WallSides
from scripts.compare_wall_solver import weather_sample, run_graph


def sky_equilibrium(alpha, sky_fraction):
    sigma_t4 = 5.67e-8 * 300**4
    wall = SimpleNamespace(X=4, Y=4, absorptivity=alpha, T_prof=np.array([300., 300.]))
    adjacent = {'inside': dict(wall=wall, weight=1, nodes=WallSides('inside', 'outside'))}
    rad = Radiation(solveType='sky')
    # Sky and ground have the same radiative temperature in this equilibrium test.
    forcing = sky_fraction*sigma_t4 + (1-sky_fraction)*sigma_t4
    rad.initialize(adjacent, solarGain=0, longwaveGain=forcing)
    with np.errstate(invalid='ignore', divide='ignore'):
        actual = float(rad.timeStep()['inside'])
    return dict(expected_W_m2=0, actual_W_m2=actual if np.isfinite(actual) else None,
                is_finite=bool(np.isfinite(actual)))


def main():
    report = {}
    report['isothermal_exterior_radiation'] = {
        'roof_alpha_075': sky_equilibrium(.75, 1),
        'wall_alpha_070': sky_equilibrium(.7, .5),
        'alpha_one': sky_equilibrium(1, 1)}
    report['radiation_step_response'] = []
    for dt in [15, 30, 60]:
        sim = BuildingSimulation(delt=dt, simLength=900, Tbound={}, windSpeed={}, radG={})
        applied = 0.0
        for _ in range(int(900/dt)):
            applied = (1-sim.radDamping)*100 + sim.radDamping*applied
        report['radiation_step_response'].append(dict(dt_seconds=dt, after_900s_W_m2=applied,
            effective_time_constant_seconds=-dt/np.log(sim.radDamping) if sim.radDamping else 0))
    report['pivot_window_scale_test'] = []
    for scale in [1, 2, 4]:
        vent = VentilationSimulation(H=scale, W=scale, ventType=None, alphas=[], As=[], Ls=[])
        report['pivot_window_scale_test'].append(dict(H=scale, W=scale, angle_degrees=45,
                                                     Cd=float(vent.get_Cd(np.pi/4))))
    vent = VentilationSimulation(H=1, W=1, ventType='HWP4', alphas=[45], As=[1], Ls=[1])
    try:
        report['ventilation_scalar_call'] = float(vent.timeStep(3600, Tint=300., Tout=290.))
    except Exception as error:
        report['ventilation_scalar_call'] = type(error).__name__ + ': ' + str(error)

    sample = weather_sample()
    captured = []
    original = BuildingSimulation.run
    def capture(sim):
        original(sim)
        captured.append(sim)
    with patch.object(BuildingSimulation, 'run', capture):
        run_graph(sample, ws.WallSimulation)
    sim = captured[0]
    report['geometry'] = []
    report['room_radiation_balance'] = []
    report['radiation_order_test'] = []
    report['view_factor_sums'] = {}
    for room in ['CR', 'SS', 'DR', 'CV']:
        room_data = sim.bG.G.nodes[room]
        floor = sim.bG.G.edges[room, 'FL']
        area = floor['wall'].Af * floor['weight']
        volume = room_data['room'].V
        report['geometry'].append(dict(room=room, floor_area_m2=area, air_volume_m3=volume,
                                       implied_height_m=volume/area, radiation_height_m=room_data['rad'].storyHeight))
        power = np.zeros(sim.N)
        for other, edge in sim.bG.G[room].items():
            for side in ['front', 'back']:
                if getattr(edge['nodes'], side) == room:
                    power += edge['wall'].Af * edge['weight'] * getattr(edge['radEApplied'], side)
        report['room_radiation_balance'].append(dict(room=room,
            rms_net_internal_radiation_W=float(np.sqrt(np.mean(power[1:]**2))),
            max_abs_net_internal_radiation_W=float(np.max(np.abs(power[1:])))))
        reference = Radiation(solveType='room')
        reference.initialize(dict(sim.bG.G[room]))
        reverse = Radiation(solveType='room')
        try:
            reverse.initialize(dict(reversed(list(sim.bG.G[room].items()))))
            delta = reference.timeStep() - reverse.timeStep()
            report['radiation_order_test'].append(dict(room=room,
                max_abs_flux_change_W_m2=float(delta.abs().max())))
        except Exception as error:
            report['radiation_order_test'].append(dict(room=room,error=type(error).__name__+': '+str(error)))
        report['view_factor_sums'][room] = {
            surface: sum(1/e['radianceResistance'] for e in reference.G[surface].values())/data['A']
            for surface, data in reference.G.nodes(data=True)}
    loop = sim.bG.G.edges['DR', 'DR']
    report['self_loop_partition'] = dict(
        area_one_face_m2=loop['wall'].Af*loop['weight'],
        area_used_by_radiation_m2=sim.bG.G.nodes['DR']['rad'].G.nodes['DR']['A'],
        front_max_abs_applied_W_m2=float(np.max(np.abs(loop['radEApplied'].front))),
        back_max_abs_applied_W_m2=float(np.max(np.abs(loop['radEApplied'].back))))
    report['historical_sky_longwave_absorption_correction'] = {}
    for prefix, alpha in [('Horizontal', .75), ('Vertical', .7)]:
        deficit = (.9-alpha)*sample[prefix+' Sky Longwave Radiation']
        report['historical_sky_longwave_absorption_correction'][prefix] = dict(
            mean_W_m2=float(deficit.mean()), max_W_m2=float(deficit.max()))
    report['notes'] = ['No production model code changed by this audit.',
        'Burbank August first 5000 weather samples at 30 s, same settings as comparison.',
        'Room radiation power sums actual applied face fluxes with areas and edge weights.',
        'Missing physical processes are outside this audit; view-factor sums below one alone are not flagged.']
    output = ROOT/'analysis/physics_fixes/audit_after.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
