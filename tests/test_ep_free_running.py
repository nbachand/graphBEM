import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from model.Radiation import Radiation
from model.utils import WallSides
from scripts.compare_ep_free_running import FreeBuilding, radiation_operator, build_forcing, weather_ir


def case():
    zones = pd.DataFrame([dict(ZoneIndex=1, Multiplier=1, ListMultiplier=1,
        ExtWindowArea=0, Volume=48.8, CeilingHeight=3.05,
        MinimumX=0, MaximumX=4, MinimumY=0, MaximumY=4)]).set_index('ZoneIndex')
    surfaces = pd.DataFrame([dict(SurfaceIndex=i+1, SurfaceName=str(i+1), ZoneIndex=1,
        ExtBoundCond=-1 if i == 5 else 0, ClassName='Wall' if i < 4 else 'Roof' if i == 4 else 'Floor',
        Area=12.2 if i < 4 else 16., ConstructionIndex=1, Azimuth=i*90 if i < 4 else 0,
        Tilt=90 if i < 4 else 0 if i == 4 else 180) for i in range(6)]).set_index('SurfaceIndex')
    materials = pd.DataFrame([dict(key='Material', Thickness=.1, Conductivity=1.,
        Density=1000., Specific_Heat=1000., Thermal_Resistance=.1, Roughness=4)])
    constructions = pd.DataFrame([dict(ConstructionIndex=1, Uvalue=10.,
        OutsideAbsorpSolar=.7, InsideAbsorpThermal=.9, OutsideAbsorpThermal=.9)]).set_index('ConstructionIndex')
    return zones, surfaces, constructions, {1: materials}


class FreeRunningTests(unittest.TestCase):
    def test_epw_hour_endpoints_and_first_day_midnight(self):
        raw = np.zeros((48, 13))
        raw[:, 0] = 2008  # Weather source year differs from the run-period year.
        raw[:, 1] = 8
        raw[:, 2] = np.repeat([1, 2], 24)
        raw[:, 3] = np.tile(np.arange(1, 25), 2)
        raw[:, 12] = 300+np.arange(48)
        times = pd.DataFrame(dict(Year=2017, timestamp=pd.date_range('2017-08-01 00:15', periods=100, freq='15min')))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'weather.epw'
            path.write_text('header\n'*8)
            with path.open('a') as stream:
                pd.DataFrame(raw).to_csv(stream, header=False, index=False)
            ir = weather_ir(path, times)
        np.testing.assert_allclose(ir[:4], [317.25, 311.5, 305.75, 300])
        np.testing.assert_allclose(ir[95:100], [323, 323.25, 323.5, 323.75, 324])

    def test_operator_matches_production_with_distinct_spectral_properties(self):
        mapping = {}
        for name, y, temp in [('wall', 3.05, 302.), ('RF', 4., 320.), ('FL', 4., 288.)]:
            mapping[name] = dict(wall=SimpleNamespace(X=4., Y=y, absorptivity=.7,
                T_prof=np.array([temp, temp])), weight=1., nodes=WallSides('room', name))
        rad = Radiation(solveType='room', storyHeight=3.05, emissivity=.9)
        rad.initialize(mapping)
        temperatures = np.array([mapping[n]['wall'].T_prof[0] for n in rad.G])
        np.testing.assert_allclose(radiation_operator(rad) @ temperatures**4,
                                   rad.timeStep().to_numpy(), atol=1e-11)
        self.assertAlmostEqual(rad.G.nodes['RF']['boundaryResistance'], .1/(.9*16))
        self.assertEqual(mapping['RF']['wall'].absorptivity, .7)

    def test_uniform_equilibrium_and_coupled_energy_conservation(self):
        building = FreeBuilding(*case(), dt=30, cells=4)
        self.assertEqual(building.walls[0]['wall'].roughness.back, 1.11)
        forcing = np.tile([291.15, 0., 0., 5.67e-8*291.15**4], (6, 1))
        for room in building.rooms.values():
            room.Tint = 291.15
        for entry in building.walls:
            entry['wall'].initialize(30, 291.15, 291.15)
        original = building.state()
        for _ in range(10):
            building.step(forcing)
        np.testing.assert_allclose(building.state(), original, atol=1e-10)
        forcing[:, 0] += 10
        forcing[:, 2] = 300
        for _ in range(100):
            building.step(forcing)
        self.assertLess(building.max_energy_residual, 1e-6)
        self.assertGreater(building.rooms[1].Tint, 291.15)

    def test_forcing_does_not_require_ep_thermal_predictions(self):
        building = FreeBuilding(*case())
        weather = pd.DataFrame({'Surface Outside Face Outdoor Air Drybulb Temperature':[20., 21.],
            'Surface Outside Face Outdoor Air Wind Speed':[1., 2.],
            'Surface Outside Face Incident Solar Radiation Rate per Area':[100., 200.]})
        data = {str(i):weather.copy() for i in range(1, 7)}
        forcing = build_forcing(building, data, np.array([300., 310.]))
        for frame in data.values():
            frame['Surface Inside Face Temperature'] = [-1000., 1000.]
            frame['Surface Outside Face Net Thermal Radiation Heat Gain Rate per Area'] = [1e9, -1e9]
        np.testing.assert_array_equal(forcing, build_forcing(building, data, np.array([300., 310.])))

    def test_variable_model_equilibrium_and_timestep_convergence(self):
        records = []
        for dt in [60, 30, 15]:
            building = FreeBuilding(*case(), dt=dt, cells=4,
                                    interior='tarp', exterior='doe2', sky='energyplus')
            forcing = np.tile([291.15, 0., 0., 5.67e-8*291.15**4, 0.], (6, 1))
            building.rooms[1].Tint = 291.15
            for e in building.walls:
                e['wall'].initialize(dt, 291.15, 291.15, windDirection=0.)
            original = building.state()
            building.step(forcing)
            np.testing.assert_allclose(original, building.state(), atol=1e-10)
            forcing[:, 0] = 301.15
            forcing[:, 1] = 2.
            forcing[:, 2] = 300.
            temperatures = []
            for k in range(int(7200/dt)):
                building.step(forcing)
                if (k+1) % int(60/dt) == 0:
                    temperatures.append(building.rooms[1].Tint)
            self.assertEqual(building.walls[-1]['wall'].convection.back, 'fixed')
            self.assertLess(building.max_energy_residual, 1e-6)
            records.append(np.array(temperatures))
        coarse = np.sqrt(np.mean((records[0]-records[1])**2))
        fine = np.sqrt(np.mean((records[1]-records[2])**2))
        self.assertLess(fine, .7*coarse)
