"""Physical checks for the wall solver; run with unittest discovery."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from model.WallSimulation import WallSimulation, processMaterials
from model.utils import WallSides

ROOT = Path(__file__).resolve().parents[1]


def solid(thickness=.1, conductivity=.1, density=1000, specific_heat=1000):
    return dict(key='Material', Thickness=thickness, Conductivity=conductivity,
                Density=density, Specific_Heat=specific_heat, Thermal_Resistance=np.nan)


def gap(resistance=.15):
    return dict(key='Material:AirGap', Thickness=np.nan, Conductivity=np.nan,
                Density=np.nan, Specific_Heat=np.nan, Thermal_Resistance=resistance)


def wall(material=None, n=9, dt=30, implicit=True, h=(2, 2), roughness=(0, 0)):
    return WallSimulation(X=4, Y=3,
                          material_df=pd.DataFrame([solid()]) if material is None else material,
                          h=WallSides(*h), roughness=WallSides(*roughness),
                          absorptivity=.7, n=n, delt=dt, implicit=implicit)


def constructions():
    definitions = pd.read_csv(ROOT / 'energyPlus/My_Constructions.csv', index_col='Name')
    materials = pd.read_csv(ROOT / 'energyPlus/ASHRAE_2005_HOF_Materials.csv', index_col='Name')
    for name, row in definitions.iterrows():
        names = [row[k] for k in ['Outside_Layer', 'Layer_2', 'Layer_3', 'Layer_4', 'Layer_5']
                 if pd.notna(row[k])]
        layers = materials.loc[names[::-1] if name != 'My Floor' else names].copy()
        if name == 'My Floor':
            layers = pd.concat([layers, pd.DataFrame([solid(.5, 1.5, 2800, 850)], index=['Soil'])])
        yield name, layers


class WallPhysicsTests(unittest.TestCase):
    def test_material_resistance_and_mass_are_preserved(self):
        for name, layers in constructions():
            with self.subTest(name=name):
                original = layers.copy(deep=True)
                processed = processMaterials(layers, 9, dt=1e6)
                pd.testing.assert_frame_equal(layers, original)
                pd.testing.assert_frame_equal(processed[layers.columns.drop('Thermal_Resistance')],
                                              layers.drop(columns='Thermal_Resistance'))
                w = wall(layers)
                resistance = layers.Thickness.div(layers.Conductivity).fillna(layers.Thermal_Resistance).sum()
                mass = (layers.Thickness * layers.Density * layers.Specific_Heat).sum()
                self.assertAlmostEqual(w.total_resistance, resistance)
                self.assertAlmostEqual(w.capacity.sum(), mass)
                self.assertEqual(len(w.x), w.n + 2)
                self.assertTrue(np.all(np.diff(w.x) > 0))

    def test_steady_flux_matches_total_resistance(self):
        cases = [('uniform', pd.DataFrame([solid()]))] + list(constructions())
        # Also exercise leading, trailing and consecutive resistance-only layers.
        cases.append(('boundary gaps', pd.DataFrame([gap(), gap(.2), solid(), gap(.3)])))
        for name, layers in cases:
            for n in [1, 9, 30]:
                with self.subTest(name=name, n=n):
                    w = wall(layers, n=n, h=(2, 7))
                    w.initialize(1e9, 300, 310)
                    for _ in range(8):
                        ef = w.timeStep(300, 310)
                    expected = 10 / (1/2 + w.total_resistance + 1/7)
                    self.assertAlmostEqual(ef.front / w.Af, expected, places=8)
                    self.assertAlmostEqual(-ef.back / w.Af, expected, places=8)
                    interior_flux = w.link_conductance * np.diff(w.T)
                    np.testing.assert_allclose(interior_flux, expected, atol=1e-8)

    def test_transient_storage_and_surfaces_conserve_heat(self):
        for implicit in [True, False]:
            for n in [1, 9]:
                with self.subTest(implicit=implicit, n=n):
                    w = wall(pd.DataFrame([solid(.02, .2), gap(), solid(.08, 1.5)]),
                             n=n, dt=.1, implicit=implicit, roughness=(.5, 1.64))
                    w.initialize(.1, 300, 305)
                    for i in range(60):
                        w.windSpeed = (i % 7) * .8
                        w.Erad = WallSides(20*np.sin(i), 100*np.cos(i/7))
                        before = w.T.copy()
                        air = (299 + np.sin(i/3), 308 + np.cos(i/5))
                        ef = w.timeStep(*air)
                        storage = np.dot(w.capacity, w.T-before) / w.delt
                        net_input = w.Erad.front + w.Erad.back - (ef.front+ef.back)/w.Af
                        self.assertAlmostEqual(storage, net_input, delta=2e-6)
                        for j, side, ta in [(0, 'front', air[0]), (-1, 'back', air[1])]:
                            g = getattr(w.surface_conductance, side)
                            h = getattr(w.hCalced, side)
                            rad = getattr(w.Erad, side)
                            residual = g*(w.T[j]-w.T_prof[j]) + h*(ta-w.T_prof[j]) + rad
                            self.assertAlmostEqual(residual, 0, delta=1e-8)

    def test_one_cell_accumulates_both_boundary_sources(self):
        w = wall(n=1)
        w.initialize(60, 300, 300)
        w.Erad = WallSides(80, 120)
        g = 2 * .1 / .1
        effective_h = g * 2 / (g + 2)
        expected = (w.capacity[0]/60*300 + 2*effective_h*300 + g/(g+2)*200) / (w.capacity[0]/60 + 2*effective_h)
        w.timeStep(300, 300)
        self.assertAlmostEqual(w.T[0], expected)

    def test_insulated_wall_keeps_energy(self):
        w = wall(pd.DataFrame([solid(.03, .1), gap(), solid(.07, 1)]), h=(0, 0))
        w.initialize(30, 300, 310)
        before = np.dot(w.capacity, w.T)
        for _ in range(100):
            ef = w.timeStep(250, 400)
            self.assertEqual(ef.front, 0)
            self.assertEqual(ef.back, 0)
        self.assertAlmostEqual(np.dot(w.capacity, w.T), before, delta=1e-5)

    def test_large_ground_coefficient_conserves_heat(self):
        w = wall(h=(2, 1e6))
        w.initialize(30, 300, 295)
        before = w.T.copy()
        ef = w.timeStep(303, 290)
        self.assertAlmostEqual(np.dot(w.capacity, w.T-before)/30,
                               -(ef.front+ef.back)/w.Af, delta=1e-7)
        self.assertAlmostEqual(w.T_prof[-1], 290, delta=.001)

    def test_explicit_stability_rechecked_after_wind_change(self):
        w = wall(n=1, implicit=False, roughness=(0, 1.64))
        w.initialize(40000, 300, 310)
        w.timeStep(300, 310)
        w.windSpeed = 100
        with self.assertRaisesRegex(ValueError, 'stability'):
            w.timeStep(300, 310)

    def test_invalid_materials_fail_clearly(self):
        for material in [pd.DataFrame([gap()]), pd.DataFrame([solid(conductivity=0)]),
                         pd.DataFrame([solid(), gap(-1)])]:
            with self.assertRaises(ValueError):
                wall(material)

    def test_diffusion_converges_in_space_and_time(self):
        def error(n, dt):
            w = wall(n=n, h=(1e12, 1e12))
            w.initialize(dt, 300, 300)
            w.T = 300 + np.sin(np.pi*w.x[1:-1]/.1)
            for _ in range(round(5000/dt)):
                w.timeStep(300, 300)
            exact = 300 + np.sin(np.pi*w.x[1:-1]/.1)*np.exp(-1e-7*(np.pi/.1)**2*5000)
            return np.sqrt(np.mean((w.T-exact)**2))
        spatial = [error(n, 1) for n in [5, 10, 20]]
        temporal = [error(60, dt) for dt in [100, 50, 25]]
        self.assertTrue(spatial[0] > spatial[1] > spatial[2], spatial)
        self.assertTrue(temporal[0] > temporal[1] > temporal[2], temporal)
        self.assertLess(spatial[-1], .001)
        self.assertLess(temporal[-1], .001)


if __name__ == '__main__':
    unittest.main()
