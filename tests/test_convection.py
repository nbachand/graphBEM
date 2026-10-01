"""Checks of physical limits and face conventions for convection."""
import unittest
import numpy as np
from model.utils import WallSides
from test_wall_simulation import wall


class ConvectionTests(unittest.TestCase):
    def test_fixed_coefficient_ignores_wind_and_preserves_flux_sign(self):
        w = wall(h=(2, 3), roughness=(1.11, 1.11))
        w.convection = WallSides('fixed', 'fixed')
        w.initialize(30, 300, 300, windSpeed=10)
        power = w.timeStep(290, 310)
        self.assertEqual(w.hCalced.front, 2)
        self.assertEqual(w.hCalced.back, 3)
        self.assertGreater(power.front, 0)
        self.assertLess(power.back, 0)
        self.assertEqual(w.convection_metadata()['front']['model'], 'fixed')

    def test_exterior_limits_and_wind_direction(self):
        from model.Convection import exterior_convection as h
        self.assertAlmostEqual(h(308, 300, 0, 1.11, 90, 0, 0), 2.62)
        self.assertEqual(h(300, 300, 0, 1.11, 0), 0)
        self.assertAlmostEqual(h(300, 300, 4, 1, 90, 0, 359), 3.26*4**.89)
        self.assertAlmostEqual(h(300, 300, 4, 1, 90, 0, 180), 3.55*4**.617)
        self.assertEqual(h(300, 300, 4, 1, 0), h(300, 300, 4, 1, 90, 0, 0))
        with self.assertRaises(ValueError):
            h(300, 300, 4, 1, 90)
        with self.assertRaises(ValueError):
            h(300, 300, -1, 1, 0)
        self.assertAlmostEqual(h(300, 300, 4, 1, 90, exposure='average'),
                               (3.26*4**.89+3.55*4**.617)/2)

    def test_buoyancy_reversal_and_interior_minimum(self):
        from model.Convection import natural_convection as h
        self.assertGreater(h(308, 300, 0), h(292, 300, 0))
        self.assertAlmostEqual(h(308, 300, 0), h(292, 300, 180))
        self.assertAlmostEqual(h(308, 300, 180), h(292, 300, 0))
        self.assertAlmostEqual(h(308, 300, 90), h(292, 300, 90))
        w = wall()
        w.convection = WallSides('tarp', 'tarp')
        w.tilt = WallSides(0, 180)
        w.initialize(30, 300, 300)
        self.assertEqual(w.hCalced.front, .1)
        w.T_prof[[0, -1]] = 308
        w.timeStep(300, 300)
        self.assertGreater(w.hCalced.front, w.hCalced.back)

    def test_variable_convection_conserves_wall_energy(self):
        w = wall(roughness=(0, 1.11))
        w.convection = WallSides('tarp', 'doe2')
        w.tilt = WallSides(180, 0)
        w.initialize(30, 295, 305)
        for k in range(50):
            w.windSpeed = k % 5
            w.Erad = WallSides(-20., 200.)
            before = w.T.copy()
            power = w.timeStep(295+np.sin(k), 305+np.cos(k))
            storage = w.capacity @ (w.T-before)/w.delt
            self.assertAlmostEqual(storage, 180-(power.front+power.back)/w.Af, delta=1e-7)
