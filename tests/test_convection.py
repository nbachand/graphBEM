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
