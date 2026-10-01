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
