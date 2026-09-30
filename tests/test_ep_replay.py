"""Check the boundary replay against steady heat-transfer solutions."""
import unittest
import numpy as np
import pandas as pd
from scripts.compare_ep_replay import make_wall, advance


class ReplayTests(unittest.TestCase):
    def setUp(self):
        layers = pd.DataFrame([dict(key='Material', Thickness=.1, Conductivity=.2,
                                   Density=1000., Specific_Heat=1000., Thermal_Resistance=.5)])
        self.wall = make_wall(layers, 9, 1e9)
        # Columns: surface T, air T, h, radiation, reference conduction/convection.
        self.boundary = np.array([[300., 295., 2., 10., 0., 0.],
                                  [310., 315., 5., 25., 0., 0.]])

    def steady(self, mode, ground=False):
        for _ in range(10):
            result = advance(self.wall, self.boundary, self.boundary, mode, ground, 1)
        return result

    def test_robin_with_radiation(self):
        result = self.steady('robin')
        # Radiation shifts the equivalent boundary air temperatures by q/h.
        flux = ((315 + 25/5) - (295 + 10/2)) / (.5 + 1/2 + 1/5)
        np.testing.assert_allclose(result[:, 1], [flux, -flux], atol=1e-7)
        np.testing.assert_allclose(result[:, 0], [295 + (flux+10)/2, 315 + (25-flux)/5], atol=1e-7)

    def test_ground_is_prescribed_surface_temperature(self):
        result = self.steady('robin', ground=True)
        flux = (310 - (295 + 10/2)) / (.5 + 1/2)
        np.testing.assert_allclose(result[:, 1], [flux, -flux], atol=1e-6)
        self.assertAlmostEqual(result[1, 0], 310, places=6)

    def test_dirichlet_ignores_air_and_radiation(self):
        result = self.steady('dirichlet')
        np.testing.assert_allclose(result[:, 0], [300, 310], atol=1e-6)
        np.testing.assert_allclose(result[:, 1], [20, -20], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
