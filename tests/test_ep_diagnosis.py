import unittest
import numpy as np
from test_ep_free_running import case
from scripts.compare_ep_free_running import FreeBuilding
from scripts.diagnose_ep_discrepancies import CounterfactualBuilding


class DiagnosisTests(unittest.TestCase):
    def test_control_reproduces_production_driver(self):
        baseline=FreeBuilding(*case(),dt=60,cells=4)
        control=CounterfactualBuilding(*case(),dt=60,cells=4,variant='control',directions=np.zeros((1,6)))
        forcing=np.tile([295.15,2.,300.,350.],(6,1))
        for _ in range(30):
            expected=baseline.step(forcing)
            actual=control.step(forcing)
            np.testing.assert_array_equal(actual[0],expected[0])
            np.testing.assert_array_equal(actual[1],expected[1])
        np.testing.assert_array_equal(control.state(),baseline.state())

    def test_correlation_variants_preserve_energy_balance(self):
        forcing=np.tile([295.15,2.,300.,350.],(6,1))
        for variant in ['exterior_convection','sky','exterior_and_sky','all_convection_and_sky']:
            building=CounterfactualBuilding(*case(),dt=60,cells=4,variant=variant,directions=np.zeros((1,6)))
            original=forcing.copy()
            for _ in range(100):
                building.step(forcing)
            np.testing.assert_array_equal(forcing,original)
            self.assertLess(building.max_energy_residual,1e-6)
            self.assertTrue(np.isfinite(building.state()).all())
