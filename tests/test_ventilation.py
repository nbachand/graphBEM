import unittest
import numpy as np
from scipy.integrate import quad
from model.VentilationSimulation import VentilationSimulation


def vent(scale=1):
    return VentilationSimulation(H=scale, W=scale, ventType='HWP4',
                                 alphas=[45], As=[scale**2], Ls=[1])


class VentilationTests(unittest.TestCase):
    def test_published_width_and_geometric_similarity(self):
        # Independent Eq. 4 quadrature, split at the change in window geometry.
        alpha = np.pi/4
        h = 1-np.cos(alpha)
        f = lambda z: (1 + (2*(1-z)*np.tan(alpha)+np.sin(alpha))**-2)**-.5
        expected = .611*(h + quad(f,h,1)[0])
        for scale in [0.1, 1, 2, 4]:
            self.assertAlmostEqual(vent(scale).get_Cd(alpha), expected, delta=2e-4)
        values = [vent(s).get_Cd(alpha) for s in [.1, 1, 2, 4]]
        np.testing.assert_allclose(values, values[0], atol=1e-12)
        v = vent()
        self.assertEqual(v.get_Cd(0), 0)
        self.assertLess(v.get_Cd(1e-8), 1e-7)
        self.assertAlmostEqual(v.get_Cd(np.pi/2), .611)

    def test_scalar_vector_schedule_and_heat_sign(self):
        v = vent()
        times = np.array([0, 6, 7, 12, 19, 20, 24])*3600
        vector = v.timeStep(times,Tint=300.,Tout=290.)
        scalar = [v.timeStep(float(t),Tint=300.,Tout=290.) for t in times]
        np.testing.assert_allclose(vector, scalar)
        self.assertTrue(np.all(vector[[0,1,5,6]] < 0))
        np.testing.assert_array_equal(vector[[2,3,4]],0)
        self.assertGreater(v.timeStep(0,Tint=290.,Tout=300.),0)
        self.assertEqual(v.timeStep(0,Tint=300.,Tout=300.),0)
        varying = np.array([300.,301.,302.])
        self.assertEqual(v.get_Vnv(varying,290.,[0,1,2]).shape,(1,3))

    def test_multiple_windows_and_no_windows(self):
        one = vent().timeStep(0,Tint=300.,Tout=290.)
        v = VentilationSimulation(H=1,W=1,ventType='HWP4',alphas=[45,45],As=[1,1],Ls=[1,1])
        self.assertAlmostEqual(v.timeStep(0,Tint=300.,Tout=290.),2*one)
        v = VentilationSimulation(H=1,W=1,ventType='HWP4',alphas=[],As=[],Ls=[])
        self.assertEqual(v.timeStep(0,Tint=300.,Tout=290.),0)
