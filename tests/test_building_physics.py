import unittest
import numpy as np
from building_fixture import example
from model.utils import WallSides


class BuildingPhysicsTests(unittest.TestCase):
    def test_both_face_update(self):
        sides = WallSides(0, 0)
        sides.setUpdateBoth()
        sides.update(10)
        self.assertEqual((sides.front, sides.back), (10, 10))
        sides.setUpdateBack()
        sides.update(20)
        self.assertEqual((sides.front, sides.back), (10, 20))

    def test_partition_faces_and_room_radiation_conserve(self):
        sim = example(steps=15)
        sim.run()
        loop = sim.bG.G.edges['DR', 'DR']
        self.assertGreater(np.max(np.abs(loop['radEApplied'].front)), .001)
        np.testing.assert_allclose(loop['radEApplied'].front, loop['radEApplied'].back)
        np.testing.assert_allclose(loop['T_profs'][0], loop['T_profs'][-1], atol=1e-10)
        for room in ['CR', 'SS', 'DR', 'CV']:
            power = np.zeros(sim.N)
            for _, edge in sim.bG.G[room].items():
                for side in ['front', 'back']:
                    if getattr(edge['nodes'], side) == room:
                        power += edge['wall'].Af * edge['weight'] * getattr(edge['radEApplied'], side)
            np.testing.assert_allclose(power, 0, atol=1e-8)
