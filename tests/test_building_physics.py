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

    def test_air_volume_matches_floor_plan(self):
        sim = example()
        for room, expected in dict(CR=48, SS=48, DR=96, CV=96).items():
            floor = sim.bG.G.edges[room, 'FL']
            volume = sim.bG.G.nodes[room]['room'].V
            self.assertEqual(volume, expected)
            self.assertEqual(volume, floor['wall'].Af * floor['weight'] * 3)

    def test_radiation_is_applied_without_temporal_filter(self):
        for dt in [15, 30, 60]:
            sim = example(dt=dt)
            sim.run()
            for _, _, edge in sim.bG.G.edges(data=True):
                for side in ['front', 'back']:
                    np.testing.assert_array_equal(getattr(edge['radECalc'], side),
                                                  getattr(edge['radEApplied'], side))

    def test_explicit_radiation_coupling_converges(self):
        temperatures = []
        for dt in [30, 15, 7.5]:
            sim = example(dt=dt, steps=int(7200/dt))
            for node in ['RF', 'OD']:
                sim.radG[node]['shortwave'] = 400*np.sin(np.pi*sim.times/7200)**2
            sim.run()
            values = np.array([sim.bG.G.nodes[r]['Tints'][::int(30/dt)]
                               for r in ['CR', 'SS', 'DR', 'CV']])
            self.assertTrue(np.isfinite(values).all())
            temperatures.append(values)
        coarse = np.sqrt(np.mean((temperatures[0]-temperatures[1])**2))
        fine = np.sqrt(np.mean((temperatures[1]-temperatures[2])**2))
        self.assertLess(fine, .7*coarse)
