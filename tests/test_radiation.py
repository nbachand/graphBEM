import unittest
from types import SimpleNamespace
import numpy as np
from model.Radiation import Radiation
from model.utils import WallSides


def surface(name, X=4, Y=3, alpha=.7, temperature=300, weight=1):
    wall = SimpleNamespace(X=X, Y=Y, absorptivity=alpha,
                           T_prof=np.array([temperature, temperature], dtype=float))
    return dict(wall=wall, weight=weight, nodes=WallSides('R', name))


def enclosure():
    return {'wall': surface('wall'), 'RF': surface('RF', Y=4, temperature=310),
            'FL': surface('FL', Y=4, temperature=290)}


class RadiationTests(unittest.TestCase):
    def test_interior_emissivity_is_independent_of_solar_absorptivity(self):
        mapping = enclosure()
        r = Radiation(solveType='room', emissivity=.9, storyHeight=3.05)
        r.initialize(mapping)
        self.assertAlmostEqual(r.G.nodes['RF']['boundaryResistance'], .1/(.9*16))
        self.assertEqual(mapping['RF']['wall'].absorptivity, .7)
        self.assertEqual(r.storyHeight, 3.05)

    def test_shared_edge_tolerates_roundoff_in_imported_dimensions(self):
        mapping = enclosure()
        mapping['wall']['wall'].X = 12.2/3.05
        r = Radiation(solveType='room')
        r.initialize(mapping)
        self.assertTrue(np.isfinite(r.timeStep()).all())

    def test_exterior_equilibrium_and_cold_sky(self):
        for alpha in [.6, .75, 1]:
            r = Radiation(solveType='sky')
            r.initialize({'R': surface('OD', alpha=alpha)}, longwaveGain=r.sigma*300**4)
            self.assertAlmostEqual(r.timeStep()['R'], 0)
            r.longwaveGain = 300
            self.assertAlmostEqual(r.timeStep()['R'], .9*(300-r.sigma*300**4))
            r.solarGain = 500
            self.assertAlmostEqual(r.timeStep()['R'], alpha*500+.9*(300-r.sigma*300**4))

    def test_black_enclosure_conserves_and_matches_edge_exchange(self):
        mapping = enclosure()
        for edge in mapping.values():
            edge['wall'].absorptivity = 1
        r = Radiation(solveType='room')
        r.initialize(mapping)
        q = r.timeStep()
        self.assertTrue(np.isfinite(q).all())
        self.assertAlmostEqual(sum(q[n]*d['A'] for n,d in r.G.nodes(data=True)),0,places=10)
        expected = sum(r.sigma*(mapping[j]['wall'].T_prof[0]**4-310**4)/e['radianceResistance']
                       for j,e in r.G['RF'].items()) / 16
        self.assertAlmostEqual(q['RF'], expected)
        for edge in mapping.values(): edge['wall'].T_prof[:] = 300
        np.testing.assert_allclose(r.timeStep(),0,atol=1e-10)

    def test_surface_order_does_not_change_exchange(self):
        from building_fixture import example
        sim = example()
        rng = np.random.default_rng(42)
        for room in ['CR', 'SS', 'DR', 'CV']:
            mapping = dict(sim.bG.G[room])
            for other, edge in mapping.items():
                edge['wall'].T_prof[:] = 310 if other == 'RF' else 290 if other == 'FL' else 300
            ref = Radiation(solveType='room')
            ref.initialize(mapping)
            q = ref.timeStep().sort_index()
            permutations = [list(reversed(mapping))] + [rng.permutation(list(mapping)).tolist() for _ in range(12)]
            for order in permutations:
                candidate = Radiation(solveType='room')
                candidate.initialize({key: mapping[key] for key in order})
                np.testing.assert_allclose(candidate.timeStep().sort_index(), q, atol=1e-10)
                for a, b, edge in ref.G.edges(data=True):
                    self.assertAlmostEqual(edge['radianceResistance'], candidate.G[a][b]['radianceResistance'])
