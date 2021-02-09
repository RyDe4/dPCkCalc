import unittest
import numpy as np
import igraph
from DispersalCalc import calc_pc_numerator
from DispersalCalc import make_removed
from DispersalCalc import calc_dpck
from DispersalCalc import calc_dpck_intra
from DispersalCalc import calc_dpck_flux
from DispersalCalc import calc_dpck_connector
import pytest

class MyTestCase(unittest.TestCase):
    testProbs1 = np.ndarray(shape=(4,4), buffer=np.array([0.1, 0.5, 0.0, 0.4,
                                                          0.5, 0.1, 0.3, 0.1,
                                                          0.0, 0.3, 0.5, 0.2,
                                                          0.4, 0.1, 0.2, 0.3]))
    testScores1 = np.array((1, 2, 3, 4))

    def test_calc_pc_numerator(self):
        # create igraph graph object from loaded adjacency matrix
        g = igraph.Graph.Adjacency((self.testProbs1 > 0).tolist(), mode="DIRECTED")
        g.es['weight'] = -1 * np.log(self.testProbs1[self.testProbs1.nonzero()])
        # find the highest probability path between patches using dijkstra with 1 on diag and then undoing -log operation
        init_probs = np.exp(-1 * np.array(g.shortest_paths_dijkstra(weights='weight')))
        self.assertEqual(calc_pc_numerator(init_probs, self.testScores1), 47.7)

    def test_calc_dpck(self):
        # create igraph graph object from loaded adjacency matrix
        g = igraph.Graph.Adjacency((self.testProbs1 > 0).tolist(), mode="DIRECTED")
        g.es['weight'] = -1 * np.log(self.testProbs1[self.testProbs1.nonzero()])
        # find the highest probability path between patches using dijkstra with 1 on diag and then undoing -log operation
        init_probs = np.exp(-1 * np.array(g.shortest_paths_dijkstra(weights='weight')))
        g_removed_probs = make_removed(0, g)
        init_pc = calc_pc_numerator(init_probs, self.testScores1)
        assert pytest.approx(calc_dpck(0, g_removed_probs, self.testScores1, init_pc), 0.000001) == 18.23899371

    def test_calc_dpck_multiple(self):
        # create igraph graph object from loaded adjacency matrix
        g = igraph.Graph.Adjacency((self.testProbs1 > 0).tolist(), mode="DIRECTED")
        g.es['weight'] = -1 * np.log(self.testProbs1[self.testProbs1.nonzero()])
        # find the highest probability path between patches using dijkstra with 1 on diag and then undoing -log operation
        init_probs = np.exp(-1 * np.array(g.shortest_paths_dijkstra(weights='weight')))
        g_removed_probs_1 = make_removed(1, g)
        g_removed_probs_0 = make_removed(0, g)
        init_pc = calc_pc_numerator(init_probs, self.testScores1)
        calc_dpck(1, g_removed_probs_1, self.testScores1, init_pc)
        assert pytest.approx(calc_dpck(0, g_removed_probs_0, self.testScores1, init_pc), 0.000001) == 18.23899371

    def test_calc_dpck_intra(self):
        # create igraph graph object from loaded adjacency matrix
        g = igraph.Graph.Adjacency((self.testProbs1 > 0).tolist(), mode="DIRECTED")
        g.es['weight'] = -1 * np.log(self.testProbs1[self.testProbs1.nonzero()])
        # find the highest probability path between patches using dijkstra with 1 on diag and then undoing -log operation
        init_probs = np.exp(-1 * np.array(g.shortest_paths_dijkstra(weights='weight')))
        g_removed_probs = make_removed(0, g)
        init_pc = calc_pc_numerator(init_probs, self.testScores1)
        assert pytest.approx(calc_dpck_intra(0, self.testScores1, init_pc), 0.000001) == 2.096436059

    def test_calc_dpck_flux(self):
        # create igraph graph object from loaded adjacency matrix
        g = igraph.Graph.Adjacency((self.testProbs1 > 0).tolist(), mode="DIRECTED")
        g.es['weight'] = -1 * np.log(self.testProbs1[self.testProbs1.nonzero()])
        # find the highest probability path between patches using dijkstra with 1 on diag and then undoing -log operation
        init_probs = np.exp(-1 * np.array(g.shortest_paths_dijkstra(weights='weight')))
        g_removed_probs = make_removed(0, g)
        init_pc = calc_pc_numerator(init_probs, self.testScores1)
        assert pytest.approx(calc_dpck_flux(0, self.testProbs1, self.testScores1, init_pc, 4), 0.000001) == 10.9014675

# 5.241090147
if __name__ == '__main__':
    unittest.main()
