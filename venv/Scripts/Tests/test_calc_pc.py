import unittest
import numpy as np
import igraph
from DispersalCalc import calc_pc

class MyTestCase(unittest.TestCase):
    testProbs1 = np.ndarray(shape=(4,4), buffer=np.array([0.1, 0.5, 0.0, 0.4,
                                                          0.5, 0.1, 0.3, 0.1,
                                                          0.0, 0.3, 0.5, 0.2,
                                                          0.4, 0.1, 0.2, 0.3]))
    testScores1 = np.array((1, 2, 3, 4))
    def test_something(self):
        # create igraph graph object from loaded adjacency matrix
        g = igraph.Graph.Adjacency((self.testProbs1 > 0).tolist(), mode="DIRECTED")
        g.es['weight'] = -1 * np.log(self.testProbs1[self.testProbs1.nonzero()])
        # find the highest probability path between patches using dijkstra with 1 on diag and then undoing -log operation
        init_probs = np.exp(-1 * np.array(g.shortest_paths_dijkstra(weights='weight')))
        self.assertEqual(calc_pc(init_probs, self.testScores1), 47.7)


if __name__ == '__main__':
    unittest.main()
