import unittest
import numpy as np

class MyTestCase(unittest.TestCase):
    testProbs1 = np.ndarray(shape=(4,4), buffer=np.array([0.1, 0.5, 0.0, 0.4,
                                                          0.5, 0.1, 0.3, 0.1,
                                                          0.3, 0.1, 0.0, 0.3,
                                                          0.4, 0.1, 0.2, 0.3]))
    testscores1 = np.ndarray(buffer=np.array([1, 2, 3, 4]))
    def test_something(self):
        self.assertEqual(testProbs1, True)


if __name__ == '__main__':
    unittest.main()
