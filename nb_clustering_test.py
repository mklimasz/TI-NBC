import unittest

import numpy as np

import nb_clustering


class NeighborhoodBasedClusteringTest(unittest.TestCase):

    def test_nbc(self):
        # given
        cluster_0 = np.array([
            [0, 0],
            [1, 1],
            [2, 2]
        ])
        # Cluster 1 moved "far" from Cluster 0
        cluster_1 = cluster_0 + 10
        k = 2

        # when
        c, _ = nb_clustering.nbc(vectors=np.vstack((cluster_0, cluster_1)), k=k)

        # then
        expected_clusters = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
        self.assertDictEqual(c, expected_clusters)


if __name__ == '__main__':
    unittest.main()
