import unittest

import numpy as np

import nb


class KNeighbourhoodTest(unittest.TestCase):

    def test_each_with_each(self):
        # given
        k = 2
        vectors = np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 1, 1],
        ])

        # when
        knb, r_knb = nb.k_neighbourhood(vectors, k)

        # then
        expected_nb = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
        self.assertDictEqual(knb, expected_nb)

    def test_multiple_points_on_border(self):
        # given
        k = 2
        vectors = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [2, 3]
        ])

        # when
        knb, r_knb = nb.k_neighbourhood(vectors, k)

        # then
        self.assertSetEqual(knb[2], {1, 3, 4})

        for idx in [1, 3, 4]:
            self.assertIn(2, r_knb[idx])


class NeighbourhoodBasedDensityFactorTest(unittest.TestCase):

    def test_ndf(self):
        # given
        # *     - 0
        # *     - 1
        # * *   - 2 3
        knb = {0: {1, 2}, 1: {0, 2}, 2: {1, 3}, 3: {1, 2}}
        r_knb = {0: {1}, 1: {0, 2, 3}, 2: {0, 1, 3}, 3: {2}}

        # when
        ndf = nb.ndf(knb, r_knb)

        # then
        expected_ndfs = {0: 0.5, 1: 1.5, 2: 1.5, 3: 0.5}
        self.assertEqual(ndf, expected_ndfs)


if __name__ == '__main__':
    unittest.main()
