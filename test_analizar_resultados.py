import unittest

from analizar_resultados import (
    bootstrap_mean_interval,
    exact_binomial_two_sided,
    wilson,
)


class TestEstadistica(unittest.TestCase):

    def test_wilson_y_binomial_bajo_hipotesis_nula(self):
        low, high = wilson(50, 100)
        self.assertAlmostEqual(low, 0.4038, places=3)
        self.assertAlmostEqual(high, 0.5962, places=3)
        self.assertAlmostEqual(exact_binomial_two_sided(50, 100), 1.0)

    def test_bootstrap_es_reproducible(self):
        data = [-4, -2, 0, 2, 4]
        first = bootstrap_mean_interval(data, resamples=1000, seed=2025)
        second = bootstrap_mean_interval(data, resamples=1000, seed=2025)
        self.assertEqual(first, second)
        self.assertLess(first[0], first[1])


if __name__ == '__main__':
    unittest.main()
