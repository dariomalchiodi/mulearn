from unittest import TestCase

from mulearn.kernel import *


class TestLinearKernel(TestCase):
    def test_compute(self):
        k = LinearKernel()
        self.assertEqual(k.compute([1, 0, 1], [2, 2, 2]), 4)
        self.assertEqual(k.compute((1, 0, 2), (-1, 2, 5)), 9)
        self.assertAlmostEqual(k.compute([1.2, -0.4, -2], [4, 1.2, .5]), 3.32)
        self.assertAlmostEqual(k.compute((1.2, -0.4, -2), [4, 1.2, .5]), 3.32)
        with self.assertRaises(ValueError):
            k.compute([1, 0, 1], [2, 2])


class TestPolynomialKernel(TestCase):
    def test_compute(self):
        with self.assertRaises(ValueError):
            PolynomialKernel(3.2)

        with self.assertRaises(ValueError):
            PolynomialKernel(-2)

        p = PolynomialKernel(2)
        self.assertEqual(p.compute((1, 0, 2), (-1, 2, 5)), 100)
        self.assertAlmostEqual(p.compute([1.2, -0.4, -2], [4, 1.2, .5]),
                               18.6624)

        p = PolynomialKernel(5)
        self.assertEqual(p.compute((1, 0, 2), [-1, 2, 5]), 10 ** 5)
        self.assertAlmostEqual(p.compute((1.2, -0.4, -2), (4, 1.2, .5)),
                               1504.59195, delta=10**-6)

        with self.assertRaises(ValueError):
            p.compute((1, 0, 2), (-1, 2))


class TestHomogeneousPolynomialKernel(TestCase):
    def test_compute(self):
        with self.assertRaises(ValueError):
            HomogeneousPolynomialKernel(3.2)

        with self.assertRaises(ValueError):
            HomogeneousPolynomialKernel(-2)

        h = HomogeneousPolynomialKernel(2)
        self.assertEqual(h.compute((1, 0, 2), (-1, 2, 5)), 81.0)
        self.assertAlmostEqual(h.compute([1.2, -0.4, -2], [4, 1.2, .5]),
                               11.0224)

        h = HomogeneousPolynomialKernel(5)
        self.assertEqual(h.compute((1, 0, 2), [-1, 2, 5]), 59049.0)
        self.assertAlmostEqual(h.compute((1.2, -0.4, -2), (4, 1.2, .5)),
                               403.357761, delta=10**-6)

        with self.assertRaises(ValueError):
            h.compute((1, 0, 2), (-1, 2)
                      )


class TestGaussianKernel(TestCase):
    def test_compute(self):
        with self.assertRaises(ValueError):
            GaussianKernel(-5)

        k = GaussianKernel(1)
        self.assertAlmostEqual(k.compute((1, 0, 1), (0, 0, 1)), 0.60653065)
        self.assertAlmostEqual(k.compute([-3, 1, 0.5], [1, 1.2, -8]), 6.73e-20)
        self.assertAlmostEqual(k.compute([-1, -4, 3.5], (1, 3.2, 6)), 3.29e-14)

        with self.assertRaises(ValueError):
            k.compute([-1, 3.5], (1, 3.2, 6))


class TestHyperbolicKernel(TestCase):
    def test_compute(self):
        k = HyperbolicKernel(1, 5)
        self.assertAlmostEqual(k.compute((1, 0, 1), (0, 0, 1)), 0.9999877)
        self.assertAlmostEqual(k.compute([-3, 1, 0.5], [1, 1.2, -8]),
                               -0.6640367, delta=10**-7)
        self.assertAlmostEqual(k.compute([-1, -4, 3.5], (1, 3.2, 6)),
                               0.9999999, delta=10**-7)

        with self.assertRaises(ValueError):
            k.compute([-1, 3.5], (1, 3.2, 6))


class TestPrecomputedKernel(TestCase):
    def test_compute(self):
        with self.assertRaises(ValueError):
            PrecomputedKernel(((1, 2), (3, 4, 5)))

        k = PrecomputedKernel(((1, 2), (3, 4)))
        self.assertEqual(k.compute([1], [1]), 4.0)
        self.assertEqual(k.compute([1], [0]), 3.0)

        with self.assertRaises(IndexError):
            k.compute([1], [2])

        with self.assertRaises(TypeError):
            k.compute([0], [1.6])
