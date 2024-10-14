
import numpy as np
import unittest

import mulearn.kernel as kernel


class Test_LinearKernel(unittest.TestCase):
    def test_compute(self):
        k =kernel.LinearKernel()
        self.assertEqual(k.compute(np.array([1, 0, 1]).reshape(1,-1), 
                                   np.array([2, 2, 2]).reshape(1,-1)), 4)
        self.assertEqual(k.compute(np.array((1, 0, 2)).reshape(1,-1), 
                                   np.array((-1, 2, 5)).reshape(1,-1)), 9)
                         
        self.assertAlmostEqual(k.compute(
                    np.array([1.2, -0.4, -2]).reshape(1,-1), 
                    np.array([4, 1.2, .5]).reshape(1,-1))[0], 3.32)
        self.assertAlmostEqual(k.compute(
                    np.array((1.2, -0.4, -2)).reshape(1,-1), 
                    np.array([4, 1.2, .5]).reshape(1,-1))[0], 3.32)

        with self.assertRaises(ValueError):
            k.compute(np.array([1, 0, 1]).reshape(1,-1), 
                      np.array([2, 2]).reshape(1,-1))


class TestPolynomialKernel(unittest.TestCase):
    def test_compute(self):
        with self.assertRaises(ValueError):
            kernel.PolynomialKernel(3.2)

        with self.assertRaises(ValueError):
            kernel.PolynomialKernel(-2)

        p = kernel.PolynomialKernel(2)
        self.assertEqual(p.compute(np.array((1, 0, 2)).reshape(1,-1), 
                                   np.array((-1, 2, 5)).reshape(1,-1)), 100)
        self.assertAlmostEqual(p.compute(
                    np.array([1.2, -0.4, -2]).reshape(1,-1), 
                    np.array([4, 1.2, .5]).reshape(1,-1)), 18.6624)

        p = kernel.PolynomialKernel(5)
        self.assertEqual(p.compute(np.array((1, 0, 2)).reshape(1,-1), 
                                   np.array([-1, 2, 5]).reshape(1,-1)), 10 ** 5)
        self.assertAlmostEqual(p.compute(
                    np.array((1.2, -0.4, -2)).reshape(1,-1), 
                    np.array((4, 1.2, .5)).reshape(1,-1)),
                1504.59195, delta=10**-6)

        with self.assertRaises(ValueError):
            p.compute(np.array((1, 0, 2)).reshape(1,-1), 
                      np.array((-1, 2)).reshape(1,-1))



class TestHomogeneousPolynomialKernel(unittest.TestCase):
    def test_compute(self):
        with self.assertRaises(ValueError):
            kernel.HomogeneousPolynomialKernel(3.2)

        with self.assertRaises(ValueError):
            kernel.HomogeneousPolynomialKernel(-2)

        h = kernel.HomogeneousPolynomialKernel(2)
        self.assertEqual(h.compute(np.array((1, 0, 2)).reshape(1,-1), 
                                   np.array((-1, 2, 5)).reshape(1,-1)), 81.0)
        self.assertAlmostEqual(h.compute(
                    np.array([1.2, -0.4, -2]).reshape(1,-1), 
                    np.array([4, 1.2, .5]).reshape(1,-1))[0], 11.0224)

        h = kernel.HomogeneousPolynomialKernel(5)
        self.assertEqual(h.compute(
                    np.array((1, 0, 2)).reshape(1,-1), 
                    np.array([-1, 2, 5]).reshape(1,-1)) , 59049.0)
        self.assertAlmostEqual(h.compute(
                    np.array((1.2, -0.4, -2)).reshape(1,-1), 
                    np.array((4, 1.2, .5)).reshape(1,-1)),
                403.357761, delta=10**-6)

        with self.assertRaises(ValueError):
            h.compute(np.array((1, 0, 2)).reshape(1,-1), 
                      np.array((-1, 2)).reshape(1,-1))


class TestGaussianKernel(unittest.TestCase):
    def test_compute(self):
        with self.assertRaises(ValueError):
            kernel.GaussianKernel(-5)

        k = kernel.GaussianKernel(1)
        self.assertAlmostEqual(k.compute(np.array((1, 0, 1)).reshape(1,-1), 
                                         np.array((0, 0, 1)).reshape(1,-1))[0],
                               0.60653065)
        self.assertAlmostEqual(k.compute(
                    np.array([-3, 1, 0.5]).reshape(1,-1), 
                    np.array([1, 1.2, -8]).reshape(1,-1))[0], 6.73e-20)
        self.assertAlmostEqual(k.compute(
                    np.array([-1, -4, 3.5]).reshape(1,-1), 
                    np.array((1, 3.2, 6)).reshape(1,-1))[0], 3.29e-14)

        with self.assertRaises(ValueError):
            k.compute(np.array([-1, 3.5]).reshape(1,-1), 
                      np.array((1, 3.2, 6)).reshape(1,-1))


class TestHyperbolicKernel(unittest.TestCase):
    def test_compute(self):
        k = kernel.HyperbolicKernel(1, 5)
        self.assertAlmostEqual(k.compute(
                    np.array((1, 0, 1)).reshape(1,-1), 
                    np.array((0, 0, 1)).reshape(1,-1))[0], 0.9999877)
        self.assertAlmostEqual(k.compute(
                    np.array([-3, 1, 0.5]).reshape(1,-1), 
                    np.array([1, 1.2, -8]).reshape(1,-1))[0],
                -0.6640367, delta=10**-7)
        self.assertAlmostEqual(k.compute(
                    np.array([-1, -4, 3.5]).reshape(1,-1), 
                    np.array((1, 3.2, 6)).reshape(1,-1))[0],
                0.9999999, delta=10**-7)

        with self.assertRaises(ValueError):
            k.compute(np.array([-1, 3.5]).reshape(1,-1), 
                      np.array((1, 3.2, 6)).reshape(1,-1))


class TestPrecomputedKernel(unittest.TestCase):
    def test_compute(self):
        with self.assertRaises(ValueError):
            kernel.PrecomputedKernel(np.array(((1, 2), (3, 4, 5))))

        k = kernel.PrecomputedKernel(np.array(((1, 2), (3, 4))))
        self.assertEqual(k.compute(
                    np.array([1]).reshape(1,-1),
                    np.array([1]).reshape(1,-1)), [4.0])
        self.assertEqual(k.compute(
                    np.array([1]).reshape(1,-1),
                    np.array([0]).reshape(1,-1)), [3.0])

        with self.assertRaises(IndexError):
            k.compute(np.array([1]).reshape(1,-1), np.array([2]).reshape(1,-1))

        with self.assertRaises(IndexError):
            k.compute(np.array([0]).reshape(1,-1),
                      np.array([1.6]).reshape(1,-1))


if __name__ == '__main__':
    unittest.main()
