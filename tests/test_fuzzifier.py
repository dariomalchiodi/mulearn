
import numpy as np
import unittest

from sklearn.datasets import load_iris

from mulearn import FuzzyInductor
import mulearn.fuzzifier as fuzz


class TestCrispFuzzifier(unittest.TestCase):
    def test_compute(self):

        squared_R = np.array([1, 3, 6, 12, 34, 416])
        mu = np.array([1, .9, .5, .3, .2, .1])

        f = fuzz.CrispFuzzifier(profile='fixed')

        f.fit(squared_R, mu, 2)
        self.assertTrue(set(f.get_membership(squared_R)) <= {0, 1})

        target = np.array([1, 0, 0, 0, 0, 0])
        result = f.get_membership(squared_R)
        self.assertTrue((result==target).all())

        f = fuzz.CrispFuzzifier(profile='infer')   
        f.fit(squared_R, mu, 2)
        self.assertTrue(set(f.get_membership(squared_R)) <= {0, 1})

        target = np.array([0, 0, 0, 0, 0, 0])
        result = f.get_membership(squared_R)
        self.assertTrue((result==target).all())

class TestLinearFuzzifier(unittest.TestCase):
    def test_compute(self):

        squared_R = np.array([1, 3, 6, 12, 34, 416])
        mu = np.array([1, .9, .5, .3, .2, .1])

        f = fuzz.LinearFuzzifier(profile='fixed')

        f.fit(squared_R, mu, 2)
        self.assertIsInstance(f.slope_, (int, float))
        self.assertIsInstance(f.intercept_, (int, float))
        self.assertTrue(f.slope_ < 0)

        target = np.array([1, 0, 0, 0, 0, 0])
        result = f.get_membership(squared_R)
        self.assertTrue((result==target).all())

        f = fuzz.LinearFuzzifier(profile='triangular')

        f.fit(squared_R, mu, 2)
        self.assertIsInstance(f.slope_, (int, float))
        self.assertIsInstance(f.intercept_, (int, float))
        self.assertTrue(f.slope_ < 0)

        target = np.array([0.99763019, 0.99289056, 0.98578113,
                           0.97156225, 0.91942638, 0.01415807])
        result = f.get_membership(squared_R)
        for t, r in zip(target, result):
            self.assertAlmostEqual(t, r)

        f = fuzz.LinearFuzzifier(profile='infer')

        f.fit(squared_R, mu, 2)
        self.assertIsInstance(f.slope_, (int, float))
        self.assertIsInstance(f.intercept_, (int, float))
        self.assertTrue(f.slope_ < 0)

        target = np.array([0.60031447, 0.59773126, 0.59385645,
                           0.58610684, 0.55769158, 0.06429942])
        result = f.get_membership(squared_R)
        for t, r in zip(target, result):
            self.assertAlmostEqual(t, r)

        
class TestExponentialFuzzifier(unittest.TestCase):
    def test_compute(self):

        squared_R = np.array([1, 3, 6, 12, 34, 416])
        mu = np.array([1, .9, .5, .3, .2, .1])

        f = fuzz.ExponentialFuzzifier(profile='fixed')

        f.fit(squared_R, mu, 5)
        self.assertIsInstance(f.slope_, (int, float))
        self.assertIsInstance(f.intercept_, (int, float))
        self.assertTrue(f.slope_ < 0)

        target = np.array([1.00000000e+00, 7.79143601e-01, 4.00540331e-01,
                           1.05853139e-01, 8.04602165e-04, 1.28797649e-40])
        result = f.get_membership(squared_R)
        for t, r in zip(target, result):
            self.assertAlmostEqual(t, r)

        f = fuzz.ExponentialFuzzifier(profile='infer')

        f.fit(squared_R, mu, 5)
        self.assertIsInstance(f.slope_, (int, float))
        self.assertIsInstance(f.intercept_, (int, float))
        self.assertTrue(f.slope_ < 0)

        target = np.array([1.00000000e+00, 8.49986669e-01, 5.87419583e-01,
                           2.80556951e-01, 1.86761467e-02, 6.90097154e-23])
        result = f.get_membership(squared_R)
        for t, r in zip(target, result):
            self.assertAlmostEqual(t, r)

        target = [
            np.array([1.00000000e+000, 1.00000000e+000, 4.96678058e-001,
                      2.37588306e-003, 7.38315600e-012, 1.39850564e-159]),
            np.array([1.00000000e+00, 1.00000000e+00, 4.81156517e-01,
                      1.11393302e-01, 5.21145506e-04, 1.82566559e-44]),
            np.array([1.00000000e+00, 8.87724624e-01, 4.09013407e-01,
                      8.68272604e-02, 2.95590722e-04, 4.14800012e-47]),
            np.array([1.00000000e+00, 8.42106726e-01, 6.00356674e-01,
                      3.05136216e-01, 2.55154525e-02, 4.94512909e-21]),
            np.array([9.91487538e-01, 8.35522108e-01, 6.46344377e-01,
                      3.86790709e-01, 5.88645164e-02, 3.74071500e-16]),
            np.array([9.58324529e-01, 8.80111737e-01, 7.74596670e-01,
                      6.00000000e-01, 2.35195248e-01, 2.03818374e-08]),
            np.array([0.98461202, 0.95454278, 0.91115192,
                      0.83019782, 0.59022078, 0.00157868]),
            np.array([0.99345844, 0.98050343, 0.96138697,
                      0.92426491, 0.8, 0.06520449]),
            np.array([0.99953184, 0.99859618, 0.99719433,
                      0.99439653, 0.98420493, 0.82299933])
        ]

        for t, alpha in enumerate(np.linspace(0.1, 0.9, 9)):
            f = fuzz.ExponentialFuzzifier(profile=alpha)

            f.fit(squared_R, mu, 5)
            self.assertIsInstance(f.slope_, (int, float))
            self.assertIsInstance(f.intercept_, (int, float))
            self.assertTrue(f.slope_ < 0)

            result = f.get_membership(squared_R)
            for t, r in zip(target[t], result):
                self.assertAlmostEqual(t, r)

class TestFuzzifierSerialization(unittest.TestCase):
    def test_dump(self):
        X, y = load_iris()

    def test_serialization(self):
        pass


if __name__ == '__main__':
    unittest.main()
