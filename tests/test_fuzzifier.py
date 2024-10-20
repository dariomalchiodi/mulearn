import json
import numpy as np
import os
import pickle
import unittest

from sklearn.datasets import load_iris

from mulearn import FuzzyInductor
import mulearn.fuzzifier as fuzz


class BaseTest:
    def setUp(self):
        d = load_iris()
        X = d['data']
        self.X = 4 - X[:, 1] # otherwise membership would increase with distance
        self.y = np.array([1] * 50 + [0] * 100)
    
    def _test_compute(self, fuzzifier, squared_R, mu, squared_radius, target):
        fuzzifier.fit(squared_R, mu, squared_radius)
        if isinstance(fuzzifier, fuzz.CrispFuzzifier):
            self.assertTrue(set(fuzzifier.get_membership(squared_R)) <= {0, 1})
        else:
            self.assertIsInstance(fuzzifier.slope_, (int, float))
            self.assertIsInstance(fuzzifier.intercept_, (int, float))
            self.assertTrue(fuzzifier.slope_ < 0)
            self.assertTrue(fuzzifier.intercept_ > 0)

        result = fuzzifier.get_membership(squared_R)
        for t, r in zip(target, result):
            self.assertAlmostEqual(t, r)

    def _test_profile(self, fuzzifier, attributes, values):
         fuzzifier.fit(self.X, self.y, 3.5)
         for a, v in zip(attributes, values):
             self.assertAlmostEqual(fuzzifier.__getattribute__(a), v)
    
         _, _ ,_ = fuzzifier.get_profile(self.X)
    
    def _test_serialize(self, fuzzifier):
        fuzzifier.fit(self.X, self.y, 3.5)
        s = pickle.dumps(fuzzifier)  
        fuzzifier_clone = pickle.loads(s)
        self.assertEqual(fuzzifier, fuzzifier_clone)

    def _test_persist(self, fuzzifier):
        fuzzifier.fit(self.X, self.y, 3.5)

        with open('object.pickle', 'wb') as f:
            pickle.dump(fuzzifier, f)

        with open('object.pickle', 'rb') as f:
            fuzzifier_clone = pickle.load(f)

        os.remove('object.pickle')

        self.assertEqual(fuzzifier, fuzzifier_clone)
    
    def _test_json(self, fuzzifier, target):
        s = json.dumps(fuzzifier)
        repr = json.loads(s)
        self.assertEqual(repr, target)


class TestCrispFuzzifier(BaseTest, unittest.TestCase):

    def test_compute(self):
        squared_R = np.array([1, 3, 6, 12, 34, 416])
        mu = np.array([1, .9, .5, .3, .2, .1])

        self._test_compute(fuzz.CrispFuzzifier(profile='fixed'),
                           squared_R, mu, 2, np.array([1, 0, 0, 0, 0, 0]))
        
        self._test_compute(fuzz.CrispFuzzifier(profile='infer'),
                           squared_R, mu, 2, np.array([0, 0, 0, 0, 0, 0]))

    def test_profile(self):
        for profile, threshold in zip(['fixed', 'infer'], (3.5, 1)):
            self._test_profile(fuzz.CrispFuzzifier(profile=profile),
                               ['threshold_'], [threshold])
    
    def test_serialize(self):
        for profile in ['fixed', 'infer']:
            self._test_serialize(fuzz.CrispFuzzifier(profile=profile))

    def test_persist(self):
        for profile in ['fixed', 'infer']:
            self._test_persist(fuzz.CrispFuzzifier(profile=profile))
    
    def test_json(self):
        for profile in ['fixed', 'infer']:
            self._test_json(fuzz.CrispFuzzifier(profile=profile),
                        {'class': 'CrispFuzzifier', 'profile': profile})


class TestLinearFuzzifier(BaseTest, unittest.TestCase):

    def test_compute(self):
        squared_R = np.array([1, 3, 6, 12, 34, 416])
        mu = np.array([1, .9, .5, .3, .2, .1])
        self._test_compute(fuzz.LinearFuzzifier(profile='fixed'),
                           squared_R, mu, 2, np.array([1, 0, 0, 0, 0, 0]))
        target = [0.99763019, 0.99289056, 0.98578113,
                  0.97156225, 0.91942638, 0.01415807]
        self._test_compute(fuzz.LinearFuzzifier(profile='triangular'),
                           squared_R, mu, 2, target)
        target = np.array([0.60031447, 0.59773126, 0.59385645,
                           0.58610684, 0.55769158, 0.06429942])
        self._test_compute(fuzz.LinearFuzzifier(profile='infer'),
                           squared_R, mu, 2, target)

    def test_profile(self):
        slopes = (-0.14285714285714302, -1.1665378077030684)
        intercepts = (1.0000000000000007, 1.380763878264059)
        for profile, slope, intercept in zip(['fixed', 'infer'],
                                slopes, intercepts):

            self._test_profile(fuzz.LinearFuzzifier(profile=profile),
                               ['slope_', 'intercept_'],
                               [slope, intercept])
    
    def test_serialize(self):
        for profile in ['fixed', 'infer']:
            self._test_serialize(fuzz.LinearFuzzifier(profile=profile))

    def test_persist(self):
        for profile in ['fixed', 'infer']:
            self._test_persist(fuzz.LinearFuzzifier(profile=profile))
    
    def test_json(self):
        for profile in ['fixed', 'infer']:
            self._test_json(fuzz.LinearFuzzifier(profile=profile),
                        {'class': 'LinearFuzzifier', 'profile': profile})

        
class TestExponentialFuzzifier(BaseTest, unittest.TestCase):

    def test_compute(self):
        squared_R = np.array([1, 3, 6, 12, 34, 416])
        mu = np.array([1, .9, .5, .3, .2, .1])
        target = np.array([1.00000000e+00, 7.79143601e-01, 4.00540331e-01,
                           1.05853139e-01, 8.04602165e-04, 1.28797649e-40])
        self._test_compute(fuzz.ExponentialFuzzifier(profile='fixed'),
                           squared_R, mu, 5, target)
        target = np.array([1.00000000e+00, 8.49986669e-01, 5.87419583e-01,
                           2.80556951e-01, 1.86761467e-02, 6.90097154e-23])
        self._test_compute(fuzz.ExponentialFuzzifier(profile='infer'),
                           squared_R, mu, 5, target)

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

        for alpha, t in zip(np.linspace(0.1, 0.9, 9), target):
            self._test_compute(fuzz.ExponentialFuzzifier(profile=alpha),
                               squared_R, mu, 5, t)

    def test_profile(self):

        slopes = (-0.19804205158856666, -3.3844465544605287,
                  -46.051701859880865, -1.7001724434999912)
        intercepts = (3.7978835933515624e-14, 1.7098945314973162,
                      27.631021115928526, 0.7838817116258364)
        

        for profile, slope, intercept in zip(['fixed', 'infer', 0.01, 0.4],
                                slopes, intercepts):

            self._test_profile(fuzz.ExponentialFuzzifier(profile=profile),
                               ['slope_', 'intercept_'],
                               [slope, intercept])

    def test_serialize(self):
        for profile in ['fixed', 'infer', 0.01, 0.4]:
            self._test_serialize(fuzz.ExponentialFuzzifier(profile=profile))

    def test_persist(self):
        for profile in ['fixed', 'infer', 0.01, 0.4]:
            self._test_persist(fuzz.ExponentialFuzzifier(profile=profile))
    
    def test_json(self):
        for profile in ['fixed', 'infer', 0.01, 0.4]:
            self._test_json(fuzz.ExponentialFuzzifier(profile=profile),
                        {'class': 'ExponentialFuzzifier', 'profile': profile})


if __name__ == '__main__':
    unittest.main()
