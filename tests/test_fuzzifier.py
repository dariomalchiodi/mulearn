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
        self.X = X[:, 2]
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
             self.assertAlmostEqual(fuzzifier.__getattribute__(a), v, places=5)
    
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
        slopes = (-0.7926829266641029, -0.9984776805603038)
        intercepts = (3.27439024332436, 2.9373025049777697)
        
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
        squared_R = np.array([1, 3, 6, 12, 34, 37])
        mu = np.array([1, .9, .8, .7, .2, .1])
        target = np.array([0.84089642, 0.59460356, 0.35355339, 0.125     , 0.00276214,
       0.00164238])
        self._test_compute(fuzz.ExponentialFuzzifier(profile='fixed'),
                           squared_R, mu, 4, target)
        target = np.array([1.        , 0.93790848, 0.80971391,
                           0.60349527, 0.20539768, 0.17732365])
        self._test_compute(fuzz.ExponentialFuzzifier(profile='infer'),
                           squared_R, mu, 4, target)

        target = [
            np.array([1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                      6.99944998e-01, 4.16624055e-05, 1.10551439e-05]),
            np.array([1.00000000e+00, 1.00000000e+00, 7.54962443e-01,
                      8.24960386e-02, 2.46014950e-05, 8.13233710e-06]),
            np.array([1.00000000e+00, 8.81800354e-01, 4.08231664e-01,
                      8.74942523e-02, 3.08493908e-04, 1.42818021e-04]),
            np.array([1., 0.89834481, 0.62190419,
                      0.29804671, 0.02009127, 0.01390874]),
            np.array([1., 0.90433429, 0.74223711,
                      0.5, 0.11745466, 0.09640153]),
            np.array([1., 0.97403053, 0.89765262,
                      0.76239452, 0.41889818, 0.38605058]),
            np.array([1., 0.96809506, 0.92213607,
                      0.83666002, 0.58566202, 0.55785852]),
            np.array([0.99345844, 0.98050343, 0.96138697,
                      0.92426491, 0.8       , 0.78440274]),
            np.array([0.9970365 , 0.99113582, 0.9823502 ,
                      0.96501192, 0.9040156 , 0.89600224])
        ]

        for alpha, t in zip(np.linspace(0.1, 0.9, 9), target):
            self._test_compute(fuzz.ExponentialFuzzifier(profile=alpha),
                               squared_R, mu, 4, t)

    def test_profile(self):

        slopes = (-98.52763209367015,-14.305178809577578,
                  -1.142022483767279, -0.32724662996583)
        intercepts = (344.1535651472856, 27.17983975471548,
                      2.1513025936983423, 0.5563191029720798)
        
        for profile, slope, intercept in zip(['fixed', 'infer', 0.1, 0.4],
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
