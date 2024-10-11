
import numpy as np
import unittest

from mulearn import FuzzyInductor
import mulearn.fuzzifier as fuzz


class TestCrispFuzzifier(unittest.TestCase):
    def test_compute(self):

        X = np.array([[0.91232935],
                      [0.49196389],
                      [0.01257953],
                      [0.65664674],
                      [0.44446359],
                      [0.4211273 ],
                      [0.72152853],
                      [0.05744546],
                      [0.30902402],
                      [0.52455777],
                      [0.39387502],
                      [0.59580251],
                      [0.24884305],
                      [0.95579843],
                      [0.23678875]])
        
        mus = np.array(
            [0.00836525, 0.99818461, 0.00124992, 0.5013639 , 0.91687722,
             0.83942725, 0.25137643, 0.0040433 , 0.35836777, 0.98317438,
             0.72841124, 0.77240852, 0.16950794, 0.00289302, 0.14237189])

        f = fuzz.CrispFuzzifier()
        
        m = FuzzyInductor(fuzzifier=f).fit(X, mus)
        result = m.predict(X)

        self.assertEqual(result, [0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1])

        f = fuzz.CrispFuzzifier(profile='infer')
        
        m = FuzzyInductor(fuzzifier=f).fit(X, mus)

        result = m.predict(X)

        self.assertEqual(result, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

class TestLinearFuzzifier(unittest.TestCase):
    def test_compute(self):

        X = np.array([[0.91232935],
                      [0.49196389],
                      [0.01257953],
                      [0.65664674],
                      [0.44446359],
                      [0.4211273 ],
                      [0.72152853],
                      [0.05744546],
                      [0.30902402],
                      [0.52455777],
                      [0.39387502],
                      [0.59580251],
                      [0.24884305],
                      [0.95579843],
                      [0.23678875]])
        
        mus = np.array(
            [0.00836525, 0.99818461, 0.00124992, 0.5013639 , 0.91687722,
             0.83942725, 0.25137643, 0.0040433 , 0.35836777, 0.98317438,
             0.72841124, 0.77240852, 0.16950794, 0.00289302, 0.14237189])

        f = fuzz.LinearFuzzifier()
        
        m = FuzzyInductor(fuzzifier=f).fit(X, mus)

        result = m.predict(X)
        
        correct = [0.19171390089298312,
                   0.9623537182595308,
                   0.06400343952613685,
                   0.7109833150411717,
                   0.9187589912747525,
                   0.880668602044669,
                   0.5789235960915939,
                   0.15236435530657688,
                   0.6623815130195112,
                   0.9455734796766168,
                   0.8308184495157517,
                   0.8308184492677784,
                   0.5394812027285296,
                   0.10554521718033616,
                   0.5148178265851036]
        
        for chi, chi_opt in zip(result, correct):
            self.assertAlmostEqual(float(chi), chi_opt, places=5)

        f = fuzz.LinearFuzzifier(profile='infer')
        
        m = FuzzyInductor(fuzzifier=f).fit(X, mus)

        result = m.predict(X)

        correct = [0.0,
                   1.0,
                   0.0,
                   0.5062218217097789,
                   0.9178584632914968,
                   0.8423953475724786,
                   0.24459052693051053,
                   0.0,
                   0.4099339264100187,
                   0.9709822271100618,
                   0.7436342714864057,
                   0.7436342709951314,
                   0.16644887611636805,
                   0.0,
                   0.11758680761161522]
        
        for chi, chi_opt in zip(result, correct):
            self.assertAlmostEqual(chi, chi_opt, places=5)


if __name__ == '__main__':
    unittest.main()
