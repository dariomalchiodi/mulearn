from unittest import TestCase

from fuzzifier import *
from __init__ import *




class TestCrispFuzzifier(TestCase):
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
        
        mus = np.array([0.00836525, 0.99818461, 0.00124992, 0.5013639 , 0.91687722,
                        0.83942725, 0.25137643, 0.0040433 , 0.35836777, 0.98317438,
                        0.72841124, 0.77240852, 0.16950794, 0.00289302, 0.14237189])

        f = CrispFuzzifier()
        
        m = FuzzyInductor(fuzzifier=f).fit(X, mus)

        result = list(m.fuzzifier.get_membership()(X))

        self.assertEqual(result, [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

        f = CrispFuzzifier(profile='infer')
        
        m = FuzzyInductor(fuzzifier=f).fit(X, mus)

        result = list(m.fuzzifier.get_membership()(X))

        self.assertEqual(result, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

class TestLinearFuzzifier(TestCase):
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
        
        mus = np.array([0.00836525, 0.99818461, 0.00124992, 0.5013639 , 0.91687722,
                        0.83942725, 0.25137643, 0.0040433 , 0.35836777, 0.98317438,
                        0.72841124, 0.77240852, 0.16950794, 0.00289302, 0.14237189])

        f = LinearFuzzifier()
        
        m = FuzzyInductor(fuzzifier=f).fit(X, mus)

        result = list(m.fuzzifier.get_membership()(X))
        
        correct = [0.0,
                   0.6235405260563452,
                   0.0,
                   0.0,
                   0.4751637193324021,
                   0.34552115570341846,
                   0.0,
                   0.0,
                   0.0,
                   0.5664281451951747,
                   0.17585370486588414,
                   0.17585370402191836,
                   0.0,
                   0.0,
                   0.0]
        
        for chi, chi_opt in zip(result, correct):
            self.assertAlmostEqual(chi, chi_opt, places=5)

        f = LinearFuzzifier(profile='infer')
        
        m = FuzzyInductor(fuzzifier=f).fit(X, mus)

        result = list(m.fuzzifier.get_membership()(X))

        correct = [0.0,
                   0.24161921015433518,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.1265658369160101,
                   0.0,
                   0.0,
                   0.0,
                   0.0,
                   0.0]
        
        for chi, chi_opt in zip(result, correct):
            self.assertAlmostEqual(chi, chi_opt, places=5)


        