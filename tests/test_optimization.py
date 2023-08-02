from unittest import TestCase

from kernel.optimization import *

class TestSolver(TestCase):
    
    def testCompute(self):

        s = Solver()

        xs = None
        mus = None
        k = None
        
        c = -1
        
        with self.assertRaises(ValueError):
            s.solve(xs, mus, c, k)


class TestGurobiSolver(TestCase):

    def testCompute(self):

        s = GurobiSolver()

        xs = np.array([[0.2526861], [0.77908776], [0.5120937], [0.52646533], [0.01438627]])

        mus = np.array([0.63883086, 0.56515446, 0.99892903, 0.99488161, 0.17768801])

        c = 1

        #obtained from the original mulearn module
        chis_opt = [0.26334825774012194, 0.5651531004941153, -0.0010709737624955377, -0.00511839469274798, 0.17768801022078584]
        
        k = np.array([[1.        , 0.87062028, 0.96691358, 0.96321606, 0.9720059 ],
                      [0.87062028, 1.        , 0.96498481, 0.96859468, 0.74648169],
                      [0.96691358, 0.96498481, 1.        , 0.99989673, 0.88350675],
                      [0.96321606, 0.96859468, 0.99989673, 1.        , 0.87711911],
                      [0.9720059 , 0.74648169, 0.88350675, 0.87711911, 1.        ]])

        for chi, chi_opt in zip(s.solve(xs, mus, c, k), chis_opt):
            self.assertAlmostEqual(chi, chi_opt, places=5)




        


        