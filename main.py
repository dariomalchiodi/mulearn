
import os
from mulearn import FuzzyInductor
from mulearn.fuzzifier import ExponentialFuzzifier
import matplotlib.pyplot as plt

if __name__ == '__main__':
    os.environ['GRB_LICENSE_FILE'] = '/home/malchiodi/.gurobi/gurobi.lic'
    f = FuzzyInductor(fuzzifier=ExponentialFuzzifier(profile="infer"))
    X, y = [[1], [2], [3], [4]], [1, 1, 0, 0]
    f.fit(X, y)
    [rdata, rdata_synth, estimate] = f.fuzzifier.get_profile(X)

    plt.plot(rdata_synth, estimate)
    plt.plot(rdata, y, "o")
    plt.show()
