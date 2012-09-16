"""
Comparative graph between MK12 and this model.
"""
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
numpy.seterr(divide="ignore")

from convertible_bond import dS_total as dS, payoff, B
from model import BinomialModel, FDEModel

def main():
    N = 250
    S = range(121)
    P = [[], []]
    dS.sigma = 0.25
    dS.lambd_ = 0.062
    dSv = copy.copy(dS)
    dSv.lambd_ = lambda S: 0.062 * (S / 50)**-0.5
    dSv.cap_lambda = True
    B.R = 0.4
    model1 = BinomialModel(N, dS, payoff)
    model2 = BinomialModel(N, dSv, payoff)
    model3 = FDEModel(N, dS, payoff)
    for s in S:
        P[0].append(float(model1.price(s)))
        P[1].append(float(model2.price(s)))
    plt.plot(S, P[0])
    plt.plot(S, P[1])
    plt.plot(S, model3.price(0, 120, 120).V[0])
    plt.ylim([40, 160])
    plt.xlabel("Share price")
    plt.ylabel("Convertible Bond Price")
    plt.title("Convertible Bond Price Profile, N = %i, R = %i%%" % (N, B.R * 100))
    plt.legend(["Constant Hazard Rate", "Synthesis Hazard Rate", "FDE"])
    plt.savefig("../common/fig_mk12.svg")
    #plt.show()

if __name__ == "__main__":
    main()
