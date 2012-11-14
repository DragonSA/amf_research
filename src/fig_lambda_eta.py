"""
Graph showing the pricing for different lambda and eta values.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from convertible_bond import dS, payoff, T
from model import FDEModel

def main():
    Sl = 0
    Su = 200
    N = 128 * T
    K = 8 * (Su - Sl)
    S = 100
    Sk = K * (S - Sl) / (Su - Sl)
    ETA = np.linspace(0, 1, 65)
    name = ["No default"]
    fde = FDEModel(N, dS, payoff)
    plt.plot(ETA, [fde.price(Sl, Su, K).V[0][Sk]] * len(ETA))
    for lambd_ in (0.01, 0.02, 0.03):
        dS.lambd_ = lambd_
        V = []
        for eta in ETA:
            dS.eta = eta
            V.append(fde.price(Sl, Su, K).V[0][Sk])
        plt.plot(ETA, V)
        name.append("$\\lambda = %i\\%%$" % (lambd_ * 100))
    plt.xlabel("$\\eta$")
    plt.ylabel("Price at $S=100$")
    plt.legend(name, loc=3)
    plt.savefig("../common/fig_lambda_eta.pdf")
    #plt.show()

if __name__ == "__main__":
    main()
