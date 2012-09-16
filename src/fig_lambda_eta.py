"""
Graph showing the pricing for different lambda and eta values.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from convertible_bond import dS, payoff
from model import BinomialModel, FDEModel
from payoff import CallA

def label(plt, model, name):
    plt.set_title("American Call ($K=80$) using\n%s pricing model" % model)
    plt.set_xlabel("$\\eta$")
    plt.set_ylabel("Price at $S=100$")
    plt.legend(name, loc=2)

def main():
    N = 200
    Z = 100
    S = 100
    ETA = np.linspace(0, 1, 100)
    name = []
    payoff = CallA(5, 80)
    binom = BinomialModel(N, dS, payoff)
    fde = FDEModel(N, dS, payoff)
    plt_binom = plt.subplot(1, 2, 1)
    plt_fde = plt.subplot(1, 2, 2, sharex = plt_binom, sharey = plt_binom)
    for lambd_ in (0.0, 0.1, 0.2):
        binomV = []
        fdeV = []
        for eta in ETA:
            dS.lambd_ = lambd_
            dS.eta = eta
            binomV.append(float(binom.price(S)))
            fdeV.append(fde.price(S - Z, S + Z, Z*2).V[0][Z])
        plt_binom.plot(ETA, binomV)
        plt_fde.plot(ETA, fdeV)
        name.append("$\\lambda = %f$" % (lambd_))
    label(plt_binom, "Binomial", name)
    label(plt_fde, "FDE", name)
    plt.savefig("../common/fig_lambda_eta.png")
    plt.savefig("../common/fig_lambda_eta.svg")
    #plt.show()

if __name__ == "__main__":
    main()
