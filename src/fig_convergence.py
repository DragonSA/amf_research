"""
Graph the convergence of the various models for pricing convertible bonds.
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import time

from convertible_bond import dS_total as dS, payoff, T
from model import BinomialModel, FDEModel, ImplicitScheme, CrankNicolsonScheme, \
                    PenaltyScheme
from plot import plotmain

def label(plt, xlabel):
    rng = np.arange(122.7, 123.5, 0.1)
    plt.set_ylim(rng[0], rng[-1])
    plt.set_yticks(rng, ["%.1f" for i in rng])
    plt.set_xlabel(xlabel)
    plt.set_ylabel("Convertible Bond Price")
    plt.legend(["Binomial", "FDE - Implicit", "FDE - Crank-Nicolson", "FDE - Penalty"],
               prop={"size": "small"})

def main():
    S0 = 100
    Sl = 0
    Su = 200
    X = range(11)
    N = lambda x: 2**(x + 1) * T
    K = lambda x: 2**((x + 1) // 2) * (Su - Sl)

    fig = plt.figure()
    fig.set_figwidth(1.8 * fig.get_figwidth())

    plt_x = fig.add_subplot(1, 2, 1)
    plt_t = fig.add_subplot(1, 2, 2)

    p = []
    t = []
    for x in X:
        start = time.time()
        p.append(float(BinomialModel(N(x), dS, payoff).price(S0)))
        t.append(time.time() - start)
    plt_x.plot(X, p)
    plt_t.plot(t, p)
    print p[-1]

    for scheme in (ImplicitScheme, CrankNicolsonScheme, PenaltyScheme):
        p = []
        t = []
        for x in X:
            k = K(x)
            Sk = k * (S0 - Sl) / (Su - Sl)
            start = time.time()
            p.append(FDEModel(N(x), dS, payoff).price(Sl, Su, k, scheme=scheme).V[0][Sk])
            t.append(time.time() - start)
        plt_x.plot(X, p[:len(X)])
        plt_t.plot(t, p)
        print p[-1]

    label(plt_x, "$\\delta_t = 2^{-(x + 1)}$; $\\delta_S = 2^{-\\frac{x + 1}{2}}$")
    label(plt_t, "Computational Time (seconds)")
    #plt_t.set_xlim(0, 2.5)
    plt_t.set_xscale('log')

if __name__ == "__main__":
    plotmain(main)
