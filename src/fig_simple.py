"""
Graph comparing different call times and price surface for C.times = [2].
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from convertible_bond import dS_total as dS, payoff, C, T
from model import FDEModel
from plot import plot_model

def main():
    S = np.linspace(0, 200, 200 * 8 + 1)
    Sl = 0
    Su = 250
    N = 128 * T
    K = 8 * (Su - Sl)
    Sk = (K * (S - Sl) / (Su - Sl)).astype(int)

    model = FDEModel(N, dS, payoff)
    fig = plt.figure()
    fig.set_figwidth(1.8 * fig.get_figwidth())

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(S, model.price(Sl, Su, K).V[0][Sk])
    C.times = [2]
    ax.plot(S, model.price(Sl, Su, K).V[0][Sk])
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(["$\\Omega^c = [2, 5)$", "$\\Omega^c = \\{2\\}$"])

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    plot_model(ax, dS, payoff)
    plt.savefig("../common/fig_simple.pdf")
    #plt.show()


if __name__ == "__main__":
    main()
