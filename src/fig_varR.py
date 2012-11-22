"""
Graph comparing different R and price surface for zero-redemption.
"""
import matplotlib.pyplot as plt
import numpy as np

from convertible_bond import dS_total as dS, payoff, A, T
from model import FDEModel
from plot import plot_model, plotmain

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
    A.N = -A.C
    ax.plot(S, model.price(Sl, Su, K).V[0][Sk])
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(["$R = 104$", "$R = 0$"])

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    plot_model(ax, dS, payoff)


if __name__ == "__main__":
    plotmain(main)
