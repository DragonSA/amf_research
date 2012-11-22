"""
Graph comparing different call times.
"""
import matplotlib.pyplot as plt
import numpy as np

from convertible_bond import dS_total as dS, payoff, C, T
from model import FDEModel
from plot import plotmain

def main():
    S = np.linspace(0, 200, 200 * 8 + 1)
    Sl = 0
    Su = 250
    N = 128 * T
    K = 8 * (Su - Sl)
    Sk = (K * (S - Sl) / (Su - Sl)).astype(int)
    legend = []
    label = "$\\Omega^c = \\{%i\\}$"

    model = FDEModel(N, dS, payoff)
    fig = plt.figure()

    ax = fig.add_subplot(111)
    for i in range(1, 5):
        C.times = [i]
        ax.plot(S, model.price(Sl, Su, K).V[0][Sk])
        legend.append(label % i)
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(legend)


if __name__ == "__main__":
    plotmain(main)
