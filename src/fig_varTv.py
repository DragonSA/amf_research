"""
Graph comparing different conversion time and price surface for \Tv = {3, 5}.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from convertible_bond import dS_total as dS, payoff, S as Conv, T
from model import FDEModel
from plot import plot_model

def main():
    S = np.linspace(0, 200, 200 * 8 + 1)
    Sl = 0
    Su = 250
    N = 128 * T
    K = 8 * (Su - Sl)
    Sk = (K * (S - Sl) / (Su - Sl)).astype(int)
    legend = []
    label = "$\\Omega^v = [%i, 5]$"

    model = FDEModel(N, dS, payoff)
    fig = plt.figure()
    fig.set_figwidth(1.8 * fig.get_figwidth())

    ax = fig.add_subplot(1, 2, 1)
    for i in range(1, 5):
        Conv.times = [(i, 5)]
        ax.plot(S, model.price(Sl, Su, K).V[0][Sk])
        legend.append(label % i)
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(legend)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    plot_model(ax, dS, payoff)
    plt.savefig("../common/fig_varTv.pdf")
    #plt.show()


if __name__ == "__main__":
    main()
