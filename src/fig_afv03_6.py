"""
Comparative graph between AFV03 figure 6 and this model.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from convertible_bond.afv03 import dS_total as dS, dS_var12, dS_var20, payoff, T
from model import FDEModel

def delta(S, model):
    Sl = 1
    Su = 200
    K = 8
    S = S - 1
    V = model.price(Sl, Su, 199 * K).V[0]
    return K * (V[S * K + 1] - V[S * K - 1]) / 2

def main():
    N = 128 * T
    S = np.arange(24, 121)
    model1 = FDEModel(N, dS, payoff)
    model2 = FDEModel(N, dS_var12, payoff)
    model3 = FDEModel(N, dS_var20, payoff)
    plt.plot(S, delta(S, model3))
    plt.plot(S, delta(S, model2))
    plt.plot(S, delta(S, model1))
    plt.xlim(S[0], S[-1])
    plt.ylim(-1, 1)
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(["$\\alpha = -2.0$", "$\\alpha = -1.2$", "Constant hazard rate"], loc=4)
    plt.savefig("../common/fig_afv03_6.pdf")
    #plt.show()

if __name__ == "__main__":
    main()
