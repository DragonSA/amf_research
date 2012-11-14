"""
Comparative graph between AFV03 figure 7 and this model.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from convertible_bond.afv03 import dS_total as dS, dS_var12, dS_var20, payoff, T
from model import FDEModel

def gamma(S, model):
    Sl = 1
    Su = 200
    K = 8
    S = S - 1
    V = model.price(Sl, Su, 199 * K).V[0]
    return K * K * (V[S * K + 1] + V[S * K - 1] - 2 * V[S * K])

def main():
    N = 128 * T
    S = np.arange(24, 121)
    model1 = FDEModel(N, dS, payoff)
    model2 = FDEModel(N, dS_var12, payoff)
    model3 = FDEModel(N, dS_var20, payoff)
    plt.plot(S, gamma(S, model1))
    plt.plot(S, gamma(S, model2))
    plt.plot(S, gamma(S, model3))
    plt.xlim(S[0], S[-1])
    plt.ylim(-0.1, 0.1)
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(["Constant hazard rate", "$\\alpha = -1.2$", "$\\alpha = -2.0$"], loc=2)
    plt.savefig("../common/fig_afv03_7.pdf")
    #plt.show()

if __name__ == "__main__":
    main()
