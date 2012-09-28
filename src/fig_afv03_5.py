"""
Comparative graph between AFV03 figure 5 and this model.
"""
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from convertible_bond.afv03  import dS_total as dS, payoff, T
from model import FDEModel

def main():
    S = np.arange(24, 121)
    Sl = 0
    Su = 200
    N = 128 * T
    K = 8 * (Su - Sl)
    Sk = K * (S - Sl) / (Su - Sl)
    dS1 = copy.copy(dS)
    dS1.lambd_ = lambda S: 0.02 * (S / 100)**-1.2
    dS1.cap_lambda = True
    dS2 = copy.copy(dS)
    dS2.lambd_ = lambda S: 0.02 * (S / 100)**-2.0
    dS2.cap_lambda = True
    model1 = FDEModel(N, dS, payoff)
    model2 = FDEModel(N, dS1, payoff)
    model3 = FDEModel(N, dS2, payoff)
    plt.plot(S, model1.price(Sl, Su, K).V[0][Sk])
    plt.plot(S, model2.price(Sl + 1, Su, K - 8).V[0][Sk - 8])
    plt.plot(S, model3.price(Sl + 1, Su, K - 8).V[0][Sk - 8])
    plt.xlim(S[0], S[-1])
    plt.ylim(50, 150)
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(["Constant hazard rate", "$\\alpha = -1.2$", "$\\alpha = -2.0$"], loc=2)
    plt.savefig("../common/fig_afv03_5.png")
    plt.savefig("../common/fig_afv03_5.svg")
    #plt.show()

if __name__ == "__main__":
    main()
