"""
Comparative graph between MK12 and this model.
"""
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
np.seterr(divide="ignore")

from convertible_bond import dS_total as dS, payoff, B, T
from model import FDEModel

def main():
    S = np.linspace(0, 120, 120 * 8 + 1)
    Sl = 0
    Su = 200
    N = 128 * T
    K = 8 * (Su - Sl)
    Sk = (K * (S - Sl) / (Su - Sl)).astype(int)
    dS.sigma = 0.25
    dS.lambd_ = 0.062
    dSv = copy.copy(dS)
    dSv.lambd_ = lambda S: 0.062 * (S / 50)**-0.5
    dSv.cap_lambda = True
    B.R = 0.4
    model1 = FDEModel(N, dS, payoff)
    model2 = FDEModel(N, dSv, payoff)
    plt.plot(S, model1.price(Sl, Su, K).V[0][Sk])
    plt.plot(S, np.append(B.R * B.N, model2.price(Sl + 1, Su, K - 8).V[0][Sk[:-1]]))
    plt.ylim([40, 160])
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(["Constant $\\lambda$", "Synthesis $\\lambda$"], loc=2)
    plt.savefig("../common/fig_mk12.png")
    plt.savefig("../common/fig_mk12.svg")
    #plt.show()

if __name__ == "__main__":
    main()
