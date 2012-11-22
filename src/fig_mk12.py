"""
Comparative graph between MK12 and this model.
"""
import matplotlib.pyplot as plt
import numpy as np
np.seterr(divide="ignore")

from convertible_bond.mk12 import dS, dS_var, payoff, A, T
from model import FDEModel
from plot import plotmain

def main():
    S = np.linspace(0, 120, 120 * 8 + 1)
    Sl = 0
    Su = 200
    N = 128 * T
    K = 8 * (Su - Sl)
    Sk = (K * (S - Sl) / (Su - Sl)).astype(int)
    model1 = FDEModel(N, dS, payoff)
    model2 = FDEModel(N, dS_var, payoff)
    plt.plot(S, model1.price(Sl, Su, K).V[0][Sk])
    plt.plot(S, np.append(A.R * A.N, model2.price(Sl + 1, Su, K - 8).V[0][Sk[:-1]]))
    plt.ylim([40, 160])
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(["Constant $\\lambda$", "Synthesis $\\lambda$"], loc=2)

if __name__ == "__main__":
    plotmain(main)
