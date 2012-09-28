"""
Comparative graph between AFV03 figure 2 and this model.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from convertible_bond.afv03 import dS, dS_partial, dS_total, payoff, T
from model import FDEModel

def main():
    S = np.arange(80, 121)
    Sl = 0
    Su = 200
    N = 128 * T
    K = 8 * (Su - Sl)
    Sk = K * (S - Sl) / (Su - Sl)
    model1 = FDEModel(N, dS, payoff)
    model2 = FDEModel(N, dS_partial, payoff)
    model3 = FDEModel(N, dS_total, payoff)
    plt.plot(S, model1.price(Sl, Su, K).V[0][Sk])
    plt.plot(S, model2.price(Sl, Su, K).V[0][Sk])
    plt.plot(S, model3.price(Sl, Su, K).V[0][Sk])
    plt.ylim(100, 150)
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(["No default", "Partial default", "Total Default"], loc=2)
    plt.savefig("../common/fig_afv03_2.png")
    plt.savefig("../common/fig_afv03_2.svg")
    #plt.show()

if __name__ == "__main__":
    main()
