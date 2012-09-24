"""
Comparative graph between AFV03 figure 7 and this model.
"""
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from convertible_bond import dS_total as dS, payoff, B
from model import FDEModel, RannacherScheme

def gamma(S, model):
    Sl = 1
    Su = 201
    K = 32
    S = S - 1
    V = model.price(Sl, Su, 200 * K, scheme=RannacherScheme).V[0]
    return K * K * (V[S * K + 1] + V[S * K - 1] - 2 * V[S * K])

def main():
    N = 640
    S = np.arange(20, 121)
    dS1 = copy.copy(dS)
    dS1.lambd_ = lambda S: 0.02 * (S / 100)**-1.2
    dS1.cap_lambda = True
    dS2 = copy.copy(dS)
    dS2.lambd_ = lambda S: 0.02 * (S / 100)**-2.0
    dS2.cap_lambda = True
    model1 = FDEModel(N, dS, payoff)
    model2 = FDEModel(N, dS1, payoff)
    model3 = FDEModel(N, dS2, payoff)
    plt.plot(S, gamma(S, model1))
    plt.plot(S, gamma(S, model2))
    plt.plot(S, gamma(S, model3))
    plt.ylim([-0.1, 0.1])
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(["Constant hazard rate", "$\\alpha = -1.2$", "$\\alpha = -2.0$"], loc=2)
    plt.savefig("../common/fig_afv03_7.png")
    plt.savefig("../common/fig_afv03_7.svg")
    #plt.show()

if __name__ == "__main__":
    main()
