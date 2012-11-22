"""
Comparative graph between AFV03 figure 4 and this model.
"""
import copy

import matplotlib.pyplot as plt
import numpy as np

from convertible_bond.afv03 import dS, dS_total, payoff, A, T
from model import FDEModel
from plot import plotmain

def main():
    S = np.arange(80, 121)
    Sl = 0
    Su = 200
    N = 128 * T
    K = 8 * (Su - Sl)
    Sk = K * (S - Sl) / (Su - Sl)
    model1 = FDEModel(N, dS_total, payoff)
    model2 = FDEModel(N, dS, payoff)
    A.R = 1.0
    plt.plot(S, model1.price(Sl, Su, K).V[0][Sk])
    plt.plot(S, model2.price(Sl, Su, K).V[0][Sk])
    A.R = 0.5
    plt.plot(S, model1.price(Sl, Su, K).V[0][Sk])
    A.R = 0.0
    plt.plot(S, model1.price(Sl, Su, K).V[0][Sk])
    plt.ylim(100, 150)
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(["Total default (R=100%)", "No default", "Total default (R=50%)", "Total default (R=0%)"], loc=2)

if __name__ == "__main__":
    plotmain(main)
