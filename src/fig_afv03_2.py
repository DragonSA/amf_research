"""
Comparative graph between AFV03 figure 2 and this model.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from convertible_bond import dS, dS_partial, dS_total, payoff
from model import FDEModel

def main():
    N = 200
    S = range(80, 121)
    Sl = 0
    Su = 200
    model1 = FDEModel(N, dS, payoff)
    model2 = FDEModel(N, dS_partial, payoff)
    model3 = FDEModel(N, dS_total, payoff)
    plt.plot(S, model1.price(Sl, Su, N).V[0][S])
    plt.plot(S, model2.price(Sl, Su, N).V[0][S])
    plt.plot(S, model3.price(Sl, Su, N).V[0][S])
    plt.ylim([100, 150])
    plt.xlabel("Stock Price")
    plt.ylabel("Convertible Bond Price")
    plt.legend(["No default", "Partial default", "Total Default"], loc=2)
    plt.savefig("../common/fig_afv03_2.png")
    plt.savefig("../common/fig_afv03_2.svg")
    #plt.show()

if __name__ == "__main__":
    main()
