"""
Plot the traded surface using the binomial surface,
"""
import matplotlib.pyplot as plt
import numpy as np

from convertible_bond import dS_typical as dS, payoff, A
from model import BinomialModel
from plot import plot_model, payoff as payoff_, HOLD, plotmain

def plot_H(ax, Sl, Su, model, P):
    po = model.dS.binomial(model.dt)[3][2]

    y = slice(0, 1)
    for x, t in zip(range(len(P.t) - 1), P.t[:-1]):
        S = P.S[x][y]
        V = P.V[x][y]
        y1 = slice(y.start, y.stop + 1)
        S1 = P.S[x + 1][y1]
        V1 = P.V[x + 1][y1]
        I1 = P.I[x + 1][y1]
        t1 = P.t[x + 1]

        delta = np.diff(V1) / np.diff(S1)
        pi = (V - delta * S - (A.C if t in A else 0)) * np.exp(model.dt * model.dS.r)

        Vu = pi + delta * S1[:-1]
        Vd = pi + delta * S1[1:]
        Xo  = pi + (1 - model.dS.eta) * delta * S

        H = -(Xo - P.X[x][y]) / (1 - po)
        np.testing.assert_array_almost_equal(H, (Vu - V1[:-1]) / po)
        np.testing.assert_array_almost_equal(H, (Vd - V1[1:]) / po)

        if x:
            flt_ = (Sl <= S_) & (S_ <= Su)
            flt = (Sl <= S) & (S <= Su)
            ax.plot_trisurf(np.append(t_[flt_], (t * np.ones(S.shape))[flt]),
                            np.append(S_[flt_], S[flt]),
                            np.append(H_[flt_], H[flt]), linewidth=0)

        start = 0
        stop = y1.stop - y1.start - 1
        while start <= stop and payoff_(t1, S1[start], V1[start], I1[start]) != HOLD:
            start += 1
        while start <= stop and payoff_(t1, S1[stop], V1[stop], I1[stop]) != HOLD:
            stop -= 1
        y = slice(y1.start + start, y1.start + stop + 1)
        t_ , H_, S_ = t * np.ones(S.shape), H, S


def main():
    N = 80
    S0 = 100
    Sl = 0
    Su = 200
    model = BinomialModel(N, dS, payoff)
    P = model.price(S0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_H(ax, Sl, Su, model, P)
    ax.set_ylim(Sl, Su)
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.set_zlabel("Default Cost after Hedging ($H^{c}_t$)")

if __name__ == "__main__":
    plotmain(main)