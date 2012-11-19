"""
Utility functions for plotting
"""
import matplotlib.cm as cm
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from convertible_bond import dS_total as dS, payoff, A, P, C, T
from model import FDEModel, CrankNicolsonScheme

colourise = cm.gist_ncar
PUT   = 0. / 6
CALL  = 1. / 6
CONV  = 2. / 6
FCONV = 3. / 6
REDEM = 4. / 6
HOLD  = 5. / 6

def choices(Z):
    colours = np.zeros((len(Z.S), len(Z.t)))
    for y, t in zip(range(len(Z.t)), Z.t):
        Kp = P.payoff.strike(t)
        Kc = C.payoff.strike(t)
        for x, S in zip(range(len(Z.S)), Z.S):
            V = Z.V[y][x]
            if t in A and t != T:
                V -= A.C

            if V == S:
                if t in C and t != T and Z.I[y][x] > S:
                    colours[x, y] = FCONV
                else:
                    colours[x, y] = CONV
            elif t == T:
                colours[x, y] = REDEM
            elif t in P and V == Kp:
                colours[x, y] = PUT
            elif t in C and V == Kc:
                colours[x, y] = CALL
            else:
                colours[x, y] = HOLD
    return colourise(colours)


def legend(ax):
    proxy = []
    descr = []
    def colour(col):
        return patches.Rectangle((0, 0), 1, 1, fc=colourise(col))

    proxy.append(colour(PUT))
    descr.append("Put")

    proxy.append(colour(CALL))
    descr.append("Call")

    proxy.append(colour(CONV))
    descr.append("Conversion")

    proxy.append(colour(FCONV))
    descr.append("Forced conversion")

    proxy.append(colour(REDEM))
    descr.append("Redemption")

    proxy.append(colour(HOLD))
    descr.append("Hold")
    ax.legend(proxy, descr, loc=2, prop={'size': 10})


def plot_strips(ax, X, Y, Z, padding=0, facecolors=None):
    """Plot the graph as a series of strips along the Y axis."""
    Yn = (Y[:-1] + Y[1:]) / 2.
    Yn = np.append(Y[0] - (Y[1] - Y[0]) / 2., Yn)
    Yn = np.append(Yn, Y[-1] + (Y[-1] - Y[-2]) / 2.)
    for x in range(len(X)):
        if x == 0:
            offset = (X[1] - X[0]) / 2.
        elif x == len(X) - 1:
            offset = (X[-1] - X[-2]) / 2.
        else:
            offset = (X[x + 1] - X[x - 1]) / 4.
        offset -= padding / 2.
        Xs, Ys = np.meshgrid([X[x] - offset, X[x] + offset], Yn)
        Zs = (Z[:-1, x] + Z[1:, x]) / 2.
        Zs = np.append(Z[0, x] - (Z[1, x] - Z[0, x]) / 2., Zs)
        Zs = np.append(Zs, Z[-1, x] + (Z[-1, x] - Z[-2, x]) / 2.)
        Zs = np.array([Zs, Zs]).T
        colours = np.zeros(Xs.shape + (4,))
        colours[0:-1, 0] = facecolors[:, x]
        ax.plot_surface(Xs, Ys, Zs, linewidth=0, cstride=1, rstride=1,
                        facecolors=colours, alpha=0.75)


def plot_model(ax, dS, payoff):
    N = 40
    model = FDEModel(N, dS, payoff)
    P = model.price(0, 250, 125, scheme=CrankNicolsonScheme)
    colours = choices(P)
    plot_strips(ax, P.t, P.S[:100], np.array(P.V)[:, :100].T, facecolors=colours[:100])
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.set_zlabel("Portfolio Value")
    legend(ax)