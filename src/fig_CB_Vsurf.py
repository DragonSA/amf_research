"""
Graph showing the pricing surface of the example convertible bond.
"""
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from convertible_bond import dS_total as dS, payoff
from model import FDEModel

def main():
    N = 800
    model = FDEModel(N, dS, payoff)
    P = model.price(0, 200, 200)
    X, Y = np.meshgrid(P.S, P.t)
    Z = np.array(P.V)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, linewidth=0)
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Time")
    ax.set_zlabel("Portfolio Value")
    plt.show()

if __name__ == "__main__":
    main()
