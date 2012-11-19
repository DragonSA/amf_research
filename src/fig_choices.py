"""
Graph showing the pricing surface of the example convertible bond.
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from convertible_bond import dS_total as dS, payoff
from plot import plot_model

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_model(ax, dS, payoff)
    plt.savefig("../common/fig_choices.pdf")
    #plt.show()


if __name__ == "__main__":
    main()
