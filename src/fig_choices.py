"""
Graph showing the pricing surface of the example convertible bond.
"""
import matplotlib.pyplot as plt

from convertible_bond import dS_total as dS, payoff
from plot import plot_model, plotmain

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_model(ax, dS, payoff)


if __name__ == "__main__":
    plotmain(main)
