
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from model import FDEModel, WienerJumpProcess
from payoff import CallE

dS = WienerJumpProcess(0.1, 0.1)
# Semi-annual payments of 1
T = 3
V = CallE(T, 100)
model = FDEModel(T * 200, dS, V)
P = model.price(0, 200, 200)
X, Y = np.meshgrid(P.S, P.t)
Z = np.array(P.V)

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, linewidth=0)
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Time")
    ax.set_zlabel("Portfolio Value")
    plt.show()
