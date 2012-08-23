"""
Framework for modelling payoff processes using either a binomial or finite-
difference model.
"""
import abc
import math
import numpy as np

__all__ = ["BinomialModel", "WienerJumpProcess"]

class Payoff(object):
    """
    The payoff description for a derivative, handling terminal, transient
    and default payoffs.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, T):
        self.T = T

    @abc.abstractmethod
    def default(self, t, S):
        """Payoff in the event of default at time t"""
        pass

    @abc.abstractmethod
    def terminal(self, S):
        """Payoff at terminal time."""
        pass

    def transient(self, t, V, S):
        """Payoff during transient (non-terminal) time."""
        assert(t != self.T)
        return V


class WienerJumpProcess(object):
    """
    A model of stochastic stock movement using a Wiener drift process to model
    volatility and a Poisson jump process to model default.

    The stock price follows:
        dS = (r + \lambda \eta) S_t dt + \sigma S_t dW_t - \eta \S_t dq_t
    where:
        dS      is the instantaneous stock movement
        dt      is the instantaneous change in time
        dW_t    is the Wiener drift process
        dq_t    is the Poisson jump process with intensity \lambda

        r       is the risk free rate
        \lambda is the hazard rate
        \eta    is the drop in stop price on a default event
        \sigma  is the log-volatility of the stock price
    """

    def __init__(self, r, sigma, lambd_, eta):
        if min(r, sigma, lambd_, eta) < 0:
            raise ValueError("all parameters must be non-negative")
        if eta > 1:
            raise ValueError("eta must be between 0 and 1 inclusive")
        self.r = r
        self.sigma = sigma
        self.lambd_ = lambd_
        self.eta = eta

    def binomial(self, dt):
        """Parameters for the binomial model."""
        # Up/down/loss multipliers
        u = math.exp(self.sigma * math.sqrt(dt))
        d = 1 / u
        l = 1 - self.eta

        # Probability of up/down/loss
        po = 1 - math.exp(-self.lambd_ * dt)
        pu = (math.exp(self.r * dt) - d * (1 - po) - l * po) / (u - d)
        pd = 1 - pu - po
        return (u, d, l, pu, pd, po)

class BinomialModel(object):
    """
    A binomial lattice model for pricing derivatives using a stock process and
    intrinsic values of the stock.

        N       is the number of increments
        dt      is the time increments for the binomial lattice
        dS      is the stock movement
        V       is the intrinsic value of the derivative
    """

    def __init__(self, N, dS, V):
        self.N = np.int(N)
        self.dt = np.double(V.T) / N
        self.dS = dS
        self.V = V

    def price(self, S0):
        u, d, l, pu, pd, po = self.dS.binomial(self.dt)
        du = d / u
        erdt = math.exp(-self.dS.r * self.dt)

        # Terminal stock price and derivative value
        S = np.array([S0 * u**i * d**(self.N - i) for i in range(self.N + 1)])
        V = self.V.terminal(S)

        # Discount price backwards
        for i in range(self.N - 1, -1, -1):
            # Discount previous derivative value
            S = np.array([S0 * u**j * d ** (i - j) for j in range(i + 1)])
            V = erdt * (V[1:] * pu + V[:-1] * pd + self.V.default(self.dt * i, S * l) * po)
            V = self.V.transient(self.dt * i, V, S)

        return V[0]
