"""
Framework for modelling payoff processes using either a binomial or finite-
difference model.
"""
import math
import numpy as np

__all__ = ["BinomialModel", "WienerJumpProcess"]


class Payoff(object):
    """
    The payoff description for a derivative, handling terminal, transient
    and default payoffs.
    """

    def __init__(self, T):
        self.T = T

    def __contains__(self, t):
        return 0 <= t <= T

    def default(self, t, S):
        """Payoff in the event of default at time t"""
        assert(t != self.T)
        return np.zeros(S.shape)

    def terminal(self, S):
        """Payoff at terminal time."""
        return np.zeros(S.shape)

    def transient(self, t, V, S):
        """Payoff during transient (non-terminal) time."""
        assert(t != self.T)
        return V

    def coupon(self, t):
        """Payoff of coupon."""
        return 0


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

    def __init__(self, r, sigma, lambd_=0, eta=1, cap_lambda=False):
        if min(r, sigma, lambd_, eta) < 0:
            raise ValueError("all parameters must be non-negative")
        if eta > 1:
            raise ValueError("eta must be between 0 and 1 inclusive")
        self.r = np.double(r)
        self.sigma = np.double(sigma)
        self.lambd_ = lambd_ if callable(lambd_) else np.double(lambd_)
        self.eta = np.double(eta)
        self.cap_lambda = cap_lambda

    def binomial(self, dt, S=None):
        """Parameters for the binomial model."""
        # Up/down/loss multipliers
        if self.sigma <= self.r * np.sqrt(dt):
            raise ValueError("Time step to big for given volatility")
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        l = 1 - self.eta

        if not callable(self.lambd_):
            lambd_ = self.lambd_
        elif S is not None:
            lambd_ = self.lambd_(S)
            if (lambd_ < 0).any():
                raise ValueError("Hazard rate most be non-negative")
        else:
            return (u, d, l, False)

        # Probability of up/down/loss
        lambda_limit = (np.log(u - l) - np.log(np.exp(self.r * dt) - l)) / dt
        if self.cap_lambda:
            lambd_ = np.minimum(lambd_, lambda_limit)
        elif (lambd_ > lambda_limit).any():
            raise ValueError("Time step to big for given hazard rate")
        po = 1 - np.exp(-lambd_ * dt)
        pu = (np.exp(self.r * dt) - d * (1 - po) - l * po) / (u - d)
        pd = 1 - pu - po
        if S is not None:
            return (pu, pd, po)
        else:
            return (u, d, l, (pu, pd, po))


class BinomialModel(object):
    """
    A binomial lattice model for pricing derivatives using a stock process and
    intrinsic values of the stock.

        N       is the number of increments
        dt      is the time increments for the binomial lattice
        dS      is the stock movement
        V       is the intrinsic value of the derivative
    """

    class Value(object):
        """
        Valuation of the portfolio using the binomial model.

            N   is the number of nodes (excluding base node)
            t   is the node times
            S   is the node prices
            C   is the node coupons
            X   is the the node default value
            V   is the value of the portfolio
        """

        def __init__(self, T, N):
            self.N = N
            self.t = np.linspace(0, T, N + 1)
            self.S = [None] * (N + 1)
            self.C = [None] * (N + 1)
            self.X = [None] * (self.N)
            self.V = [None] * (N + 1)

        def __float__(self):
            return self.V[0][0]

    def __init__(self, N, dS, V):
        self.N = np.int(N)
        self.dt = np.double(V.T) / N
        self.dS = dS
        self.V = V

    def price(self, S0):
        u, d, l, prob = self.dS.binomial(self.dt)
        erdt = math.exp(-self.dS.r * self.dt)
        P = self.Value(self.V.T, self.N)
        if prob:
            pu, pd, po = prob

        # Terminal stock price and derivative value
        P.S[-1] = S = np.array([S0 * u**(self.N - i) * d**i for i in range(self.N + 1)])
        if not prob:
            pu, pd, po = self.dS.binomial(self.dt, P.S[-1])
        P.C[-1] = C = self.V.coupon(self.V.T)
        P.V[-1] = V = self.V.terminal(S) + C

        # Discount price backwards
        t = P.t
        for i in range(self.N - 1, -1, -1):
            # Discount previous derivative value
            P.S[i] = S = np.array([S0 * u**(i - j) * d**j for j in range(i + 1)])
            P.X[i] = X = self.V.default(t[i], S * l)
            P.C[i] = C = self.V.coupon(t[i])
            if not prob:
                pu, pd, po = self.dS.binomial(self.dt, S)
            V = erdt * (V[:-1] * pu + V[1:] * pd + X * po)
            P.V[i] = V = self.V.transient(t[i], V, S) + C

        return P
