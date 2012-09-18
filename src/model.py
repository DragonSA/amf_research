"""
Framework for modelling payoff processes using either a binomial or finite-
difference model.
"""
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

__all__ = [
        "BinomialModel", "FDEModel",
        "Payoff", "WienerJumpProcess",
        "CrankNicolsonScheme", "ExplicitScheme", "ImplicitScheme",
    ]


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
        if sigma <= 0:
            raise ValueError("Volatility must be positive")
        if lambd_ < 0:
            raise ValueError("Hazard rate must be non-negative")
        self.r = np.double(r)
        self.sigma = np.double(sigma)
        self.lambd_ = lambd_ if callable(lambd_) else np.double(lambd_)
        self.eta = np.double(eta)
        self.cap_lambda = cap_lambda

    def binomial(self, dt, S=None):
        """Parameters for the binomial model."""
        # Up/down/loss multipliers
        if dt <= 0:
            raise ValueError("Time step must be positive")
        if self.sigma**2 < self.r**2 * dt:
            raise ValueError("Time step to big for given volatility")
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        l = 1 - self.eta

        if not callable(self.lambd_):
            lambd_ = self.lambd_
        elif S is not None:
            lambd_ = self.lambd_(S)
            if (lambd_ < 0).any():
                raise ValueError("Hazard rate must be non-negative")
        else:
            return (u, d, l, False)

        # Probability of up/down/loss
        lambda_limit = (np.log(u - l) - np.log(np.exp(self.r * dt) - l))
        if self.cap_lambda:
            lambd_ = np.minimum(lambd_, lambda_limit / dt)
        elif (lambd_ * dt > lambda_limit).any():
            raise ValueError("Time step to big for given hazard rate")
        po = 1 - np.exp(-lambd_ * dt)
        pu = (np.exp(self.r * dt) - d * (1 - po) - l * po) / (u - d)
        pd = 1 - pu - po
        if S is not None:
            return (pu, pd, po)
        else:
            return (u, d, l, (pu, pd, po))

    def fde_l(self):
        """Parameter for stock jump on default."""
        return 1 - self.eta

    def fde(self, dt, ds, S, scheme, boundary):
        """Parameters for the finite difference scheme."""
        if dt <= 0:
            raise ValueError("Time step must be positive")
        if ds <= 0:
            raise ValueError("Stock step must be positive")
        if (S < 0).any():
            raise ValueError("Stock must be non-negative")
        if callable(self.lambd_):
            lambd_ = self.lambd_(S)
        else:
            lambd_ = self.lambd_
        rdt = (self.r + lambd_) * dt
        rS = dt * (self.r + lambd_ * self.eta) * S / ds / 2
        sS = dt * self.sigma**2 * S**2 / ds**2 / 2
        a = sS[1:] - rS[1:]
        b = -rdt - sS * 2
        c = sS[:-1] + rS[:-1]
        d = lambd_ * dt
        if boundary == "equal":
            b[0] += sS[0] - rS[0]
            b[-1] += sS[-1] + rS[-1]
        elif boundary == "diffequal":
            b[0] += 2 * (sS[0] - rS[0])
            c[0] -= sS[0] - rS[0]
            b[-1] += 2 * (sS[-1] + rS[-1])
            a[-1] -= sS[-1] + rS[-1]
        elif boundary != "ignore":
            raise ValueError("unknown boundary type: %s" % boundary)
        return (np.append(a, sS[0] - rS[0]), b, np.append(sS[-1] + rS[-1], c), d)


class Value(object):
    """
    Valuation of a portfolio.

        N   is the number of time nodes (excluding base node)
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
        self.C = np.zeros(N + 1)
        self.X = [None] * (self.N)
        self.V = [None] * (N + 1)


class BinomialModel(object):
    """
    A binomial lattice model for pricing derivatives using a stock process and
    intrinsic values of the stock.

        N       is the number of time increments
        dt      is the time increments for the binomial lattice
        dS      is the stock movement
        V       is the intrinsic value of the derivative
    """

    class Value(Value):
        """
        Valuation of a portfolio for the binomial model.

            N   is the number of time nodes (excluding base node)
            t   is the node times
            S   is the node prices
            C   is the node coupons
            X   is the the node default value
            V   is the value of the portfolio
        """
        def __float__(self):
            return self.V[0][0]

    def __init__(self, N, dS, V):
        self.N = np.int(N)
        self.dt = np.double(V.T) / N
        self.dS = dS
        self.V = V

    def price(self, S0):
        """Price the payoff for stock price S0 at time 0."""
        So = np.double(S0)
        u, d, l, prob = self.dS.binomial(self.dt)
        erdt = np.exp(-self.dS.r * self.dt)
        P = BinomialModel.Value(self.V.T, self.N)
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
            P.S[i] = S = S[:-1] / u
            P.X[i] = X = self.V.default(t[i], S * l)
            P.C[i] = C = self.V.coupon(t[i])
            if not prob:
                pu, pd, po = self.dS.binomial(self.dt, S)
            V = erdt * (V[:-1] * pu + V[1:] * pd + X * po)
            P.V[i] = V = self.V.transient(t[i], V, S) + C

        return P


class ExplicitScheme(object):
    """
    Explicit difference equation.
    """

    def __init__(self, dS, dt, ds, S, boundary):
        a, b, c, d = dS.fde(dt, ds, S, "explicit", boundary)
        self.L = sparse.dia_matrix(([a, 1 + b, c], [-1, 0, 1]), shape=S.shape*2)
        self.d = d
        if False and (abs(self.L) > 1).any():
            raise ValueError("Time step to big for given stock increments")

    def __call__(self, V, X):
        return self.L.dot(V) + self.d * X


class ImplicitScheme(object):
    """
    Implicit difference equation.
    """

    def __init__(self, dS, dt, ds, S, boundary):
        K = S.shape*2
        a, b, c, d = dS.fde(dt, ds, S, "implicit", boundary)
        self.L = sparse.dia_matrix(([-a, 1 - b, -c], [-1, 0, 1]), shape=S.shape*2).tocsr()
        self.d = d

    def __call__(self, V, X):
        return linalg.spsolve(self.L, V + self.d * X)


class CrankNicolsonScheme(object):
    """
    Crank-Nicolson difference equation.
    """

    def __init__(self, dS, dt, ds, S, boundary):
        a, b, c, d = dS.fde(dt, ds, S, "explicit", boundary)
        self.Le = sparse.dia_matrix(([a, 2 + b, c], [-1, 0, 1]), shape=S.shape*2)
        a, b, c, d = dS.fde(dt, ds, S, "implicit", boundary)
        self.Li = sparse.dia_matrix(([-a, 2 - b, -c], [-1, 0, 1]), shape=S.shape*2).tocsr()
        self.d = d

    def __call__(self, V, X):
        return linalg.spsolve(self.Li, self.Le.dot(V) + self.d * X)


class FDEModel(object):
    """
    A finite difference equation scheme for pricing derivatives using a stock
    process and intrinsic values of the stock.

        N       is the number of time increments
        dt      is the time increments for the binomial lattice
        dS      is the stock movement
        V       is the intrinsic value of the derivative
    """

    class Value(Value):
        """
        Valuation of a portfolio for the finite difference equation model.

            N   is the number of time nodes (excluding base node)
            t   is the node times
            K   is the number of stock nodes (excluding one boundary)
            S   is the node prices
            C   is the node coupons
            X   is the the node default value
            V   is the value of the portfolio
            Z   is the log price
        """
        def __init__(self, T, N, Sl, Su, K, zspace):
            super(FDEModel.Value, self).__init__(T, N)
            self.K = K
            if zspace:
                assert(Su > Sl > 0)
                Zl, Zu = np.log((Sl, Su))
                self.S = np.exp(np.linspace(Zl, Zu, K + 1))
            else:
                assert(Su > Sl >= 0)
                self.S = np.linspace(Sl, Su, K + 1)


    def __init__(self, N, dS, V):
        self.N = np.int(N)
        self.dt = np.double(V.T) / N
        self.dS = dS
        self.V = V

    def price(self, Sl, Su, K, scheme=CrankNicolsonScheme, boundary="diffequal", zspace=False):
        """
        Price the payoff for prices in range [Sl, Su], and K increments, using
        the given FD scheme, and possibility using exponential increments
        (zspace) for the price range.
        """
        Sl = np.double(Sl)
        Su = np.double(Su)
        K = np.int(K)
        P = FDEModel.Value(self.V.T, self.N, Sl, Su, K, zspace)
        S = P.S
        if zspace:
            ds = (np.log(Su) - np.log(Sl)) / K
            Z = np.ones(P.S.shape)
        else:
            ds = P.S[1] - P.S[0]
            Z = P.S
        Sl = P.S * self.dS.fde_l()

        # Terminal stock price and derivative value
        P.C[-1] = C = self.V.coupon(self.V.T)
        P.V[-1] = V = self.V.terminal(S) + C

        # Discount price backwards
        t = P.t
        scheme = scheme(self.dS, self.dt, ds, Z, boundary)
        for i in range(self.N - 1, -1, -1):
            # Discount previous derivative value
            P.C[i] = C = self.V.coupon(t[i])
            P.X[i] = X = self.V.default(t[i], Sl)
            V = scheme(V, X)
            P.V[i] = V = self.V.transient(t[i], V, S) + C

        return P
