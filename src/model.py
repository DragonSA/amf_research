"""
Framework for modelling payoff processes using either a binomial or finite-
difference model.
"""
import abc
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
np.seterr(divide="ignore")

__all__ = [
        "BinomialModel", "FDEModel",
        "Payoff", "WienerJumpProcess",
        "CrankNicolsonScheme", "ExplicitScheme", "ImplicitScheme",
        "RannacherScheme", "PenaltyScheme", "PenaltyRannacherScheme",
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

    def fde(self, dt, ds, S, scheme, boundary="diffequal", expfit=False):
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
        if expfit:
            x = (self.r + lambd_ * self.eta) * ds / self.sigma**2 / S
            coth = 1. / np.tanh(x)
            sS = dt * coth * (self.r + lambd_ * self.eta) * S / ds / 2
        else:
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


class Scheme(object):
    """
    Difference equation.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, S):
        super(Scheme, self).__init__()
        self.S = S

    def __call__(self, t, V, X, C, payoff):
        return payoff(t, self.scheme(V, X), self.S) + C

    @abc.abstractmethod
    def scheme(self, V, X):
        """Discount portfolio value back one period."""
        pass


class ExplicitScheme(Scheme):
    """
    Explicit difference equation.
    """

    def __init__(self, dS, dt, ds, S, **kwargs):
        super(ExplicitScheme, self).__init__(S)
        a, b, c, d = dS.fde(dt, ds, S, "explicit", **kwargs)
        self.L = sparse.dia_matrix(([a, 1 + b, c], [-1, 0, 1]), shape=S.shape*2)
        self.d = d
        if False and (abs(self.L) > 1).any():
            raise ValueError("Time step to big for given stock increments")

    def scheme(self, V, X):
        return self.L.dot(V) + self.d * X


class ImplicitScheme(Scheme):
    """
    Implicit difference equation.
    """

    def __init__(self, dS, dt, ds, S, **kwargs):
        super(ImplicitScheme, self).__init__(S)
        K = S.shape*2
        a, b, c, d = dS.fde(dt, ds, S, "implicit", **kwargs)
        self.L = sparse.dia_matrix(([-a, 1 - b, -c], [-1, 0, 1]), shape=S.shape*2).tocsr()
        self.d = d

    def scheme(self, V, X):
        return linalg.spsolve(self.L, V + self.d * X)


class CrankNicolsonScheme(Scheme):
    """
    Crank-Nicolson difference equation.
    """

    def __init__(self, dS, dt, ds, S, **kwargs):
        super(CrankNicolsonScheme, self).__init__(S)
        a, b, c, d = dS.fde(dt, ds, S, "explicit", **kwargs)
        self.Le = sparse.dia_matrix(([a, 2 + b, c], [-1, 0, 1]), shape=S.shape*2)
        a, b, c, d = dS.fde(dt, ds, S, "implicit", **kwargs)
        self.Li = sparse.dia_matrix(([-a, 2 - b, -c], [-1, 0, 1]), shape=S.shape*2).tocsr()
        self.d = 2 * d

    def scheme(self, V, X):
        return linalg.spsolve(self.Li, self.Le.dot(V) + self.d * X)


class RannacherScheme(CrankNicolsonScheme):
    """
    Crank-Nicolson difference equation with fully implicit quarter steps to
    handle discontinuous payoffs.
    """

    def __init__(self, dS, dt, ds, S, **kwargs):
        super(RannacherScheme, self).__init__(dS, dt, ds, S, **kwargs)
        a, b, c, d = dS.fde(dt / 4, ds, S, "implicit")
        self.Lq = sparse.dia_matrix(([-a, 1 - b, -c], [-1, 0, 1]), shape=S.shape*2).tocsr()
        self.imp = 1

    def __call__(self, t, V, X, C, payoff):
        if self.imp > 0:
            V = self.scheme_implicit(V, X)
            self.imp -= 1
        else:
            V = self.scheme(V, X)
        if C != 0:
            self.imp += 1
        return payoff(t, V, self.S) + C

    def scheme_implicit(self, V, X):
        for i in range(4):
            V = linalg.spsolve(self.Lq, V + self.d * X)
        return V


class PenaltyScheme(CrankNicolsonScheme):
    """
    Crank-Nicolson difference equation using penalty iterations to impose the
    American constraints.
    """

    def __init__(self, dS, dt, ds, S, tol=None, **kwargs):
        super(PenaltyScheme, self).__init__(dS, dt, ds, S, **kwargs)
        if tol is None:
            tol = min(dt, ds)**2
        assert(tol > 0)
        self.tol = np.double(tol)

    def __call__(self, t, V, X, C, payoff):
        # V after explicit step
        diag = lambda x: sparse.dia_matrix(([x], [0]), shape=V.shape*2)
        Vx = self.Le.dot(V) + self.d * X
        Vk = linalg.spsolve(self.Li, Vx)
        Vs = payoff(t, Vk, self.S)
        Pk = (Vs != Vk) / self.tol
        if (Pk == 0).all():
            return Vk + C
        while True:
            Vk1 = linalg.spsolve(self.Li + diag(Pk), Vx + Pk * Vs)
            if np.max(np.abs(Vk1 - Vk) / np.maximum(1, np.abs(Vk1))) < self.tol:
                break
            Vs = payoff(t, Vk1, self.S)
            Pk1 = (Vs != Vk1) / self.tol
            if (Pk1 == Pk).all():
                break
            Pk, Vk = Pk1, Vk1
        return Vk1 + C


class PenaltyRannacherScheme(PenaltyScheme, RannacherScheme):
    """
    Crank-Nicolson difference equatioin using penalty iterations to impose the
    American constrains and fully implicit quarter steps to handle discontinuous
    payoffs.
    """

    def __call__(self, t, V, X, C, payoff):
        if self.imp > 0:
            V = payoff(t, self.scheme_implicit(V, X), self.S) + C
            self.imp -= 1
        else:
            V = PenaltyScheme.__call__(self, t, V, X, C, payoff)
        if C != 0:
            self.imp += 1
        return V


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
        def __init__(self, T, N, Sl, Su, K):
            super(FDEModel.Value, self).__init__(T, N)
            assert(Su > Sl >= 0)
            self.K = K
            self.S = np.linspace(Sl, Su, K + 1)


    def __init__(self, N, dS, V):
        self.N = np.int(N)
        self.dt = np.double(V.T) / N
        self.dS = dS
        self.V = V

    def price(self, Sl, Su, K, scheme=PenaltyRannacherScheme, **kwargs):
        """
        Price the payoff for prices in range [Sl, Su], and K increments, using
        the given FD scheme, and possibility using exponential increments
        (zspace) for the price range.
        """
        Sl = np.double(Sl)
        Su = np.double(Su)
        K = np.int(K)
        P = FDEModel.Value(self.V.T, self.N, Sl, Su, K)
        S = P.S
        ds = P.S[1] - P.S[0]
        Sl = P.S * self.dS.fde_l()

        # Terminal stock price and derivative value
        P.C[-1] = C = self.V.coupon(self.V.T)
        P.V[-1] = V = self.V.terminal(S) + C

        # Discount price backwards
        t = P.t
        scheme = scheme(self.dS, self.dt, ds, S, **kwargs)
        for i in range(self.N - 1, -1, -1):
            # Discount previous derivative value
            P.C[i] = C = self.V.coupon(t[i])
            P.X[i] = X = self.V.default(t[i], Sl)
            P.V[i] = V = scheme(t[i], V, X, C, self.V.transient)
        return P


class FDEBVModel(FDEModel):
    """
    A finite difference equation scheme for pricing derivatives using a stock
    process and intrinsic values of the stock.  This process uses pricing by
    splitting the intrinsic value into a bond and equity component

        N       is the number of time increments
        dt      is the time increments for the binomial lattice
        dS      is the stock movement
        B       is the bond value of the derivative
        V       is the (total) intrinsic value of the derivative
    """

    def __init__(self, N, dS, B, V):
        super(FDEBVModel, self).__init__(N, dS, V)
        self.B = B

    def price(self, Sl, Su, K, scheme=CrankNicolsonScheme, **kwargs):
        """
        Price the payoff for prices in range [Sl, Su], and K increments, using
        the given FD scheme, and possibility using exponential increments
        (zspace) for the price range.
        """
        Sl = np.double(Sl)
        Su = np.double(Su)
        K = np.int(K)
        P = FDEModel.Value(self.V.T, self.N, Sl, Su, K)
        S = P.S
        ds = P.S[1] - P.S[0]
        Sl = P.S * self.dS.fde_l()

        # Terminal stock price and derivative value
        P.C[-1] = C = self.V.coupon(self.V.T)
        P.V[-1] = V = self.V.terminal(S) + C
        B = self.B.terminal(S) + C
        E = V - B

        # Discount price backwards
        t = P.t
        scheme = scheme(self.dS, self.dt, ds, S, **kwargs)
        for i in range(self.N - 1, -1, -1):
            # Discount previous derivative value
            P.C[i] = C = self.V.coupon(t[i])
            P.X[i] = X = self.V.default(t[i], Sl)
            XB = self.B.default(t[i], Sl)
            XE = X - XB
            B = scheme.scheme(B, XB)
            E = scheme.scheme(E, XE)
            B_ = self.B.transient(t[i], B, S)
            B = np.minimum(B_, np.maximum(B_ - E, B))
            E = self.V.transient(t[i], B + E, S) - B
            B += C
            #V = scheme(V, X)
            #P.V[i] = V = self.V.transient(t[i], V, S) + C
            P.V[i] = V = B + E
        return P
