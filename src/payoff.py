"""
Various payoff functions.
"""

import numpy as np

from model import Payoff

__all__ = ["Forward"]


class Forward(Payoff):
    """
    An Forward payoff, with total default and strike K.
    """

    def __init__(self, T, K):
        super(Forward, self).__init__(T)
        self.K = np.double(K)

    def default(self, t, S):
        """Total default."""
        assert(t != self.T)
        return np.zeros(S.shape)

    def terminal(self, S):
        """Terminal payoff of ``S - K''."""
        return S - self.K


class CallE(Payoff):
    """
    An European Call payoff, with total default and strike K.
    """

    def __init__(self, T, K):
        super(CallE, self).__init__(T)
        self.K = K

    def default(self, t, S):
        """Total default."""
        assert(t != self.T)
        return np.zeros(S.shape)

    def terminal(self, S):
        """Terminal payoff of ``max(S - K, 0)''."""
        return np.maximum(S - self.K, 0)


class CallA(CallE):
    """
    An American Call payoff, with total default and strike K.
    """

    def __init__(self, T, K):
        super(CallA, self).__init__(T, K)

    def default(self, t, S):
        """Residual value of default."""
        assert(t != self.T)
        return np.maximum(S - self.K, 0)

    def transient(self, t, V, S):
        """Transient payoff of ``max(S - K)''."""
        assert(t != self.T)
        return np.maximum(V, S - self.K)


class UpAndOut(Payoff):
    """
    A Up-and-Out derivative.
    """

    def __init__(self, payoff, L):
        super(UpAndOut, self).__init__(payoff.T)
        self.payoff = payoff
        self.L = L

    def default(self, t, S):
        """Default value of payoff."""
        assert(t != self.T)
        V = np.zeros(S.shape)
        idx = S < self.L
        V[idx] = self.payoff.default(t, S[idx])
        return V

    def transient(self, t, V, S):
        """Transient payoff, 0 if above threshold, otherwise payoff."""
        Vp = np.zeros(S.shape)
        idx = S < self.L
        Vp[idx] = self.payoff.transient(t, V[idx], S[idx])
        return Vp

    def terminal(self, S):
        """Terminal payoff, 0 if above threshold, otherwise payoff."""
        V = np.zeros(S.shape)
        idx = S < self.L
        V[idx] = self.payoff.terminal(S[idx])
        return V


class Annuity(Payoff):
    """
    Payment process that pays a fixed amount at specified times.
    """

    def __init__(self, T, times, C, N=0, R=0):
        super(Annuity, self).__init__(T)
        self.times = times
        self.C = np.double(C)
        self.N = np.double(N)
        self.R = np.double(R)

    def default(self, t, S):
        """Residual value on default"""
        assert(t != self.T)
        return np.ones(S.shape) * self.N * self.R

    def transient(self, t, V, S):
        assert(t != self.T)
        if t in self.times:
            return V + self.C
        else:
            return V

    def terminal(self, S):
        payment = self.N
        if self.T in self.times:
            payment += self.C
        return np.ones(S.shape) * payment
