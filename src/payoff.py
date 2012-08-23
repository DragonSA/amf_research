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

    def default(self, t, _V, S):
        """Total default in default."""
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

    def default(self, t, _V, S):
        assert(t != self.T)
        """Total default in default."""
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

    def transient(self, t, V, S):
        """Transient payoff of ``max(S - K)''."""
        assert(t != self.T)
        return np.maximum(V, S - self.K)
