"""
Unit tests for payoff module.
"""
import numpy as np
import unittest

from payoff import Forward, CallE, CallA, UpAndOut, Annuity

S0 = 100
T = 1
K = 100
N = 128
L = 105

class TestPayoff(unittest.TestCase):
    """Test derivative payoffs."""

    def test_default(self):
        """Test value of default."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        Vd = np.zeros(S.shape)
        for Payoff in (Forward, CallE, CallA, lambda T, K: UpAndOut(CallA(T, K), L)):
            payoff = Payoff(T, K)
            for t in np.linspace(0, 1, N, endpoint=False):
                self.assertTrue((payoff.default(t, S) == Vd).all())
            self.assertRaises(AssertionError, payoff.default, T, S)

    def test_zero_transient(self):
        """Test value of transient for European derivatives."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.linspace(S0 + 10, S0 - 10, 21)
        for Payoff in (Forward, CallE):
            payoff = Payoff(T, K)
            for t in np.linspace(0, 1, N, endpoint=False):
                self.assertTrue((payoff.transient(t, V, S) == V).all())
            self.assertRaises(AssertionError, payoff.transient, T, V, S)

    def test_call_transient(self):
        """Test value of transient for American call."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.linspace(S0 + 10, S0 - 10, 21)
        Vm = np.maximum(V, np.maximum(S - K, 0))
        payoff = CallA(T, K)
        for t in np.linspace(0, 1, N, endpoint=False):
            self.assertTrue((payoff.transient(t, V, S) == Vm).all())
        self.assertRaises(AssertionError, payoff.transient, T, V, S)

    def test_upandout_transcient(self):
        """Test value of transient for American up-and-out call."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.linspace(S0 + 10, S0 - 10, 21)
        Vm = np.maximum(V, np.maximum(S - K, 0))
        Vm[S >= L] = 0
        payoff = UpAndOut(CallA(T, K), L)
        for t in np.linspace(0, 1, N, endpoint=False):
            self.assertTrue((payoff.transient(t, V, S) == Vm).all())
        self.assertRaises(AssertionError, payoff.transient, T, V, S)

    def test_forward_terminal(self):
        """Test value of terminal for a call."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = S - K
        payoff = Forward(T, K)
        self.assertTrue((payoff.terminal(S) == V).all())

    def test_call_terminal(self):
        """Test value of terminal for a call."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.maximum(S - K, 0)
        for Call in (CallE, CallA):
            payoff = Call(T, K)
            self.assertTrue((payoff.terminal(S) == V).all())

    def test_upandout_terminal(self):
        """Test value of terminal for an up-and-out call."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.maximum(S - K, 0)
        V[S >= L] = 0
        payoff = UpAndOut(CallE(T, K), L)
        self.assertTrue((payoff.terminal(S) == V).all())


class TestAnnuity(unittest.TestCase):
    """Test Annuity payoff."""

    def test_default(self):
        """Test value of default."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        Vd = np.zeros(S.shape)
        payoff = Annuity(T, (), 1, 10)
        for t in np.linspace(0, 1, N, endpoint=False):
            self.assertTrue((payoff.default(t, S) == Vd).all())
        self.assertRaises(AssertionError, payoff.default, T, S)

    def test_transient(self):
        S = np.linspace(S0 - 10, S0 + 10, 21)
        Vp = np.ones(S.shape)
        Vo = np.zeros(S.shape)
        payoff = Annuity(T, np.arange(0, 1, 2. / N), 1, 10)
        pay = True
        for t in np.linspace(0, 1, N, endpoint=False):
            if pay:
                V = Vp
            else:
                V = Vo
            pay = not pay
            self.assertTrue((payoff.transient(t, Vo, S) == V).all())
        self.assertRaises(AssertionError, payoff.transient, T, Vo, S)

    def test_terminal(self):
        S = np.linspace(S0 - 10, S0 + 10, 21)
        Vp = np.ones(S.shape) * 11
        Vo = np.ones(S.shape) * 10
        self.assertTrue((Annuity(T, (T,), 1, 10).terminal(S) == Vp).all())
        self.assertTrue((Annuity(T, (), 1, 10).terminal(S) == Vo).all())


if __name__ == "__main__":
    unittest.main()
