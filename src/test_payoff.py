"""
Unit tests for payoff module.
"""
import numpy as np
import unittest

from payoff import Forward, CallE, CallA, CallVR, PutV, Stack, Time, UpAndOut, \
                   Annuity

S0 = 100
T = 1
K = 100
N = 128
L = 105

class TestPayoff(unittest.TestCase):
    """Test derivative payoffs."""

    def test_zero_default(self):
        """Test value of default."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        Vd = np.zeros(S.shape)
        for Payoff in (Forward, CallE, CallVR, PutV):
            payoff = Payoff(T, K)
            for t in np.linspace(0, 1, N, endpoint=False):
                self.assertTrue((payoff.default(t, S) == Vd).all())
            self.assertRaises(AssertionError, payoff.default, T, S)

    def test_call_default(self):
        """Test value of default for a call."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        Vd = np.maximum(S - K, 0)
        payoff = CallA(T, K)
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
        Vm = np.maximum(S - K, V)
        payoff = CallA(T, K)
        for t in np.linspace(0, 1, N, endpoint=False):
            self.assertTrue((payoff.transient(t, V, S) == Vm).all())
        self.assertRaises(AssertionError, payoff.transient, T, V, S)

    def test_callvr_transcient(self):
        """Test value of transient for reverse American call on portfolio."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.linspace(S0 + 10, S0 - 10, 21)
        Vm = V - np.maximum(V - K, 0)
        payoff = CallVR(T, K)
        for t in np.linspace(0, 1, N, endpoint=False):
            self.assertTrue((payoff.transient(t, V, S) == Vm).all())
        self.assertRaises(AssertionError, payoff.transient, T, V, S)

    def test_putv_transcient(self):
        """Test value of transient for American put on portfolio."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.linspace(S0 + 10, S0 - 10, 21)
        Vm = V + np.maximum(K - V, 0)
        payoff = PutV(T, K)
        for t in np.linspace(0, 1, N, endpoint=False):
            self.assertTrue((payoff.transient(t, V, S) == Vm).all())
        self.assertRaises(AssertionError, payoff.transient, T, V, S)

    def test_zero_terminal(self):
        """Test value of terminal for zero no terminal value."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.zeros(S.shape)
        for Payoff in (CallVR, PutV):
            payoff = Payoff(T, K)
            self.assertTrue((payoff.terminal(S) == V).all())

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


class TestAnnuity(unittest.TestCase):
    """Test Annuity payoff."""

    def test_default(self):
        """Test value of default."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        Vd = np.ones(S.shape) * 5
        payoff = Annuity(T, (), 1, 10, 0.5)
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


class TestStack(unittest.TestCase):
    """Test Stack payoff."""

    def test_default(self):
        """Test default value of stacked Forward and American call."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        Vd = np.maximum(S - K, 0)
        payoff = Stack([Forward(T, K), CallA(T, K)])
        for t in np.linspace(0, 1, N, endpoint=False):
            self.assertTrue((payoff.default(t, S) == Vd).all())
        self.assertRaises(AssertionError, payoff.default, T, S)

    def test_transcient(self):
        """Test value of transient for stacked normal and reversed American call."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.linspace(S0 + 10, S0 - 10, 21)
        Vm = np.minimum(np.maximum(V, np.maximum(S - K, 0)), K + 5)
        payoff = Stack([CallA(T, K), CallVR(T, K + 5)])
        for t in np.linspace(0, 1, N, endpoint=False):
            self.assertTrue((payoff.transient(t, V, S) == Vm).all())
        self.assertRaises(AssertionError, payoff.transient, T, V, S)

    def test_terminal(self):
        """Test value of terminal for stacked Forward and American call."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.maximum(S - K + 5, 0)
        payoff = Stack([Forward(T, K - 5), CallE(T, K)])
        self.assertTrue((payoff.terminal(S) == V).all())


class TestTime(unittest.TestCase):
    """Test Time payoff."""

    def test_default(self):
        """Test default value of American call with time restrictions."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        times = list(np.linspace(0, 1, N // 4, endpoint=False)) + [(0.5, 1)]
        Vd = np.maximum(S - K, 0)
        Vo = np.zeros(S.shape)
        payoff = Time(CallA(T, K), times)
        for t in np.linspace(0, 1, N, endpoint=False):
            if float(t) in times or t > 0.5:
                self.assertTrue((payoff.default(t, S) == Vd).all())
            else:
                self.assertTrue((payoff.default(t, S) == Vo).all())
        self.assertRaises(AssertionError, payoff.default, T, S)

    def test_transcient(self):
        """Test value of transient for American call with time restrictions."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.linspace(S0 + 10, S0 - 10, 21)
        times = list(np.linspace(0, 1, N // 4, endpoint=False)) + [(0.5, 1)]
        Vm = np.maximum(S - K, V)
        payoff = Time(CallA(T, K), times)
        for t in np.linspace(0, 1, N, endpoint=False):
            if float(t) in times or t > 0.5:
                self.assertTrue((payoff.transient(t, V, S) == Vm).all())
            else:
                self.assertTrue((payoff.transient(t, V, S) == V).all())
        self.assertRaises(AssertionError, payoff.transient, T, V, S)

    def test_terminal(self):
        """Test value of terminal for American call with time restrictions."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.maximum(S - K, 0)
        payoff = Time(CallE(T, K), (T,))
        self.assertTrue((payoff.terminal(S) == V).all())
        payoff = Time(CallE(T, K), ())
        self.assertTrue((payoff.terminal(S) == np.zeros(S.shape)).all())


class TestUpAndOut(unittest.TestCase):
    """Test UpAndOut payoff."""

    def test_default(self):
        """Test default value for American up-and-out call."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        Vd = np.maximum(S - K, 0)
        Vd[S >= L] = 0
        payoff = UpAndOut(CallA(T, K), L)
        for t in np.linspace(0, 1, N, endpoint=False):
            self.assertTrue((payoff.default(t, S) == Vd).all())
        self.assertRaises(AssertionError, payoff.default, T, S)

    def test_transcient(self):
        """Test value of transient for American up-and-out call."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.linspace(S0 + 10, S0 - 10, 21)
        Vm = np.maximum(V, np.maximum(S - K, 0))
        Vm[S >= L] = 0
        payoff = UpAndOut(CallA(T, K), L)
        for t in np.linspace(0, 1, N, endpoint=False):
            self.assertTrue((payoff.transient(t, V, S) == Vm).all())
        self.assertRaises(AssertionError, payoff.transient, T, V, S)

    def test_terminal(self):
        """Test value of terminal for an up-and-out call."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.maximum(S - K, 0)
        V[S >= L] = 0
        payoff = UpAndOut(CallA(T, K), L)
        self.assertTrue((payoff.terminal(S) == V).all())


if __name__ == "__main__":
    unittest.main()
