"""
Unit tests for payoff module.
"""
import numpy as np
import unittest

from payoff import Forward, CallE, CallA, CallVR, PutA, PutE, PutV, \
                   Stack, Time, UpAndOut, \
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
        for Payoff in (Forward, CallE, CallVR, PutE, PutV):
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

    def test_put_default(self):
        """Test value of default for a put."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        Vd = np.maximum(K - S, 0)
        payoff = PutA(T, K)
        for t in np.linspace(0, 1, N, endpoint=False):
            self.assertTrue((payoff.default(t, S) == Vd).all())
        self.assertRaises(AssertionError, payoff.default, T, S)

    def test_zero_transient(self):
        """Test value of transient for European derivatives."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.linspace(S0 + 10, S0 - 10, 21)
        for Payoff in (Forward, CallE, PutE):
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

    def test_put_transient(self):
        """Test value of transient for American put."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.linspace(S0 + 10, S0 - 10, 21)
        Vm = np.maximum(K - S, V)
        payoff = PutA(T, K)
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

    def test_put_terminal(self):
        """Test value of terminal for a call."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        V = np.maximum(K - S, 0)
        for Put in (PutE, PutA):
            payoff = Put(T, K)
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

    def test_coupon(self):
        """Test coupon value of annuity."""
        payoff = Annuity(T, np.linspace(0, 1, N // 2 + 1), 1, 10)
        pay = True
        for t in np.linspace(0, 1, N + 1):
            if pay:
                V = 1
            else:
                V = 0
            pay = not pay
            self.assertTrue(payoff.coupon(t) == V)

    def test_terminal(self):
        """Test terminal value of annuity."""
        S = np.linspace(S0 - 10, S0 + 10, 21)
        Vo = np.ones(S.shape) * 10
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


class TestConvertibleBond(unittest.TestCase):
    """Test Payoff of unit tests, based on AKW08 Table 1"""

    def setUp(self):
        T = 9
        B = Annuity(T, [8], 4, 100, 0.5)
        P = Time(PutV(T, 105), [2, 4, 5, 6])
        C = Time(CallVR(T, 110), [3, 4, 6, 7])
        S = Time(CallA(T, 0), [1, 2, 3, 4, 9])
        self.payoff = Stack([B, P, C, S])
        Ssect = [np.linspace(0, 0.25, 10), np.linspace(0.25, -.25, 10),
                 np.linspace(-.25, 0.75, 10), np.linspace(0.75, 1, 9, endpoint=False)]
        Vsect = [np.linspace(0, -.25, 10), np.linspace(-.25, 0.25, 10),
                 np.linspace(0.25, 0.50, 10), np.linspace(0.50, 1, 9, endpoint=False)]
        Ssect = np.array(sum((list(i) for i in Ssect), []))
        Vsect = np.array(sum((list(i) for i in Vsect), []))
        critical = [10, 50, 100, 105, 110, 200]
        S = []
        V = []
        for i in range(len(critical) - 1):
            l, u = critical[i:i + 2]
            lu = u -l
            S.extend(Ssect * lu + l)
            V.extend(Vsect * lu + l)
        self.S = np.array(S)
        self.V = np.array(V)

    def test_SV(self):
        """Test sample stock and portfolio values."""
        S = self.S
        V = self.V
        for i in [50, 100, 105, 110]:
            self.assertTrue((S > i).any())
            self.assertTrue((S == i).any())
            self.assertTrue((S < i).any())
            self.assertTrue((V > i).any())
            self.assertTrue((V == i).any())
            self.assertTrue((V < i).any())
            self.assertTrue(((S > i) & (V < i)).any())
            self.assertTrue(((S < i) & (V > i)).any())

    def test_conv(self):
        """Test conversion option, without call or put option."""
        t = 1
        S = self.S
        V = self.V
        Vm = np.maximum(V, S)
        self.assertTrue((self.payoff.transient(t, V, S) == Vm).all())

    def test_conv_put(self):
        """Test conversion and put option without call option."""
        t = 2
        S = self.S
        V = self.V
        Vm = np.maximum(V, S)  # Normal conversion
        # Put option
        put = (V < 105) & (S < 105)
        self.assertTrue(put.any())
        Vm[put] = 105
        self.assertTrue((self.payoff.transient(t, V, S) == Vm).all())

    def test_conv_call(self):
        """Test conversion and call option, without put option."""
        t = 3
        S = self.S
        V = self.V
        Vm = np.maximum(V, S)  # Normal conversion
        # Call option
        call = (V > 110) & (S <= 110)
        Vm[call] = 110
        self.assertTrue(call.any())
        # Conversion option (forced conversion)
        conv = (V > 110) & (S > 110) & (V > S)
        self.assertTrue(conv.any())
        Vm[conv] = S[conv]
        self.assertTrue((self.payoff.transient(t, V, S) == Vm).all())

    def test_conv_call_put(self):
        """Test conversion, call and put option,"""
        t = 4
        S = self.S
        V = self.V
        Vm = np.maximum(V, S)  # Normal conversion
        # Put option
        put = (V < 105) & (S < 105)
        self.assertTrue(put.any())
        Vm[put] = 105
        # Call option
        call = (V > 110) & (S <= 110)
        Vm[call] = 110
        self.assertTrue(call.any())
        # Conversion option (forced conversion)
        conv = (V > 110) & (S > 110) & (V > S)
        self.assertTrue(conv.any())
        Vm[conv] = S[conv]
        self.assertTrue((self.payoff.transient(t, V, S) == Vm).all())
        # Default
        Vd = np.maximum(S, 50)
        self.assertTrue((self.payoff.default(t, S) == Vd).all())

    def test_put(self):
        """Test put option, without conversion or call option."""
        t = 5
        S = self.S
        V = self.V
        Vm = np.maximum(V, 105)
        self.assertTrue((self.payoff.transient(t, V, S) == Vm).all())

    def test_put_call(self):
        """Test put and call option, without conversion option."""
        t = 6
        S = self.S
        V = self.V
        Vm = V.copy()
        # Put option
        put = (V < 105)
        Vm[put] = 105
        # Call option
        call = (V > 110)
        Vm[call] = 110
        self.assertTrue(call.any())
        self.assertTrue((self.payoff.transient(t, V, S) == Vm).all())

    def test_call(self):
        """Test call option, without conversion or put option."""
        t = 7
        S = self.S
        V = self.V
        Vm = np.minimum(V, 110)
        self.assertTrue((V > 110).any())
        self.assertTrue((self.payoff.transient(t, V, S) == Vm).all())

    def test_coupon(self):
        t = 8
        self.assertTrue((self.payoff.coupon(t) == 4).all())

    def test_redemption(self):
        """Test redemption of convertible bond."""
        S = self.S
        V = np.maximum(100, S)
        self.assertTrue((self.payoff.terminal(S) == V).all())


if __name__ == "__main__":
    unittest.main()
