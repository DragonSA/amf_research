"""
Unit tests for the model module.
"""
import numpy as np
np.seterr(divide="ignore")
import unittest
from scipy import stats
import scipy.sparse as sparse

from model import BinomialModel, FDEModel, WienerJumpProcess
from model import CrankNicolsonScheme, ExplicitScheme, ImplicitScheme
from payoff import Annuity, CallA, CallE, Forward, Stack

SCHEMES = (CrankNicolsonScheme, ExplicitScheme, ImplicitScheme)

class TestWienerJumpProcess(unittest.TestCase):
    """Test the Wiener Jump process."""

    def test_bad_values(self):
        """Test for bad input values."""
        r = 0.1
        s = 0.2
        l = 0.1
        e = 0.3
        self.assertRaises(ValueError, WienerJumpProcess, r, -s, l, e)
        self.assertRaises(ValueError, WienerJumpProcess, r, 0, l, e)
        self.assertRaises(ValueError, WienerJumpProcess, r, s, -l, e)
        WienerJumpProcess(0, s, l, e)
        WienerJumpProcess(-r, s, l, e)
        WienerJumpProcess(r, s, l, 0)
        WienerJumpProcess(r, s, l, -e)
        WienerJumpProcess(r, s, l, 1.1)

    def test_binomial(self):
        """Test parameters for the binomial model."""
        dS = WienerJumpProcess(0.1, 0.2, 0.1, 0.3)

        u, d, l, (pu, pd, po) = dS.binomial(1)
        self.assertGreater(u, np.exp(0.1))
        self.assertLess(d, np.exp(0.1))
        self.assertGreater(d, 0)

        self.assertGreaterEqual(l, 0)
        self.assertLessEqual(l, 1)

        self.assertGreaterEqual(min(pu, pd, po), 0)
        self.assertEqual(pu + pd + po, 1)

        dS.r = 1
        self.assertRaises(ValueError, dS.binomial, 0.2)

        dS.lambd_ = np.double('inf')
        self.assertRaises(ValueError, dS.binomial, 0.01)

        self.assertRaises(ValueError, dS.binomial, 0)
        self.assertRaises(ValueError, dS.binomial, -0.01)

    def test_binomial_variable_hazard(self):
        """Test variable hazard rate."""
        dS = WienerJumpProcess(0.1, 0.2, lambda S: (S / 100)**-1, cap_lambda=True)

        S = np.linspace(0, 200, 401)
        for pu, pd, po in zip(*dS.binomial(1, S)):
            self.assertGreaterEqual(min(pu, pd, po), 0)
            self.assertEqual(pu + pd + po, 1)
        dS.cap_lambda = False
        self.assertRaises(ValueError, dS.binomial, 1, S)

    def test_fde(self):
        """Test parameters for the finite difference model."""
        dS = WienerJumpProcess(0.1, 0.2, 0.1, 0.3)

        dt = 0.1
        S = np.linspace(0, 200, 401)
        ds = S[1] - S[0]
        for i in ("explicit", "implicit"):
            a, b, c = dS.fde(dt, ds, S, i)
            L = sparse.dia_matrix(([a, b, c], [-1, 0, 1]), shape=S.shape*2)
            self.assertTrue((np.abs(L.sum(1) + dS.r * dt) < 1e-13).all())


class TestBinomialModel(unittest.TestCase):
    """Test the Binomial Model."""

    def test_value(self):
        """Test the value object for the binomial model."""
        dS = WienerJumpProcess(0.1, 0.1, 0.02, 0)
        N = 16
        T = 1
        ct = np.linspace(0, T, N // 2 + 1)
        c = 1
        S = 100
        model = BinomialModel(N, dS, Stack([CallA(T, S), Annuity(T, ct, c)]))
        P = model.price(S)
        # Test N
        self.assertEqual(P.N, N)
        # Test time series
        self.assertEqual(P.t[0], 0)
        self.assertEqual(P.t[-1], T)
        self.assertEqual(len(P.t), N + 1)
        # Test Coupon sequence
        self.assertTrue((P.C[::2] == c).all())
        self.assertTrue((P.C[1::2] == 0).all())
        # Test price sequence
        self.assertEqual(P.S[0][0], S)
        self.assertEqual(len(P.S), N + 1)
        for i in range(1, N + 1):
            self.assertGreaterEqual(P.S[i][0], P.S[i - 1][0])
            self.assertLessEqual(P.S[i][-1], P.S[i - 1][-1])
            self.assertTrue((P.S[i][:-1] > P.S[i][1:]).all())
        # Test default sequence
        self.assertEqual(len(P.V), N + 1)
        for i in range(1, N + 1):
            self.assertGreaterEqual(P.V[i][0] - P.C[i], P.V[i - 1][0])
            self.assertLessEqual(P.V[i][-1] - P.C[i], P.V[i - 1][-1])
            self.assertTrue((P.V[i][:-1] >= P.V[i][1:]).all())
        self.assertEqual(len(P.X), N)
        for i in range(1, N):
            self.assertGreaterEqual(P.X[i][0], P.X[i - 1][0])
            self.assertLessEqual(P.X[i][-1], P.X[i - 1][-1])
            self.assertTrue((P.X[i][:-1] >= P.X[i][1:]).all())
        # Test coupon sequence

    def test_call(self):
        """Test that the binomial model correctly prices a call option."""
        def price(r):
            d1 = (np.log(100. / K) + r + 0.0005) / 0.1
            d2 = d1 - 0.1
            price = stats.norm.cdf(d1) * 100
            price -= stats.norm.cdf(d2) * K * np.exp(-r)
            return price

        dS = WienerJumpProcess(0.1, 0.1)
        dSdq = WienerJumpProcess(0.1, 0.1, 0.1)

        accuracy = 10
        step_down = set((59, 61, 64, 68, 72, 78, 83, 89, 100, 119))
        step_up = set((136, 158, 172, 183, 193))
        for K in range(1, 200):
            if K in step_down:
                accuracy -= 1
            elif K in step_up:
                accuracy += 1
            V = CallE(1, K)
            model = BinomialModel(128, dS, V)
            self.assertAlmostEqual(float(model.price(100)), price(0.1), accuracy)
            model = BinomialModel(128, dSdq, V)
            self.assertAlmostEqual(float(model.price(100)), price(0.2), accuracy)

    def test_forward(self):
        """Test that the binomial model correctly prices a forward contract."""
        dS = WienerJumpProcess(0.1, 0.1)
        dSdq = WienerJumpProcess(0.1, 0.1, 0.02)

        for K in range(200):
            V = Forward(1, K)
            model = BinomialModel(2, dS, V)
            self.assertAlmostEqual(float(model.price(100)), 100 - K * np.exp(-0.1))
            model = BinomialModel(2, dSdq, V)
            self.assertAlmostEqual(float(model.price(100)), 100 - K * np.exp(-0.12))

    def test_annuity(self):
        """Test the pricing of a series of payments."""
        def price(r):
            erdt = np.exp(-r)
            i = 1 / erdt - 1
            return (1 - erdt**(T * 2)) / i + 10 * erdt**(T * 2)

        dS = WienerJumpProcess(0.1, 0.1)
        dSdq = WienerJumpProcess(0.1, 0.1, 0.02)

        # Semi-annual payments of 1
        T = 10
        V = Annuity(T, np.arange(0.5, T + 0.5, 0.5), 1, 10)
        model = BinomialModel(T * 2, dS, V)
        self.assertAlmostEqual(float(model.price(100)), price(0.05))
        model = BinomialModel(T * 2, dSdq, V)
        self.assertAlmostEqual(float(model.price(100)), price(0.06))


class TestFDEModel(unittest.TestCase):
    """Test the finite difference Model."""

    def test_value(self):
        """Test the value object for the finite difference model."""
        dS = WienerJumpProcess(0.1, 0.1, 0.02, 0)
        N = 16
        T = 1
        ct = np.linspace(0, T, N // 2 + 1)
        c = 1
        Sl = 0
        Su = 200
        K = 40
        model = FDEModel(N, dS, Stack([CallA(T, 100), Annuity(T, ct, c)]))
        P = model.price(Sl, Su, K)
        S = P.S
        # Test N
        self.assertEqual(P.N, N)
        # Test time series
        self.assertEqual(P.t[0], 0)
        self.assertEqual(P.t[-1], T)
        self.assertEqual(len(P.t), N + 1)
        # Test Coupon sequence
        self.assertTrue((P.C[::2] == c).all())
        self.assertTrue((P.C[1::2] == 0).all())
        # Test K
        self.assertEqual(P.K, K)
        # Test Stock series
        self.assertEqual(P.S[0], Sl)
        self.assertEqual(P.S[-1], Su)
        self.assertEqual(len(P.S), K + 1)
        # Test default sequence
        self.assertEqual(len(P.V), N + 1)
        for i in range(1, N + 1):
            self.assertLessEqual(P.V[i][0] - P.C[i], P.V[i - 1][0])
            self.assertLessEqual(P.V[i][-1] - P.C[i], P.V[i - 1][-1])
            self.assertTrue((P.V[i][:-1] - 1e14 <= P.V[i][1:]).all())
        self.assertEqual(len(P.X), N)
        #for i in range(1, N):
            #self.assertGreaterEqual(P.X[i][0], P.X[i - 1][0])
            #self.assertLessEqual(P.X[i][-1], P.X[i - 1][-1])
            #self.assertTrue((P.X[i][:-1] >= P.X[i][1:]).all())

    def test_call(self):
        """Test that the finite difference model correctly prices a call option."""
        K = 100
        def price(r, S):
            d1 = (np.log(S / K) + r + 0.0005) / 0.1
            d2 = d1 - 0.1
            price = stats.norm.cdf(d1) * S
            price -= stats.norm.cdf(d2) * K * np.exp(-r)
            return price

        dS = WienerJumpProcess(0.1, 0.1)
        #dSdq = WienerJumpProcess(0.1, 0.1, 0.1)

        S = np.linspace(0, 200, 41)
        V = CallE(1, K)
        model = FDEModel(64, dS, V)
        accuracy = np.array((15, 15, 15, 14, 12, 11, 9, 8, 6, 5, 4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2))
        for scheme in SCHEMES:
            P = model.price(0, 200, 40, scheme=scheme)
            self.assertTrue((abs(P.V[0] - price(0.1, S)) < 10.0**-accuracy).all())
        return

    def test_forward(self):
        """Test that the finite difference model correctly prices a forward contract."""
        dS = WienerJumpProcess(0.1, 0.1)
        #dSdq = WienerJumpProcess(0.1, 0.1, 0.02)

        S = np.linspace(0, 200, 41)
        V = Forward(1, 100)
        model = FDEModel(64, dS, V)
        for scheme in SCHEMES:
            P = model.price(0, 200, 40, scheme=scheme)
            if scheme is CrankNicolsonScheme:
                accuracy = 5
            else:
                accuracy = 2
            self.assertTrue((abs(P.V[0] - (S - 100 * np.exp(-0.1))) < 10.0**-accuracy).all())

    def test_annuity(self):
        """Test the pricing of a series of payments."""
        def price(r):
            erdt = np.exp(-r)
            i = 1 / erdt - 1
            return (1 - erdt**(T * 2)) / i + 10 * erdt**(T * 2)

        dS = WienerJumpProcess(0.1, 0.1)
        #dSdq = WienerJumpProcess(0.1, 0.1, 0.02)

        # Semi-annual payments of 1
        T = 10
        V = Annuity(T, np.arange(0.5, T + 0.5, 0.5), 1, 10)
        model = FDEModel(T * 8, dS, V)
        for scheme in SCHEMES:
            P = model.price(0, 200, 25, scheme=scheme)
            self.assertTrue((np.abs(P.V[0] - price(0.05)) < 1e-1).all())


if __name__ == "__main__":
    unittest.main()
