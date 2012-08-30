"""
Unit tests for the model module.
"""
import numpy as np
np.seterr(divide="ignore")
import math
import unittest
from scipy import stats

from model import BinomialModel, WienerJumpProcess
from payoff import Annuity, CallE, Forward

class TestWienerJumpProcess(unittest.TestCase):
    """Test the Wiener Jump process."""

    def test_bad_values(self):
        """Test for bad input values."""
        r = 0.1
        s = 0.2
        l = 0.1
        e = 0.3
        self.assertRaises(ValueError, WienerJumpProcess, -r, s, l, e)
        self.assertRaises(ValueError, WienerJumpProcess, r, -s, l, e)
        self.assertRaises(ValueError, WienerJumpProcess, r, s, -l, e)
        self.assertRaises(ValueError, WienerJumpProcess, r, s, l, -e)
        self.assertRaises(ValueError, WienerJumpProcess, r, s, l, 1.1)

    def test_binomial(self):
        """Test parameters for the binomial model."""
        dS = WienerJumpProcess(0.1, 0.2, 0.1, 0.3)

        u, d, l, (pu, pd, po) = dS.binomial(1)
        self.assertGreater(u, math.exp(0.1))
        self.assertLess(d, math.exp(0.1))
        self.assertGreater(d, 0)

        self.assertGreaterEqual(l, 0)
        self.assertLessEqual(l, 1)

        self.assertGreaterEqual(min(pu, pd, po), 0)
        self.assertEqual(pu + pd + po, 1)

    def test_binomial_variable_hazard(self):
        """Test variable hazard rate."""
        dS = WienerJumpProcess(0.1, 0.2, lambda S: (S / 100)**-1, cap_lambda=True)

        S = np.linspace(0, 200, 401)
        for pu, pd, po in zip(*dS.binomial(1, S)):
            self.assertGreaterEqual(min(pu, pd, po), 0)
            self.assertEqual(pu + pd + po, 1)
        dS.cap_lambda = False
        self.assertRaises(ValueError, dS.binomial, 1, S)


class TestBinomialModel(unittest.TestCase):
    """Test the Binomial Model."""

    def test_value(self):
        """Test the value object for the binomial model."""
        # TODO

    def test_call(self):
        """Test that the binomial model correctly prices a call option."""
        def price(r):
            d1 = (np.log(100. / K) + r + 0.0005) / 0.1
            d2 = d1 - 0.1
            price = stats.norm.cdf(d1) * 100
            price -= stats.norm.cdf(d2) * K * math.exp(-r)
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
            self.assertAlmostEqual(float(model.price(100)), 100 - K * math.exp(-0.1))
            model = BinomialModel(2, dSdq, V)
            self.assertAlmostEqual(float(model.price(100)), 100 - K * math.exp(-0.12))

    def test_annuity(self):
        """Test the pricing of a series of payments."""
        def price(r):
            erdt = math.exp(-r)
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


if __name__ == "__main__":
    unittest.main()
