"""
Unit tests for the model module.
"""
import math
import unittest

from model import WienerJumpProcess

class TestWienerJumpProcess(unittest.TestCase):
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

        u, d, l, pu, pd, po = dS.binomial(1)
        self.assertGreater(u, math.exp(0.1))
        self.assertLess(d, math.exp(0.1))
        self.assertGreater(d, 0)

        self.assertGreaterEqual(l, 0)
        self.assertLessEqual(l, 1)

        self.assertGreaterEqual(max(pu, pd, po), 0)
        self.assertEqual(pu + pd + po, 1)

if __name__ == "__main__":
    unittest.main()
