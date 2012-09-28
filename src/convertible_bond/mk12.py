"""
Convertible bond parameters as used by MK12.
"""
from __future__ import absolute_import

import numpy as np

from model import WienerJumpProcess
from payoff import CallA, Stack, Time
from convertible_bond.payoff import Annuity, Call, Put

__all__ = [
        "T",
        "dS", "dS_var",
        "A", "P", "C", "S", "B", "E", "payoff",
    ]

# Time till maturity = 5 years
T = 5

# Stock price is a Wiener process with default jump.
#       Drift rate = 5%
#       Volatility = 25%
#       Hazard rate = 6.2%
# Total default (default = 100%)
dS = WienerJumpProcess(r=0.05, sigma=0.25, lambd_=0.062, eta=1)
# Variable hazard (a=-1.2), total default
dS_var = WienerJumpProcess(r=0.05, sigma=0.25, lambd_=lambda S: 0.062 * (S / 50)**-0.5, eta=1)

# Bond
#       Nominal value = 100
#       Semi-annual coupon = 4
#       Recovery factor = 40%
A = Annuity(T, np.arange(0.5, T + 0.5, 0.5), C=4, N=100, R=0.4)

# American put option on portfolio
#       Strike = 105
#       Time = 3
P = Time(Put(T, 105, B), times=[3])

# Reversed American call option on portfolio
#       Strike = 110
#       Time = [2, 5]
C = Time(Call(T, 110, B), times=[(2, 5)])

# Stock option (conversion option into stock for portfolio)
S = CallA(T, 0)

# Bond component of convertible bond
B = Stack([A, P, C])

# Equity component of convertible bond
E = S

# Convertible bond:
#       Bond
#       American put option on portfolio
#       Reversed American call option on portfolio
#       Stock
payoff = Stack([A, P, C, S])
