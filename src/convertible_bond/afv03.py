"""
Convertible bond parameters as used by AFV03.
"""
from __future__ import absolute_import

import numpy as np

from model import WienerJumpProcess
from payoff import CallA, Stack, Time
from convertible_bond.payoff import Annuity, Call, Put

__all__ = [
        "T",
        "dS", "dS_total", "dS_partial", "dS_var12", "dS_var20",
        "A", "P", "C", "S", "B", "E", "payoff",
    ]

# Time till maturity = 5 years
T = 5

# Stock price is a Wiener process with default jump.
#       Drift rate = 5%
#       Volatility = 20%
#       Hazard rate = 2%
# Total default (default = 100%)
dS_total = WienerJumpProcess(r=0.05, sigma=0.2, lambd_=0.02, eta=1)
# Partial default (default = 0%)
dS_partial = WienerJumpProcess(r=0.05, sigma=0.2, lambd_=0.02, eta=0)
# No default
dS = WienerJumpProcess(r=0.05, sigma=0.2)
# Variable hazard (a=-1.2), total default
dS_var12 = WienerJumpProcess(r=0.05, sigma=0.2, lambd_=lambda S: 0.02 * (S / 100)**-1.2, eta=1)
# Variable hazard (a=-2.0), total default
dS_var20 = WienerJumpProcess(r=0.05, sigma=0.2, lambd_=lambda S: 0.02 * (S / 100)**-2.0, eta=1)

# Bond
#       Nominal value = 100
#       Semi-annual coupon = 4
#       Recovery factor = 0
A = Annuity(T, np.arange(0.5, T + 0.5, 0.5), C=4, N=100, R=0)

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
