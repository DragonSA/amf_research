import numpy as np

from model import WienerJumpProcess, BinomialModel, Payoff
from payoff import Annuity, CallA

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

# Bond
#       Nominal value = 100
#       Semi-annual coupon = 4
#       Recovery factor = 0
C = Annuity(T, np.arange(0.5, T + 0.5, 0.5), 4, 100, 0)

# Call option (target for conversion of bond)
S = CallA(T, 0)

# Conversion ratio of 1
kappa = 1

for N in (200, 400, 800, 1600, 3200):
    model = BinomialModel(N, dS_total, C)
    print N, model.price(100)
