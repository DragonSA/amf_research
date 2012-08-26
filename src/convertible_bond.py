import numpy as np

from model import WienerJumpProcess, BinomialModel, Payoff
from payoff import Annuity, CallA, CallVR, PutV, Stack

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
B = Annuity(T, np.arange(0.5, T + 0.5, 0.5), 4, 100, 0)

# American put option on portfolio
#       Strike = 105
#       Time = 3
P = PutV(T, 105)

# Reversed American call option on portfolio
#       Strike = 110
#       Time = [2, 5]
C = CallVR(T, 110)

# Stock option (conversion option into stock for portfolio)
S = CallA(T, 0)

# Convertible bond:
#       Bond
#       American put option on portfolio
#       Reversed American call option on portfolio
#       Stock
payoff = Stack([B, P, C, S])

print "B, B+P, B+P+C, B+P+C+S"
for N in (200, 400, 800, 1600, 3200):
    prices = []
    for i in range(1, len(payoff.stack) + 1):
        model = BinomialModel(N, dS_total, Stack(payoff.stack[:i]))
        prices.append(model.price(100))
    print ", ".join(str(i) for i in prices)
