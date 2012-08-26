import numpy as np

from model import WienerJumpProcess, BinomialModel, Payoff
from payoff import Annuity, CallA, CallVR, PutV, Stack, Time

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
P = Time(PutV(T, 105), [3])

# Reversed American call option on portfolio
#       Strike = 110
#       Time = [2, 5]
C = Time(CallVR(T, 110), [(2, 5)])

# Stock option (conversion option into stock for portfolio)
S = CallA(T, 0)

# Convertible bond:
#       Bond
#       American put option on portfolio
#       Reversed American call option on portfolio
#       Stock
payoff = Stack([B, P, C, S])

if __name__ == "__main__":
    N = 200
    dS = dS_total
    S0 = 100
    print BinomialModel(N, dS, Stack([B])).price(S0), "(B)"
    print BinomialModel(N, dS, Stack([B, S])).price(S0), "(B+S)"
    print BinomialModel(N, dS, Stack([B, C, S])).price(S0), "(B+C+S)"
    print BinomialModel(N, dS, Stack([B, P, S])).price(S0), "(B+P+S)"
    print BinomialModel(N, dS, Stack([B, P, C, S])).price(S0), "(B+P+C+S)"
