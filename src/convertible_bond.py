import numpy as np

from model import WienerJumpProcess, BinomialModel, FDEModel, FDEBVModel
from payoff import Annuity, CallA, CallVR, PutV, Stack, Time, VariableStrike

__all__ = ["dS_total", "dS_partial", "B", "P", "C", "S", "payoff", "T"]

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

# Bond
#       Nominal value = 100
#       Semi-annual coupon = 4
#       Recovery factor = 0
class Annuity(Annuity):
    """Customised Annuity where coupon is payed as part of nominal."""
    def coupon(self, t):
        if t != self.T:
            return super(Annuity, self).coupon(t)
        return 0.0

    def terminal(self, S):
        V = super(Annuity, self).terminal(S)
        if self.T in self.times:
            V += self.C
        return V
B = Annuity(T, np.arange(0.5, T + 0.5, 0.5), 4, 100, 0)

# American put option on portfolio
#       Strike = 105
#       Time = 3
class Put(PutV, VariableStrike):
    def __init__(self, T, K, B):
        super(Put, self).__init__(T, K)
        self._B = B

    def transient(self, t, V, S):
        ti = 0
        accC = 0
        for i in self._B.times:
            if i > t:
                accC = self._B.C * (t - ti) / (i - ti)
                break
            ti = i
        with self._strike(self.K + accC):
            return super(Put, self).transient(t, V, S)
P = Time(Put(T, 105, B), [3])

# Reversed American call option on portfolio
#       Strike = 110
#       Time = [2, 5]
class Call(CallVR, VariableStrike):
    def __init__(self, T, K, B):
        super(Call, self).__init__(T, K)
        self._B = B

    def transient(self, t, V, S):
        ti = 0
        accC = 0
        for i in self._B.times:
            if i > t:
                accC = self._B.C * (t - ti) / (i - ti)
                break
            ti = i
        with self._strike(self.K + accC):
            return super(Call, self).transient(t, V, S)
C = Time(Call(T, 110, B), [(2, 5)])

# Stock option (conversion option into stock for portfolio)
S = CallA(T, 0)

# Convertible bond:
#       Bond
#       American put option on portfolio
#       Reversed American call option on portfolio
#       Stock
payoff = Stack([B, P, C, S])

if __name__ == "__main__":
    x = 8
    Sl = 0
    Su = 200
    S0 = 100
    N = 2**(x + 1) * T
    K = 2**((x + 1) // 2) * (Su - Sl)
    tol = 1. / 2**(2*x + 2)
    Sk = K * (S0 - Sl) / (Su - Sl)
    dS = dS_total
    print "N = %i; K = %i\n" % (N, K)
    print "BINOMIAL"
    print float(BinomialModel(N, dS, Stack([B, P, C, S])).price(S0))
    print "\nFDE"
    print FDEModel(N, dS, Stack([B, P, C, S])).price(Sl, Su, K).V[0][Sk]
    print "\nFDE_BE"
    print FDEBVModel(N, dS, Stack([B, P, C]), payoff).price(Sl, Su, K).V[0][Sk]
