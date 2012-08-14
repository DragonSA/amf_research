"""
Framework for modelling payoff processes using either a binomial or finite-
difference model.
"""
import math

__all__ = ["WienerJumpProcess"]

class WienerJumpProcess(object):
    """
    A model of stochastic stock movement using a Wiener drift process to model
    volatility and a Poisson jump process to model default.

    The stock price follows:
        dS = (r + \lambda \eta) S_t dt + \sigma S_t dW_t - \eta \S_t dq_t
    where:
        dS      is the instantaneous stock movement
        dt      is the instantaneous change in time
        dW_t    is the Wiener drift process
        dq_t    is the Poisson jump process with intensity \lambda

        r       is the risk free rate
        \lambda is the hazard rate
        \eta    is the drop in stop price on a default event
        \sigma  is the log-volatility of the stock price
    """

    def __init__(self, r, sigma, lambd_, eta):
        if min(r, sigma, lambd_, eta) < 0:
            raise ValueError("all parameters must be non-negative")
        if eta > 1:
            raise ValueError("eta must be between 0 and 1 inclusive")
        self.r = r
        self.sigma = sigma
        self.lambd_ = lambd_
        self.eta = eta

    def binomial(self, dt):
        """Parameters for the binomial model."""
        u = math.exp(self.sigma * math.sqrt(dt))
        d = 1 / u
        l = 1 - self.eta
        po = 1 - math.exp(-self.lambd_ * dt)
        pu = (math.exp(self.r * dt) - d * (1 - po) - l * po) / (u - d)
        pd = 1 - pu - po
        return (u, d, l, pu, pd, po)
