import math

class GCosineTemperatureSchedulerM:
    def __init__(self, tau_plus=2.0,tau_minus=0.5, T=200,shift=1.0,epochs=600):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.epochs = epochs
        self.T = T
        self.s = shift
        self.e = int(self.epochs - 0.5 * self.s * self.T)
    def get_temperature(self, t):
        if(t<self.e):
          cos_t = (self.tau_plus - self.tau_minus) * (1 + math.cos(2 * math.pi * (t-self.s * self.T/2) / self.T)) / 2 + self.tau_minus
        else:
            cos_t = self.tau_plus
        return cos_t
