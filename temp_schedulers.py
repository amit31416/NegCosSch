import math
import random

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


class LogarithmicIncreaseScheduler:
    def __init__(self, tau_plus=0.1, tau_minus=0.01, epochs=600):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.epochs = epochs
        self.scale = (self.tau_plus - self.tau_minus) / math.log(self.epochs + 1)

    def get_temperature(self, t):
        if self.epochs <= 0:
            return self.tau_plus
        if t >= self.epochs:
            return self.tau_plus
        if t <= 0:
            return self.tau_minus
        return self.tau_minus + self.scale * math.log(t)

class ExponentialIncreaseScheduler:
    def __init__(self, tau_plus=0.1, tau_minus=0.01, epochs=600):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus # Must be > 0 for exponential growth
        self.epochs = epochs
        self.growth_rate = (self.tau_plus / self.tau_minus) ** (1 / self.epochs)
        
    def get_temperature(self, t):
        if self.epochs <= 0:
            return self.tau_plus
        if t >= self.epochs:
            return self.tau_plus
        if t <= 0:
            return self.tau_minus

        return self.tau_minus * (self.growth_rate ** t)


class RandomScheduler:
    def __init__(self, tau_plus=0.2,tau_minus=0.05, T=200):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

    def get_temperature(self,step):
        temperature= random.uniform(self.tau_minus, self.tau_plus)
        return temperature

class LinearScheduler:
    def __init__(self, tau_plus=0.1, tau_minus=0.1/1.5, T=600, epochs=600):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.epochs = epochs
        self.T = T

    def get_temperature(self, step):
        t = step % int(self.T)
        return self.tau_minus + (self.tau_plus - self.tau_minus) * (t / self.epochs)


class LinearDecreasingScheduler:
    def __init__(self, tau_plus=0.1, tau_minus=0.1/1.5, T=600, epochs=600):
        self.tau_plus = tau_plus    
        self.tau_minus = tau_minus
        self.epochs = epochs
        self.T = T

    def get_temperature(self, step):
        t = step % int(self.T)
        return self.tau_plus - (self.tau_plus - self.tau_minus) * (t / self.epochs)

class M_NegCosineTemperatureScheduler:
    def __init__(self, tau_plus=0.4,tau_minus=0.1, T=1200):
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.T = T
    def get_temperature(self,step):

        cycle_step = step % int(self.T/2)  
        temperature = self.tau_minus + 0.5 *(self.tau_plus - self.tau_minus) * (1 + math.cos(2*math.pi * (cycle_step-self.T/2) / self.T))

        return temperature

