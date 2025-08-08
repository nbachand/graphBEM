import numpy as np
from numpy import trapz

class VentilationSimulation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        expected_kwards = set(['H', 'W', "ventType", "alphas", "As", "Ls"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        
        # Constants
        self.rho = 1.225 #air density
        self.Cp = 1005  #specific heat capacity for air
        self.Vnv = None

        self.initialize()

    def initialize(self):
        if self.alphas is None:
            self.Cds = None
            return
        self.Cds = np.zeros(len(self.alphas))
        for i, alpha in enumerate(self.alphas):
            self.Cds[i] = self.get_Cd(np.radians(alpha))
        
    def get_Wpivot(self, z, alpha):
        h = self.H * (1 - np.cos(alpha))
        Wpivot = np.ones_like(z) * self.W
        zpivot = z[z > h]
        Wpivot[z > h] = (1 / self.W**2 + 1 / (2 * (self.H - zpivot) * np.tan(alpha) + np.sin(alpha) * self.W**2))**(-1/2)
        return Wpivot

    def get_Aeff(self, alpha):
        z = np.linspace(0, self.H, 1000)
        Wpivot = self.get_Wpivot(z, alpha)
        return trapz(Wpivot, z)

    def get_Cd(self, alpha):
        Cd0 = 0.611
        return self.get_Aeff(alpha) / (self.H * self.W) * Cd0

    def get_Vnvi(self, Cdi, Ai, Li, Tint, Tout):
        g = 9.8
        return Cdi * Ai * (2 * g * Li * np.abs((Tint - Tout) / Tout))**0.5

    def get_Vnv(self, Tint, Tout, t):
        Vnv = np.zeros((self.Cds.size, t.size))
        hours = t / 60 / 60
        day_hours = np.remainder(hours, 24)
        if isinstance(Tint, float):
            Tint = np.ones_like(hours) * Tint
        if isinstance(Tout, float):
            Tout = np.ones_like(hours) * Tout
        Tint = Tint[(day_hours < 7) | (day_hours > 12 + 7)]
        Tout = Tout[(day_hours < 7) | (day_hours > 12 + 7)]
        for i in range(len(self.As)):
            Vnv_7to7 = self.get_Vnvi(self.Cds[i], self.As[i], self.Ls[i], Tint, Tout)
            Vnv_i = Vnv[[i], :]
            Vnv_i[(day_hours < 7) | (day_hours > 12 + 7)] = Vnv_7to7
            Vnv[i, :] = Vnv_i
        return Vnv
        
    def qToEvt(self, q, Tout, Tint):
        return self.rho * self.Cp * q * (Tout - Tint)

    def timeStepHWP1(self, t):
        hour = t / 60 / 60
        Evt = -500
        if (hour % 24 > 7) and (hour % 24 < 19):  # daytime
            return Evt
        return 0
    
    def timeStepHWP4(self, t, Tint = 0, Tout = 0):
        self.Vnv = np.sum(self.get_Vnv(Tint, Tout, t), axis=0) #summing across windows
        Evt = self.qToEvt(self.Vnv, Tout, Tint)
        return Evt
    
    def timeStep(self, *args, **kwargs):
        if self.ventType == "HWP1":
            return self.timeStepHWP1(*args)
        if self.ventType == "HWP4":
            return self.timeStepHWP4(*args, **kwargs)
        if self.ventType == None:
            return 0
    



