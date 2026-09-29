import numpy as np
from numpy import trapezoid

class VentilationSimulation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        expected_kwards = set(['H', 'W', "ventType", "alphas", "As", "Ls"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        
        # Constants
        self.rho = 1.225 #air density
        self.Cp = 1005  #specific heat capacity for air
        self.Vnv = 0.0

        self.initialize()

    def initialize(self):
        if self.alphas is None:
            self.Cds = None
            return
        self.Cds = np.zeros(len(self.alphas))
        for i, alpha in enumerate(self.alphas):
            self.Cds[i] = self.get_Cd(np.radians(alpha))
        
    def get_Wpivot(self, z, alpha):
        """Effective width for a top-hung window, angle in radians.

        Hult, Iaccarino & Fischer (SimBuild 2012), Eq. 4:
        https://publications.ibpsa.org/proceedings/simbuild/2012/papers/simbuild2012_05b_3_Hult.pdf
        The entire side-plus-bottom opening width is squared.
        """
        if not np.isfinite(alpha) or not 0 <= alpha <= np.pi/2:
            raise ValueError("Window angle must be between 0 and 90 degrees")
        if self.H <= 0 or self.W <= 0:
            raise ValueError("Window dimensions must be positive")
        z = np.asarray(z, dtype=float)
        if alpha == 0:
            return np.zeros_like(z)
        if alpha == np.pi/2:
            return np.full_like(z, self.W)
        h = self.H * (1 - np.cos(alpha))
        opening = 2 * (self.H-z) * np.tan(alpha) + np.sin(alpha)*self.W
        effective = self.W * opening / np.hypot(self.W, opening)
        return np.where(z > h, effective, self.W)

    def get_Aeff(self, alpha):
        self.get_Wpivot(np.array([0.0]), alpha)  # validate angle and dimensions
        if alpha == 0:
            return 0.0
        if alpha == np.pi/2:
            return self.H*self.W
        h = self.H*(1-np.cos(alpha))
        # Split at the geometry breakpoint. A uniform grid spanning both parts
        # gives a spurious finite area as the opening angle approaches zero.
        z = np.linspace(np.nextafter(h, self.H), self.H, 1000)
        return h*self.W + trapezoid(self.get_Wpivot(z, alpha), z)

    def get_Cd(self, alpha):
        Cd0 = 0.611
        return self.get_Aeff(alpha) / (self.H * self.W) * Cd0

    def get_Vnvi(self, Cdi, Ai, Li, Tint, Tout):
        g = 9.8
        return Cdi * Ai * (2 * g * Li * np.abs((Tint - Tout) / Tout))**0.5

    def get_Vnv(self, Tint, Tout, t):
        """Return window-by-time volume flows, accepting scalar or 1D inputs."""
        times = np.atleast_1d(np.asarray(t, dtype=float))
        if times.ndim != 1:
            raise ValueError("Ventilation times must be scalar or one-dimensional")
        inside = np.broadcast_to(np.asarray(Tint, dtype=float), times.shape)
        outside = np.broadcast_to(np.asarray(Tout, dtype=float), times.shape)
        if np.any(inside <= 0) or np.any(outside <= 0):
            raise ValueError("Ventilation temperatures must be in kelvin")
        if self.Cds is None or not (len(self.Cds) == len(self.As) == len(self.Ls)):
            raise ValueError("Each window needs an angle, area and stack height")
        day_hours = np.remainder(times / 3600, 24)
        open_window = (day_hours < 7) | (day_hours > 19)
        flow = np.zeros((len(self.Cds), times.size))
        for i, (cd, area, height) in enumerate(zip(self.Cds, self.As, self.Ls)):
            if area < 0 or height < 0:
                raise ValueError("Window area and stack height must be nonnegative")
            flow[i, open_window] = self.get_Vnvi(cd, area, height,
                                               inside[open_window], outside[open_window])
        return flow

    def qToEvt(self, q, Tout, Tint):
        return self.rho * self.Cp * q * (Tout - Tint)

    def timeStepHWP1(self, t):
        hour = t / 60 / 60
        Evt = -500
        if (hour % 24 > 7) and (hour % 24 < 19):  # daytime
            return Evt
        return 0
    
    def timeStepHWP4(self, t, Tint = 0, Tout = 0):
        self.Vnv = np.sum(self.get_Vnv(Tint, Tout, t), axis=0)
        if np.ndim(t) == 0:
            self.Vnv = float(self.Vnv[0])
        Evt = self.qToEvt(self.Vnv, np.asarray(Tout), np.asarray(Tint))
        return Evt
    
    def timeStep(self, *args, **kwargs):
        if self.ventType == "HWP1":
            return self.timeStepHWP1(*args)
        if self.ventType == "HWP4":
            return self.timeStepHWP4(*args, **kwargs)
        if self.ventType == None:
            return 0
    



