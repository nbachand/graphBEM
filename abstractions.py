import numpy as np
import scipy.linalg as sp_linalg
from matplotlib import pyplot as plt
from scipy.integrate import trapz

class BuildingSimulation():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        expected_kwards = set(["delt", "simLength", "Tout"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        self.t = 0 #time (seconds)
        self.hour = 0 #time (hours)
        self.times = np.arange(0, self.simLength + self.delt, self.delt)
        self.hours = self.times / 60 / 60
        self.N = len(self.times)
        if type(self.Tout) == float:
            self.Tout = np.ones_like(self.times) * self.Tout
        elif self.Tout is None:
            self.Tout =  np.zeros_like(self.times)

    def initialize(self, room_kwargs, wall_kwargs, vent_kwargs):
        self.room = RoomSimulation(**room_kwargs)
        self.wall = WallSimulation(**wall_kwargs)
        self.vent = VentilationSimulation(**vent_kwargs)
        self.room.initialize(self.delt)
        self.wall.initialize(self.delt)


    def run(self):
        Tints = np.zeros(self.N) # initializing interior air temp vector
        Vnvs = np.zeros(self.N) # initializing ventilation energy vector
        Tint = self.room.Tint
        Tints[0] = Tint

        T_profs = np.zeros((self.wall.n + 2, self.N)) # intializing matrix to store 6 temperature profiles
        T_profs[:, 0] = self.wall.getWallProfile(Tint)
        nWalls = 3 #number of walls

        for i in range(1, self.N):
            self.t = self.times[i]
            self.hour = self.t / 60 / 60

            # Simulation logic
            Ef = self.wall.timeStep(self.room.Tint)
            Evt = self.vent.timeStep(self.t, Tint = self.room.Tint, Tout = self.Tout[i])
            self.room.timeStep(nWalls * (Ef.front + Ef.back), Evt)

            T_profs[:,i] = self.wall.T_prof
            Tints[i] = self.room.Tint
            Vnvs[i] = self.vent.Vnv
        return Tints, T_profs, Vnvs


class WallFlux:
    def __init__(self):
        self.front = None
        self.back = None

class RoomSimulation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        expected_kwards = set(["T0"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        # Constants
        self.rho = 1.225 #air density
        self.Cp = 1005  #specific heat capacity for air
        self.V = 2880 #volume of air
        self.Eint = 250 #internal heat generation

    def initialize(self, delt):
        self.Tint = self.T0

        # Scaling factors
        self.lambda_int = delt / (self.rho * self.Cp * self.V)

    def timeStep(self, Ef, Evt):
        self.Tint = self.Tint + self.lambda_int * (Ef + self.Eint + Evt)
        return self.Tint
        


class WallSimulation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        expected_kwards = set(["Tf0"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        # Constants
        self.rhof = 2300 #density of fabric
        self.Cf = 750 #specific heat capacity of fabric
        self.kf = 0.8 #thermal conductivity of fabric
        self.Af = 90 #fabric area
        self.th = 0.10 #fabric thickness 
        self.h = 4 #fabric convection coefficient
        self.delx = 0.010 #spatial discretization size
        self.x = np.arange(0, self.th + self.delx, self.delx)

    def initialize(self, delt):
        # Scaling factors
        self.lambda_val = (self.kf * delt) / (self.rhof * self.Cf * self.delx**2)
        self.lambda_bound = self.kf / (self.h * self.delx)

        # Wall setup
        self.n = round(self.th / self.delx) - 1
        r = np.zeros(self.n)
        r[0] = 1 - 2 * self.lambda_val
        r[1] = self.lambda_val
        r[-1] = self.lambda_val

        A_matrix = sp_linalg.toeplitz(r)
        A_matrix[0, 0] += self.lambda_val * self.lambda_bound / (1 + self.lambda_bound)
        A_matrix[-1, -1] = A_matrix[0, 0]
        A_matrix[-1, 0] = 0
        A_matrix[0, -1] = 0

        self.A = A_matrix

        self.b = np.zeros(self.n)
        self.T = np.ones(self.n) * self.Tf0 #initializing constant wall temp equal to initial fabric temp

    def timeStep(self, Tint):
        self.b[0] = self.lambda_val * Tint / (1 + self.lambda_bound)
        self.b[-1] = self.b[0]
        self.T = np.dot(self.A, self.T) + self.b
        self.T_prof = self.getWallProfile(Tint)

        Ef = WallFlux()
        Ef.front = self.Af * (self.T_prof[1] - self.T_prof[0]) / self.delx
        Ef.back = self.Af * (self.T_prof[-2] - self.T_prof[-1]) / self.delx
        return Ef

    def getWallProfile(self, Tint):
        T_prof = np.zeros(self.n + 2)
        T_prof[1:-1] = self.T
        T_prof[0] = self.get_Tf(self.T[0], Tint, self.lambda_bound)
        T_prof[-1] = self.get_Tf(self.T[-1], Tint, self.lambda_bound)
        return T_prof

    def get_Tf(self, T1, Tint, lambda_bound):
        return (lambda_bound * T1 + Tint) / (1 + lambda_bound)

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
        if type(Tint) == float:
            Tint = np.ones_like(hours) * Tint
        if type(Tout) == float:
            Tout = np.ones_like(hours) * Tout
        Tint = Tint[(day_hours < 7) | (day_hours > 12 + 7)]
        Tout = Tout[(day_hours < 7) | (day_hours > 12 + 7)]
        for i in range(len(self.As)):
            Vnv_7to7 = self.get_Vnvi(self.Cds[i], self.As[i], self.Ls[i], Tint, Tout)
            Vnv_i = Vnv[[i], :]
            Vnv_i[(day_hours < 7) | (day_hours > 12 + 7)] = Vnv_7to7
            Vnv[i, :] = Vnv_i
        return Vnv

    def timeStepHWP1(self, t):
        hour = t / 60 / 60
        Evt = -500
        if (hour % 24 > 7) and (hour % 24 < 19):  # daytime
            return Evt
        return 0
    
    def timeStepHWP4(self, t, Tint = 0, Tout = 0):
        self.Vnv = np.sum(self.get_Vnv(Tint, Tout, t), axis=0)
        Evt = self.rho * self.Cp * self.Vnv * (Tout - Tint)
        return Evt
    
    def timeStep(self, *args, **kwargs):
        if self.ventType == "HWP1":
            return self.timeStepHWP1(*args)
        if self.ventType == "HWP4":
            return self.timeStepHWP4(*args, **kwargs)