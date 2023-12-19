class RoomSimulation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        expected_kwards = set(["T0", "V", "Eint"])
        if set(kwargs.keys()) != expected_kwards:
            raise Exception(f"Invalid keyword arguments, expected {expected_kwards}")
        # Constants
        self.rho = 1.225 #air density
        self.Cp = 1005  #specific heat capacity for air

    def initialize(self, delt):
        self.Tint = self.T0

        # Scaling factors
        self.lambda_int = delt / (self.rho * self.Cp * self.V)

    def timeStep(self, Ef, Evt):
        self.Tint = self.Tint + self.lambda_int * (Ef + self.Eint + Evt)
        return self.Tint