import numpy as np
from model.utils import WallSides


def convectionDOE2(h_nat, V, R_f):
    """Legacy averaged wind correlation, retained for reproducibility."""
    alpha = np.mean([2.38, 2.86])
    beta = np.mean([0.617, 0.89])
    return (1 - R_f) * h_nat + R_f * (h_nat**2 + (alpha * V**beta)**2)**0.5


def processMaterials(material_df, n, dt=None, verbose=True):
    """Assign cells to layers without changing physical material properties.

    ``n`` sets a target cell width from the total solid thickness. Each solid
    layer has at least one cell, so the actual count can differ from ``n``.
    Air gaps retain their specified resistance and have no storage cells.
    ``dt`` and ``verbose`` remain accepted for existing notebook calls; material
    heat capacity is never increased to accommodate an explicit timestep.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    material_df = material_df.copy()
    gaps = material_df['key'].eq('Material:AirGap').to_numpy()
    solids = ~gaps
    if not solids.any():
        raise ValueError("A wall must contain at least one solid material layer")
    for field in ['Thickness', 'Conductivity', 'Density', 'Specific_Heat']:
        values = material_df[field].to_numpy(dtype=float)[solids]
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError(f"Solid material {field} must be finite and positive")
    gap_r = material_df['Thermal_Resistance'].to_numpy(dtype=float)[gaps]
    if not np.all(np.isfinite(gap_r) & (gap_r > 0)):
        raise ValueError("Air gap thermal resistance must be finite and positive")
    thickness = material_df['Thickness'].to_numpy(dtype=float)
    target_width = thickness[solids].sum() / n
    counts = np.zeros(len(material_df), dtype=int)
    counts[solids] = np.maximum(1, np.rint(thickness[solids] / target_width)).astype(int)
    resistance = material_df['Thermal_Resistance'].to_numpy(dtype=float, copy=True)
    resistance[solids] = thickness[solids] / material_df['Conductivity'].to_numpy(dtype=float)[solids]
    material_df['n'] = counts
    material_df['Thermal_Resistance'] = resistance
    material_df['depth'] = material_df['Thickness'].fillna(0).cumsum()
    return material_df


class WallSimulation:
    """One-dimensional finite-volume wall with massless surface balances.

    Temperatures in ``T`` are cell-center values. ``T_prof`` and ``x`` include
    both surface values for the building and plotting interfaces. Conductances
    and fluxes are per unit area; ``timeStep`` returns convective power to the
    adjoining air in W (end-of-step for implicit, start-of-step for explicit).
    Applied radiation is positive into each surface.
    """

    def __init__(self, **kwargs):
        expected = {'X', 'Y', 'material_df', 'h', 'roughness', 'absorptivity',
                    'n', 'delt', 'implicit'}
        optional = {"convection", "tilt", "azimuth", "wind_exposure"}
        if not expected <= set(kwargs) or set(kwargs) - expected - optional:
            raise ValueError(f"Invalid keyword arguments, expected {expected}")
        self.__dict__.update(kwargs)
        self.convection = kwargs.get("convection", WallSides("legacy", "legacy"))
        self.tilt = kwargs.get("tilt", WallSides(90., 90.))
        self.azimuth = kwargs.get("azimuth", WallSides(None, None))
        self.wind_exposure = kwargs.get("wind_exposure", WallSides("directional", "directional"))
        self.windDirection = None
        for side in ("front", "back"):
            if getattr(self.convection, side) not in {"legacy", "fixed"}:
                raise ValueError("Unknown convection model")
        self.Af = self.X * self.Y
        self.processMaterialDict(self.material_df)

    def processMaterialDict(self, material_df, verbose=False):
        self.material_df = processMaterials(material_df, self.n, verbose=verbose)
        widths, centers, resistance_centers, ks, densities, heat_capacities = [], [], [], [], [], []
        depth = resistance = 0.0
        for _, layer in self.material_df.iterrows():
            thickness = 0.0 if np.isnan(layer['Thickness']) else layer['Thickness']
            count = int(layer['n'])
            if count:
                dx = thickness / count
                for j in range(count):
                    widths.append(dx)
                    centers.append(depth + (j + 0.5) * dx)
                    resistance_centers.append(resistance + (j + 0.5) * dx / layer['Conductivity'])
                    ks.append(layer['Conductivity'])
                    densities.append(layer['Density'])
                    heat_capacities.append(layer['Specific_Heat'])
            depth += thickness
            resistance += layer['Thermal_Resistance']
        self.n = len(widths)
        self.th = depth
        self.x = np.r_[0.0, centers, depth]
        self.cell_widths = np.array(widths)
        self.kfs = np.array(ks)
        self.rhofs = np.array(densities)
        self.Cfs = np.array(heat_capacities)
        self.capacity = self.cell_widths * self.rhofs * self.Cfs  # J/(m2 K)
        self.total_resistance = resistance
        # Half-cell resistances plus any intervening massless air gaps.
        self.link_conductance = 1 / np.diff(resistance_centers)
        self.surface_conductance = WallSides(
            1 / resistance_centers[0],
            1 / (resistance - resistance_centers[-1]))
        self.K = np.zeros((self.n, self.n))
        for i, conductance in enumerate(self.link_conductance):
            self.K[i, i] += conductance
            self.K[i + 1, i + 1] += conductance
            self.K[i, i + 1] -= conductance
            self.K[i + 1, i] -= conductance

    def convection_metadata(self):
        return {side: dict(model=getattr(self.convection, side),
                          h_natural=getattr(self.h, side),
                          tilt=getattr(self.tilt, side), azimuth=getattr(self.azimuth, side),
                          wind_exposure=getattr(self.wind_exposure, side))
                for side in ("front", "back")}

    def _update_convection(self, front_air, back_air):
        values = []
        for side, air, index in [("front", front_air, 0), ("back", back_air, -1)]:
            model = getattr(self.convection, side)
            h = getattr(self.h, side)
            if model == "legacy":
                h = convectionDOE2(h, self.windSpeed, getattr(self.roughness, side))
            if not np.isfinite(h) or h < 0:
                raise ValueError("Convection coefficients must be finite and nonnegative")
            values.append(h)
        self.hCalced = WallSides(*values)

    def initialize(self, delt, TfF, TfB, windSpeed=0, verbose=False):
        if not np.isfinite(delt) or delt <= 0:
            raise ValueError("Time step must be finite and positive")
        self.delt = delt
        self.windSpeed = windSpeed
        self.T_prof = np.array([TfF, TfB], dtype=float)
        self._update_convection(TfF, TfB)
        self.Erad = WallSides(0.0, 0.0)
        self.T = TfF + (TfB - TfF) * self.x[1:-1] / self.th
        self.T_prof = self.getWallProfile(TfF, TfB)

    def timeStep(self, TintF, TintB):
        self._update_convection(TintF, TintB)
        old_profile = self.getWallProfile(TintF, TintB) if not self.implicit else None
        matrix = self.K.copy()
        source = np.zeros(self.n)
        for index, side, air in [(0, 'front', TintF), (-1, 'back', TintB)]:
            g = getattr(self.surface_conductance, side)
            h = getattr(self.hCalced, side)
            radiation = getattr(self.Erad, side)
            fraction = g / (g + h)
            effective_h = fraction * h
            matrix[index, index] += effective_h
            source[index] += effective_h * air + fraction * radiation
        mass_rate = self.capacity / self.delt
        if self.implicit:
            self.A_implicit = matrix + np.diag(mass_rate)
            self.b = source
            self.T = np.linalg.solve(self.A_implicit, mass_rate * self.T + source)
        else:
            # A sufficient monotonicity bound, checked again when wind changes.
            if np.any(self.delt * np.diag(matrix) > self.capacity):
                raise ValueError("Time step too large for explicit wall stability")
            self.T = self.T + (source - matrix @ self.T) / mass_rate
        self.T_prof = self.getWallProfile(TintF, TintB)
        flux_profile = self.T_prof if self.implicit else old_profile
        return WallSides(
            self.Af * self.hCalced.front * (flux_profile[0] - TintF),
            self.Af * self.hCalced.back * (flux_profile[-1] - TintB))

    def getWallProfile(self, TintF, TintB):
        """Return surfaces and cell centers using air temperatures and applied radiation."""
        surfaces = []
        for index, side, air in [(0, 'front', TintF), (-1, 'back', TintB)]:
            g = getattr(self.surface_conductance, side)
            h = getattr(self.hCalced, side)
            radiation = getattr(self.Erad, side)
            surfaces.append((g * self.T[index] + h * air + radiation) / (g + h))
        profile = np.r_[surfaces[0], self.T, surfaces[1]]
        if not np.all(np.isfinite(profile)):
            raise ValueError("Wall temperature is not finite")
        return profile
