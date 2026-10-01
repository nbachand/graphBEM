"""Surface convection in W/(m² K), using local wind speed in m/s.

Reference implementation: EnergyPlus v22.2.0 ConvectionCoefficients.cc,
CalcASHRAETARPNatural, CalcDOE2Windward/Leeward and CalcDOE2Forced.
https://github.com/NREL/EnergyPlus/blob/v22.2.0/src/EnergyPlus/ConvectionCoefficients.cc
Tilt is the face normal's angle from upward (0 up, 90 vertical, 180 down).
Azimuth and meteorological wind direction are clockwise from north.
"""
import math


def natural_convection(surface_temperature, air_temperature, tilt):
    """TARP natural convection; temperatures may be K or °C."""
    if not 0 <= tilt <= 180:
        raise ValueError('Face tilt must be in [0, 180] degrees')
    cosine = math.cos(math.radians(tilt))
    delta = surface_temperature-air_temperature
    magnitude = abs(delta)**(1/3)
    if abs(cosine) < 1e-10:
        return 1.31*magnitude
    # 7.238 follows the source; the Engineering Reference prints 7.283.
    factor = (9.482/(7.238-abs(cosine)) if delta*cosine > 0
              else 1.810/(1.382+abs(cosine)))
    return factor*magnitude


def exterior_convection(surface_temperature, air_temperature, wind_speed,
                        roughness, tilt, azimuth=None, wind_direction=None,
                        exposure='directional', h_natural=None):
    """DOE-2: TARP natural plus roughness-adjusted forced convection.

    h_natural overrides only the natural component for sensitivity studies.
    'average' averages the resulting windward/leeward coefficients equally;
    it is an explicit approximation for an aggregate of unresolved facades.
    """
    if not math.isfinite(wind_speed) or wind_speed < 0:
        raise ValueError('Local wind speed must be finite and nonnegative')
    if not math.isfinite(roughness) or roughness < 0:
        raise ValueError('Roughness multiplier must be finite and nonnegative')
    if exposure not in {'directional', 'average'}:
        raise ValueError('Wind exposure must be directional or average')
    natural = natural_convection(surface_temperature, air_temperature, tilt)
    hn = natural if h_natural is None else h_natural
    if not math.isfinite(hn) or hn < 0:
        raise ValueError('Natural convection coefficient must be finite and nonnegative')
    def combined(hf):
        return hn + roughness*(math.hypot(hn, hf)-hn)
    windward = combined(3.26*wind_speed**.89)
    leeward = combined(3.55*wind_speed**.617)
    if abs(math.cos(math.radians(tilt))) >= .98:
        return windward
    if exposure == 'average':
        return (windward+leeward)/2
    if azimuth is None or wind_direction is None or not all(map(math.isfinite, [azimuth, wind_direction])):
        raise ValueError('Directional exterior convection requires azimuth and wind direction')
    angle = abs((wind_direction-azimuth+180) % 360-180)
    return windward if angle <= 90.001 else leeward
