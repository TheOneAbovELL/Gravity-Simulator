"""Physical constants and unit conversions (SI units throughout)."""

G = 6.67430e-11
"""Newtonian gravitational constant, m^3 kg^-1 s^-2 (CODATA 2018)."""

C = 299_792_458.0
"""Speed of light in vacuum, m/s."""

AU = 1.495978707e11
"""Astronomical unit, m (IAU 2012)."""

KM = 1.0e3
MINUTE = 60.0
HOUR = 3600.0
DAY = 86_400.0
YEAR = 365.25 * DAY
"""Julian year, s."""

GM_SUN = 1.32712440041e20
"""Heliocentric gravitational parameter G*M, m^3/s^2 (JPL DE440)."""

GM_EARTH = 3.986004415e14
"""Geocentric gravitational parameter, m^3/s^2 (JPL DE440)."""

# G*M is measured ~10^5 times more precisely than G alone, so masses are
# defined from GM: then G * mass reproduces the measured GM exactly and orbital
# periods come out right (G times a kg value would make Earth's year ~6 minutes short).
SOLAR_MASS = GM_SUN / G
EARTH_MASS = GM_EARTH / G
EARTH_RADIUS = 6.371e6
SOLAR_RADIUS = 6.957e8

GRAMS_PER_CM3 = 1000.0
"""Multiply a density in g/cm^3 by this to get kg/m^3."""
