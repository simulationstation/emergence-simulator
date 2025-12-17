import math

# SI constants
c = 299792458.0  # speed of light (m/s)
hbar = 1.054571817e-34  # reduced Planck constant (J·s)
kB = 1.380649e-23  # Boltzmann constant (J/K)
ln2 = math.log(2)
pi = math.pi

# Conversion helpers
def eV_to_J(x: float) -> float:
    """Convert electron-volts to joules."""
    return x * 1.602176634e-19

def J_to_eV(x: float) -> float:
    """Convert joules to electron-volts."""
    return x / 1.602176634e-19

# Time conversions
year_to_s = 365.25 * 24 * 3600  # seconds per year
Gyr_to_s = year_to_s * 1e9  # seconds per gigayear
