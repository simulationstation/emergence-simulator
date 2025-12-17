"""Physical constants and simple unit conversions (SI units)."""

import math

c = 299_792_458.0  # speed of light in vacuum (m/s)
hbar = 1.054_571_817e-34  # reduced Planck constant (J*s)
kB = 1.380_649e-23  # Boltzmann constant (J/K)
ln2 = math.log(2.0)
pi = math.pi


def eV_to_J(energy_ev: float) -> float:
    """Convert electronvolts to joules."""
    return energy_ev * 1.602_176_634e-19


def J_to_eV(energy_j: float) -> float:
    """Convert joules to electronvolts."""
    return energy_j / 1.602_176_634e-19


def year_to_s(years: float) -> float:
    """Convert (Julian) years to seconds."""
    return years * 365.25 * 24 * 3600


def Gyr_to_s(gigayears: float) -> float:
    """Convert gigayears to seconds."""
    return year_to_s(gigayears * 1e9)


hbar_c = hbar * c
