# MIT License © 2025 ISD NightNova

import numpy as np
import math

class ScientificMath:
    """Thermodynamics, quantum mechanics, chemistry, and astronomy utilities."""

    # --------------------------- THERMODYNAMICS ---------------------------

    def heat_transfer(self, mass, specific_heat, temp_change):
        """Q = m * c * ΔT"""
        return mass * specific_heat * temp_change

    def entropy_change(self, heat, temp):
        """ΔS = Q / T"""
        if temp == 0:
            raise ZeroDivisionError("Temperature cannot be zero.")
        return heat / temp

    def ideal_gas_law(self, pressure, volume, moles, R=8.314):
        """PV = nRT → returns temperature."""
        if moles == 0:
            raise ZeroDivisionError("Moles cannot be zero.")
        return (pressure * volume) / (moles * R)

    # --------------------------- QUANTUM MECHANICS ---------------------------

    def wave_energy(self, frequency):
        """E = h * f"""
        h = 6.626e-34
        return h * frequency

    def schrodinger_energy(self, n, mass, length):
        """Eₙ = (n² * h²) / (8 m L²)"""
        h = 6.626e-34
        return (n**2 * h**2) / (8 * mass * length**2)

    # --------------------------- CHEMISTRY ---------------------------

    def molecular_weight(self, elements, amounts):
        """Molecular weight from atomic symbols + counts."""
        atomic_weights = {
            "H": 1.008, "C": 12.011, "O": 15.999, "N": 14.007, "Na": 22.990,
            "Cl": 35.45, "Fe": 55.845, "S": 32.065, "P": 30.974
        }

        total = 0
        for elem, count in zip(elements, amounts):
            if elem not in atomic_weights:
                raise ValueError(f"Unknown element: {elem}")
            total += atomic_weights[elem] * count
        return total

    def reaction_rate(self, k, concentrations, orders):
        """rate = k * Π [Aᵢ]^{mᵢ}"""
        return k * np.prod([c ** o for c, o in zip(concentrations, orders)])

    # --------------------------- ASTRONOMY ---------------------------

    def orbital_velocity(self, mass, radius):
        """v = sqrt(GM / R)"""
        if radius == 0:
            raise ZeroDivisionError("Radius cannot be zero.")
        G = 6.67430e-11
        return math.sqrt(G * mass / radius)

    def escape_velocity(self, mass, radius):
        """v = sqrt(2GM / R)"""
        if radius == 0:
            raise ZeroDivisionError("Radius cannot be zero.")
        G = 6.67430e-11
        return math.sqrt(2 * G * mass / radius)
