# MIT License © 2025 ISD NightNova

import numpy as np
import matplotlib.pyplot as plt
import random

class AppliedMath:
    """Physics, statistics, signal processing, and plotting utilities."""

    # -------------------------- PHYSICS --------------------------

    def newton_second_law(self, force, mass):
        """Returns acceleration from F = ma."""
        if mass == 0:
            raise ZeroDivisionError("Mass cannot be zero.")
        return force / mass

    def kinetic_energy(self, mass, velocity):
        """Kinetic energy: KE = 0.5 * m * v^2."""
        return 0.5 * mass * velocity**2

    def ohms_law(self, voltage, resistance):
        """Current computed from V = IR."""
        if resistance == 0:
            raise ZeroDivisionError("Resistance cannot be zero.")
        return voltage / resistance

    def gravitational_force(self, m1, m2, r):
        """Newton's law of gravitation."""
        if r == 0:
            raise ZeroDivisionError("Distance cannot be zero.")
        G = 6.67430e-11
        return G * m1 * m2 / (r**2)

    def aerodynamic_lift(self, rho, v, A, C_L):
        """Lift equation: L = 0.5 * rho * v^2 * A * C_L."""
        return 0.5 * rho * v**2 * A * C_L

    # ------------------------ STATISTICS -------------------------

    def mean(self, data):
        return np.mean(data)

    def variance(self, data):
        return np.var(data)

    def standard_deviation(self, data):
        return np.std(data)

    def correlation_coefficient(self, x, y):
        return np.corrcoef(x, y)[0, 1]

    # ---------------------- MATH GENERATOR -----------------------

    def generate_math_problem(self, difficulty="medium"):
        """Generates a random arithmetic problem."""
        if difficulty == "easy":
            a, b = random.randint(1, 10), random.randint(1, 10)
            return f"{a} + {b} = ?", a + b

        if difficulty == "medium":
            a, b = random.randint(1, 20), random.randint(1, 20)
            return f"{a} × {b} = ?", a * b

        if difficulty == "hard":
            a = random.randint(2, 10)
            return f"{a}² = ?", a**2

        raise ValueError("Invalid difficulty level.")

    # ------------------- SIGNAL PROCESSING -----------------------

    def fourier_transform(self, signal):
        return np.fft.fft(signal)

    def inverse_fourier_transform(self, signal):
        return np.fft.ifft(signal)

    def convolution(self, signal1, signal2):
        return np.convolve(signal1, signal2, mode="full")

    # --------------------------- GRAPHING -------------------------

    def generate_plot(self, func, start=-10, end=10, num_points=100):
        x = np.linspace(start, end, num_points)
        return x, np.vectorize(func)(x)

    def plot_function(self, func, start=-10, end=10, num_points=100):
        x, y = self.generate_plot(func, start, end, num_points)
        plt.plot(x, y)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Function Plot")
        plt.grid()
        plt.show()
