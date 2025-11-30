# MIT License © 2025 ISD NightNova

import numpy as np

class Calculus:
    """Differentiation, integration, differential equations, and transforms."""

    # ------------------------- DIFFERENTIATION -------------------------

    def derivative(self, f, x, h=1e-5):
        """Central difference approximation of derivative."""
        return (f(x + h) - f(x - h)) / (2 * h)

    def second_derivative(self, f, x, h=1e-5):
        """Second derivative using finite differences."""
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)

    def partial_derivative(self, f, x, y, var="x", h=1e-5):
        """Partial derivative of f(x, y) with respect to x or y."""
        if var == "x":
            return (f(x + h, y) - f(x - h, y)) / (2 * h)
        if var == "y":
            return (f(x, y + h) - f(x, y - h)) / (2 * h)
        raise ValueError("Variable must be 'x' or 'y'.")

    # --------------------------- INTEGRATION ---------------------------

    def integrate(self, f, a, b, n=1000):
        """Definite integral using trapezoidal rule."""
        x = np.linspace(a, b, n)
        return np.trapz(f(x), x)

    def double_integral(self, f, x_range, y_range, n=100):
        """Double integral using a 2D trapezoidal method."""
        x = np.linspace(*x_range, n)
        y = np.linspace(*y_range, n)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(f)(X, Y)

        return np.trapz(np.trapz(Z, x, axis=0), y, axis=0)

    # -------------------- DIFFERENTIAL EQUATIONS ---------------------

    def euler_method(self, df, x0, y0, h, steps):
        """Euler's method for dy/dx = df(x, y)."""
        x, y = x0, y0
        results = [y]

        for _ in range(steps):
            y += h * df(x, y)
            x += h
            results.append(y)

        return np.array(results)

    def runge_kutta(self, df, x0, y0, h, steps):
        """4th-order Runge-Kutta method."""
        x, y = x0, y0
        results = [y]

        for _ in range(steps):
            k1 = h * df(x, y)
            k2 = h * df(x + h / 2, y + k1 / 2)
            k3 = h * df(x + h / 2, y + k2 / 2)
            k4 = h * df(x + h, y + k3)

            y += (k1 + 2*k2 + 2*k3 + k4) / 6
            x += h
            results.append(y)

        return np.array(results)

    # ----------------- FOURIER & LAPLACE TRANSFORMS ------------------

    def fourier_transform(self, f, t_range, n=1000):
        """Numerical Fourier Transform over a range."""
        t = np.linspace(*t_range, n)
        return np.fft.fft(f(t))

    def laplace_transform(self, f, s, t_max=10, n=1000):
        """Numerical Laplace Transform."""
        t = np.linspace(0, t_max, n)
        return np.trapz(np.vectorize(f)(t) * np.exp(-s * t), t)
