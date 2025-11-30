# MIT License © 2025 ISD NightNova

import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.animation import FuncAnimation

class LeftMath:
    """Complex numbers, number theory, geometry, and 4D projections."""

    # ---------------------- COMPLEX & IMAGINARY ----------------------

    def complex_magnitude(self, z):
        return abs(complex(z))

    def complex_phase(self, z):
        return np.angle(complex(z))

    def complex_conjugate(self, z):
        return np.conj(complex(z))

    def imaginary_root(self, n):
        return np.sqrt(complex(n))

    # ---------------------- HYPERBOLIC TRIG ----------------------

    def sinh(self, x): return np.sinh(x)
    def cosh(self, x): return np.cosh(x)
    def tanh(self, x): return np.tanh(x)

    # ---------------------- FIBONACCI & PASCAL ----------------------

    def fibonacci(self, n):
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    def golden_ratio(self):
        return (1 + math.sqrt(5)) / 2

    def pascal_triangle(self, n):
        triangle = [[1]]
        for _ in range(1, n):
            prev = triangle[-1]
            row = [1] + [prev[i] + prev[i + 1] for i in range(len(prev) - 1)] + [1]
            triangle.append(row)
        return triangle

    # ---------------------- PRIME NUMBER THEORY ----------------------

    def is_prime(self, n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        r = int(math.sqrt(n))
        for i in range(3, r + 1, 2):
            if n % i == 0:
                return False
        return True

    def prime_factors(self, n):
        factors = []
        while n % 2 == 0:
            factors.append(2)
            n //= 2
        i = 3
        while i * i <= n:
            while n % i == 0:
                factors.append(i)
                n //= i
            i += 2
        if n > 1:
            factors.append(n)
        return factors

    def next_prime(self, n):
        n += 1
        while not self.is_prime(n):
            n += 1
        return n

    # ---------------------- MENSURATION ----------------------

    def circle_area(self, r): return math.pi * r ** 2
    def sphere_volume(self, r): return (4 / 3) * math.pi * r ** 3
    def cylinder_volume(self, r, h): return math.pi * r ** 2 * h
    def cone_volume(self, r, h): return (1 / 3) * math.pi * r ** 2 * h
    def pyramid_volume(self, base_area, h): return (1 / 3) * base_area * h

    # ---------------------- 4D TESSERACT ANIMATION ----------------------

    def draw_tesseract(self):
        """Rotating 4D hypercube projected into 3D."""

        fig = plt.figure(figsize=8)
        ax = fig.add_subplot(111, projection="3d")

        # All 16 vertices of a 4D hypercube
        vertices = np.array(list(product([-1, 1], repeat=4)))

        # Find edges (vertices differing by exactly 1 dimension)
        edges = [
            (i, j) for i in range(len(vertices)) for j in range(i + 1, len(vertices))
            if np.sum(vertices[i] != vertices[j]) == 1
        ]

        def project(vertices, angle):
            """Rotate in 4D, project to 3D."""
            R = np.array([
                [np.cos(angle), -np.sin(angle), 0, 0],
                [np.sin(angle),  np.cos(angle), 0, 0],
                [0, 0, np.cos(angle), -np.sin(angle)],
                [0, 0, np.sin(angle),  np.cos(angle)]
            ])
            rotated = vertices @ R.T
            return rotated[:, :3]

        def update(frame):
            ax.clear()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-2, 2)
            ax.set_title(f"Tesseract Rotation – Frame {frame}")

            proj = project(vertices, angle=frame * 0.05)

            for start, end in edges:
                x, y, z = zip(proj[start], proj[end])
                ax.plot(x, y, z, color="cyan")

        FuncAnimation(fig, update, frames=200, interval=20, blit=False)
        plt.show()
