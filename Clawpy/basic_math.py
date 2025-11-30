# MIT License © 2025 ISD NightNova

import math
import numpy as np

class BasicMath:
    """Core mathematical utilities: arithmetic, algebra, and matrix operations."""

    # ----------------------- Basic Arithmetic -----------------------

    def add(self, a, b): return a + b
    def subtract(self, a, b): return a - b
    def multiply(self, a, b): return a * b

    def divide(self, a, b):
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return a / b

    def power(self, a, b): return a ** b

    def root(self, a, n):
        if n == 0:
            raise ValueError("Root degree cannot be zero.")
        return a ** (1 / n)

    def modulo(self, a, b): return a % b
    def factorial(self, n): return math.factorial(n)

    # ---------------------- Algebra Utilities -----------------------

    def gcd(self, a, b): return math.gcd(a, b)
    def lcm(self, a, b): return abs(a * b) // math.gcd(a, b)

    # ----------------------- Matrix Operations ----------------------

    def matrix_inverse(self, A):
        det = np.linalg.det(A)
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")
        return np.linalg.inv(A)

    def matrix_determinant(self, A): return np.linalg.det(A)
    def matrix_eigenvalues(self, A): return np.linalg.eigvals(A)

    def matrix_eigenvectors(self, A):
        _, vecs = np.linalg.eig(A)
        return vecs

    # ----------------------- LU Decomposition -----------------------

    def lu_decomposition(self, A):
        """Performs LU decomposition without SciPy."""
        A = A.astype(float)
        n = len(A)
        L = np.eye(n)
        U = A.copy()

        for i in range(n):
            pivot = U[i, i]
            if pivot == 0:
                raise ValueError("Zero pivot encountered in LU decomposition.")

            for j in range(i + 1, n):
                factor = U[j, i] / pivot
                L[j, i] = factor
                U[j] -= factor * U[i]

        return L, U

    # ----------------------- QR Decomposition -----------------------

    def qr_decomposition(self, A):
        return np.linalg.qr(A)
