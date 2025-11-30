# MIT License © 2025 ISD NightNova

import numpy as np
import math

class AdvancedMath:
    """Advanced matrix operations, number theory, and combinatorics."""

    # ---------------------- MATRIX OPERATIONS ----------------------

    def matrix_operations(self, matrix):
        """Performs eigen decomposition, LU, QR, determinant, and inverse."""
        matrix = np.asarray(matrix)

        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        L, U = self.lu_decomposition(matrix)
        Q, R = np.linalg.qr(matrix)
        determinant = np.linalg.det(matrix)

        if determinant == 0:
            inverse = None
        else:
            inverse = np.linalg.inv(matrix)

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "LU": (L, U),
            "QR": (Q, R),
            "determinant": determinant,
            "inverse": inverse
        }

    def lu_decomposition(self, matrix):
        """Doolittle's LU decomposition."""
        matrix = np.asarray(matrix, dtype=float)
        n = matrix.shape[0]

        L = np.eye(n)
        U = np.zeros((n, n))

        for i in range(n):
            # Upper triangular values
            for j in range(i, n):
                s = sum(L[i, k] * U[k, j] for k in range(i))
                U[i, j] = matrix[i, j] - s

            # Lower triangular values
            for j in range(i + 1, n):
                s = sum(L[j, k] * U[k, i] for k in range(i))
                if U[i, i] == 0:
                    raise ValueError("Matrix is singular.")
                L[j, i] = (matrix[j, i] - s) / U[i, i]

        return L, U

    # ---------------------- NUMBER THEORY ----------------------

    def is_prime(self, num):
        """Checks primality."""
        if num <= 1:
            return False
        if num <= 3:
            return True
        if num % 2 == 0 or num % 3 == 0:
            return False

        i = 5
        while i * i <= num:
            if num % i == 0 or num % (i + 2) == 0:
                return False
            i += 6
        return True

    def prime_factors(self, num):
        """Returns the prime factors of a number."""
        factors = []
        i = 2
        while i * i <= num:
            while num % i == 0:
                factors.append(i)
                num //= i
            i += 1
        if num > 1:
            factors.append(num)
        return factors

    def mod_exp(self, base, exp, mod):
        """Computes (base^exp) % mod efficiently."""
        result = 1
        base %= mod

        while exp > 0:
            if exp & 1:
                result = (result * base) % mod
            base = (base * base) % mod
            exp >>= 1

        return result

    # ---------------------- COMBINATORICS ----------------------

    def permutations(self, n, r):
        return math.factorial(n) // math.factorial(n - r)

    def combinations(self, n, r):
        return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))
