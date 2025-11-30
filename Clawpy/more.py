# MIT License © 2025 ISD NightNova

import numpy as np
import sympy as sp
import random

class MoreMath:
    """Miscellaneous advanced utilities: graph theory, symbolic math, and matrix algebra."""

    # ---------------------- GEOMETRY / DISTANCE ----------------------

    @staticmethod
    def distance_in_nD(p1, p2):
        """Euclidean distance between two N-dimensional points."""
        p1, p2 = np.asarray(p1), np.asarray(p2)
        return np.linalg.norm(p1 - p2)

    # ---------------------- GRAPH THEORY ----------------------

    @staticmethod
    def generate_random_graph(num_nodes, edge_prob=0.3):
        """Generates a random undirected graph (adjacency matrix)."""
        graph = np.zeros((num_nodes, num_nodes), dtype=int)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < edge_prob:
                    graph[i, j] = graph[j, i] = 1
        return graph

    # ---------------------- SIGNAL PROCESSING ----------------------

    @staticmethod
    def fourier_transform(signal):
        """Discrete Fourier Transform of a 1D signal."""
        return np.fft.fft(signal)

    # ---------------------- SYMBOLIC MATH ----------------------

    @staticmethod
    def solve_equation(equation, var):
        """Solves a symbolic equation using SymPy."""
        return sp.solve(equation, var)

    # ---------------------- MATRIX ALGEBRA ----------------------

    @staticmethod
    def qr_decomposition(matrix):
        """QR decomposition of a matrix."""
        return np.linalg.qr(matrix)

    @staticmethod
    def matrix_eigenvalues(matrix):
        """Eigenvalues of a matrix."""
        return np.linalg.eigvals(matrix)

    @staticmethod
    def matrix_eigenvectors(matrix):
        """Eigenvectors of a matrix."""
        _, vecs = np.linalg.eig(np.asarray(matrix))
        return vecs

    @staticmethod
    def matrix_inverse(matrix):
        """Matrix inverse (throws if singular)."""
        matrix = np.asarray(matrix)
        det = np.linalg.det(matrix)
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")
        return np.linalg.inv(matrix)
