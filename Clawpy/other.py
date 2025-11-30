# MIT License © 2025 ISD NightNova

import numpy as np
import math
import matplotlib.pyplot as plt
import heapq

class OtherMath:
    """Advanced number theory, graph theory, geometry, randomness, and fractal tools."""

    # ---------------------- NUMBER THEORY ----------------------

    def lucas_number(self, n):
        """Efficient Lucas numbers (iterative)."""
        if n == 0:
            return 2
        if n == 1:
            return 1
        a, b = 2, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    def collatz_sequence(self, n):
        """Generates the Collatz sequence."""
        seq = [n]
        while n > 1:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            seq.append(n)
        return seq

    def partition_function(self, n):
        """Number of ways to partition n."""
        P = [1] + [0] * n
        for k in range(1, n + 1):
            for j in range(k, n + 1):
                P[j] += P[j - k]
        return P[n]

    # ---------------------- GRAPH THEORY ----------------------

    def dijkstra(self, graph, start):
        """Shortest paths using Dijkstra's Algorithm."""
        queue = [(0, start)]
        distances = {node: float("inf") for node in graph}
        distances[start] = 0

        while queue:
            dist, node = heapq.heappop(queue)
            if dist > distances[node]:
                continue

            for neighbor, weight in graph[node].items():
                new_dist = dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(queue, (new_dist, neighbor))

        return distances

    def is_eulerian(self, graph):
        """Check if a graph has an Eulerian Path."""
        odd_count = sum(1 for node in graph if len(graph[node]) % 2 != 0)
        return odd_count in (0, 2)

    def euler_characteristic(self, V, E, F):
        """Euler characteristic for surfaces: χ = V - E + F."""
        return V - E + F

    def random_graph(self, nodes, density=0.5):
        """Random symmetric adjacency matrix."""
        A = (np.random.rand(nodes, nodes) < density).astype(int)
        A = np.triu(A, 1)
        return A + A.T

    # ---------------------- HIGHER DIMENSION GEOMETRY ----------------------

    def hypersphere_volume(self, r, d):
        """Volume of a d-dimensional hypersphere."""
        return (np.pi ** (d / 2)) / math.gamma(d / 2 + 1) * (r ** d)

    def hypercube_volume(self, side, d):
        return side ** d

    def distance_in_4d(self, p1, p2):
        return np.linalg.norm(np.asarray(p1) - np.asarray(p2))

    # ---------------------- RANDOMIZATION ----------------------

    def monte_carlo_pi(self, samples=10_000):
        """Monte Carlo approximation of π."""
        points = np.random.rand(samples, 2)
        inside = np.sum(np.sum(points**2, axis=1) <= 1)
        return 4 * inside / samples

    def random_walk(self, steps=10):
        """Simple symmetric random walk."""
        walk = [0]
        for _ in range(steps):
            walk.append(walk[-1] + np.random.choice([-1, 1]))
        return walk

    # ---------------------- PROCEDURAL GENERATION ----------------------

    def perlin_noise(self, size, scale=10, show_graph=True):
        """
        Generates pseudo-Perlin 1D noise.
        (Not true gradient-based Perlin; simple wave + noise.)
        """
        x = np.linspace(0, scale, size)
        noise = np.sin(2 * np.pi * x) + np.random.normal(scale=0.2, size=size)

        if show_graph:
            plt.plot(x, noise)
            plt.xlabel("X")
            plt.ylabel("Noise Value")
            plt.title("Pseudo Perlin Noise")
            plt.grid()
            plt.show()

        return noise

    def fractal_dimension(self, Z, threshold=0.5, show_graph=True):
        """Box-counting method for 2D fractal dimension."""
        Z = (Z > threshold)

        def boxcount(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                np.arange(0, Z.shape[1], k), axis=1
            )
            return np.count_nonzero(S)

        max_power = int(np.log2(min(Z.shape)))
        sizes = [2**i for i in range(max_power, 0, -1)]
        counts = np.array([boxcount(Z, k) for k in sizes])

        valid = counts > 0
        if np.any(valid):
            coeffs = np.polyfit(np.log(sizes[valid]), np.log(counts[valid]), 1)
            fractal_dim = -coeffs[0]
        else:
            fractal_dim = None

        if show_graph:
            plt.imshow(Z, cmap="binary")
            plt.title(f"Fractal Dimension: {fractal_dim}")
            plt.colorbar()
            plt.show()

        return fractal_dim
