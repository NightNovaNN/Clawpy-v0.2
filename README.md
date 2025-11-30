# ClawPy v0.2

ClawPy is a free, open-source Python framework that provides powerful tools for mathematics, science, AI, graph theory, higher-dimensional geometry, signal processing, visualization, and more — all in one modular package.

ClawPy is designed to be:

- Simple
- Lightweight
- Extensible
- Powerful

With pure Python modules and minimal dependencies, ClawPy serves developers, students, researchers, and hobbyists who need fast computational utilities without heavy frameworks.

---

## Features

### Mathematics

- Basic and advanced math
- Algebra and calculus
- Matrices, eigenvalues, LU/QR decomposition
- Combinatorics and prime number theory
- Symbolic solving
- Higher-dimensional geometry (3D–4D–nD)

### Science

- Thermodynamics
- Chemistry (molecular mass, reaction rates)
- Quantum mechanics
- Astronomy and orbital physics

### Machine Learning

- Activation functions
- Regression
- Backpropagation
- K-Means clustering
- (NLP utilities coming soon)

### Visualization & Graph Theory

- 4D tesseract rotation animation
- Perlin noise visualization
- Fractal dimension estimation
- Random graph generation
- Dijkstra’s algorithm
- Eulerian path and circuit detection

### Design

- Clean, modular structure
- Easy imports
- Pure Python implementation
- Zero compiled extensions

---

## Installation

Install ClawPy from PyPI:

```bash
pip install clawpy
```

## Quick Start

### Basic

```python
from clawpy import BasicMath

math_tools = BasicMath()

print(math_tools.add(8, 5))      # 13
print(math_tools.multiply(7, 6)) # 42
print(math_tools.power(12, 2))   # 144
```

### Advanced

```python
from clawpy import ScientificMath, Calculus

science = ScientificMath()
calc = Calculus()

print(science.orbital_velocity(5.97e24, 6.37e6))
print(calc.derivative(lambda x: x**3, 2))  # 12
```

## Included Modules

| Category                     | Modules                                    |
| ---------------------------- | ------------------------------------------ |
| Core Math                    | `BasicMath`, `AdvancedMath`, `AppliedMath` |
| Calculus                     | `Calculus`                                 |
| AI / ML Tools                | `AI`                                       |
| Higher Math & Geometry       | `LeftMath`, `MoreMath`                     |
| Graph Theory & Randomization | `OtherMath`                                |
| Scientific Computing         | `ScientificMath`                           |

---

## Contributing

ClawPy is open-source under the MIT License.  
Contributions, feature requests, and forks are welcome.

---

## License

The source code is licensed under the MIT License.  
See the `LICENSE` file for full details.

**Note:**  
“ClawPy”, “NightNovaNN”, the logo, branding, and design marks are **not** covered by the MIT License and require explicit permission from **ISD NightNova** for any usage.
