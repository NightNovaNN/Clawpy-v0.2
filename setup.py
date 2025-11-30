from setuptools import setup, find_packages
import pathlib

# Read README for PyPI
README = (pathlib.Path(__file__).parent / "Clawpy" / "README.md").read_text(encoding="utf-8")

setup(
    name="clawpy",
    version="0.2",
    author="ISD NightNova",
    description="ClawPy v0.2 — A high-performance scientific, mathematical, AI, and computational toolkit.",
    long_description=README,
    long_description_content_type="text/markdown",

    url="https://github.com/NightNovaNN/Clawpy",
    license="MIT",

    packages=["Clawpy"],  # <- ONLY include this package folder
    include_package_data=True,

    install_requires=[
        "numpy",
        "sympy",
        "matplotlib",
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    python_requires=">=3.8",
)
