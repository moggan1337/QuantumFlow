"""
QuantumFlow - Quantum Circuit Simulator & Optimizer

A comprehensive quantum computing framework for simulation, optimization,
and algorithm implementation.
"""

__version__ = "1.0.0"
__author__ = "moggan1337"

from .gates import *
from .circuit import QuantumCircuit
from .state import StateVector, DensityMatrix
from .simulator import Simulator
from .optimizer import CircuitOptimizer
from .noise import NoiseModel, DepolarizingNoise, AmplitudeDampingNoise
from .vqe import VQE, VQEResult
from .qaoa import QAOA, QAOAResult

__all__ = [
    "QuantumCircuit",
    "StateVector", 
    "DensityMatrix",
    "Simulator",
    "CircuitOptimizer",
    "NoiseModel",
    "DepolarizingNoise",
    "AmplitudeDampingNoise",
    "VQE",
    "VQEResult",
    "QAOA",
    "QAOAResult",
]
