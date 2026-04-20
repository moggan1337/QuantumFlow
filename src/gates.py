"""
Quantum Gates Module

Implements all fundamental quantum gates including single-qubit and multi-qubit gates.
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from abc import ABC, abstractmethod


class Gate(ABC):
    """Base class for all quantum gates."""
    
    def __init__(self, num_qubits: int, name: str = "Gate"):
        self.num_qubits = num_qubits
        self.name = name
        self._matrix = None
    
    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Return the matrix representation of the gate."""
        pass
    
    @property
    def dagger(self) -> 'Gate':
        """Return the adjoint (dagger) of this gate."""
        return GateFactory.dagger(self)
    
    def __repr__(self):
        return f"{self.name}({self.num_qubits} qubits)"
    
    def __eq__(self, other):
        if not isinstance(other, Gate):
            return False
        return np.allclose(self.matrix, other.matrix)


class SingleQubitGate(Gate):
    """Base class for single-qubit quantum gates."""
    
    def __init__(self, name: str = "SingleQubitGate"):
        super().__init__(num_qubits=1, name=name)


class TwoQubitGate(Gate):
    """Base class for two-qubit quantum gates."""
    
    def __init__(self, name: str = "TwoQubitGate"):
        super().__init__(num_qubits=2, name=name)


class ThreeQubitGate(Gate):
    """Base class for three-qubit quantum gates."""
    
    def __init__(self, name: str = "ThreeQubitGate"):
        super().__init__(num_qubits=3, name=name)


# ============== Single-Qubit Gates ==============

class IdentityGate(SingleQubitGate):
    """Identity gate - leaves qubit state unchanged."""
    
    def __init__(self):
        super().__init__(name="I")
        self._matrix = np.eye(2, dtype=complex)
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class PauliXGate(SingleQubitGate):
    """Pauli-X gate (quantum NOT gate)."""
    
    def __init__(self):
        super().__init__(name="X")
        self._matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class PauliYGate(SingleQubitGate):
    """Pauli-Y gate."""
    
    def __init__(self):
        super().__init__(name="Y")
        self._matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class PauliZGate(SingleQubitGate):
    """Pauli-Z gate (phase flip)."""
    
    def __init__(self):
        super().__init__(name="Z")
        self._matrix = np.array([[1, 0], [0, -1]], dtype=complex)
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class HadamardGate(SingleQubitGate):
    """Hadamard gate - creates superposition."""
    
    def __init__(self):
        super().__init__(name="H")
        self._matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class SGate(SingleQubitGate):
    """S gate (phase gate, sqrt of Z)."""
    
    def __init__(self):
        super().__init__(name="S")
        self._matrix = np.array([[1, 0], [0, 1j]], dtype=complex)
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class TGate(SingleQubitGate):
    """T gate (pi/8 gate)."""
    
    def __init__(self):
        super().__init__(name="T")
        self._matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class SdgGate(SingleQubitGate):
    """S-dagger gate (adjoint of S)."""
    
    def __init__(self):
        super().__init__(name="S†")
        self._matrix = np.array([[1, 0], [0, -1j]], dtype=complex)
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class TdgGate(SingleQubitGate):
    """T-dagger gate (adjoint of T)."""
    
    def __init__(self):
        super().__init__(name="T†")
        self._matrix = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class RXGate(SingleQubitGate):
    """Rotation around X-axis."""
    
    def __init__(self, theta: float):
        super().__init__(name=f"RX({theta:.4f})")
        self.theta = theta
    
    @property
    def matrix(self) -> np.ndarray:
        cos_half = np.cos(self.theta / 2)
        sin_half = np.sin(self.theta / 2)
        return np.array([[cos_half, -1j * sin_half], 
                        [-1j * sin_half, cos_half]], dtype=complex)


class RYGate(SingleQubitGate):
    """Rotation around Y-axis."""
    
    def __init__(self, theta: float):
        super().__init__(name=f"RY({theta:.4f})")
        self.theta = theta
    
    @property
    def matrix(self) -> np.ndarray:
        cos_half = np.cos(self.theta / 2)
        sin_half = np.sin(self.theta / 2)
        return np.array([[cos_half, -sin_half], 
                        [sin_half, cos_half]], dtype=complex)


class RZGate(SingleQubitGate):
    """Rotation around Z-axis."""
    
    def __init__(self, theta: float):
        super().__init__(name=f"RZ({theta:.4f})")
        self.theta = theta
    
    @property
    def matrix(self) -> np.ndarray:
        exp_neg = np.exp(-1j * self.theta / 2)
        exp_pos = np.exp(1j * self.theta / 2)
        return np.array([[exp_neg, 0], [0, exp_pos]], dtype=complex)


class PhaseGate(SingleQubitGate):
    """General phase gate P(λ)."""
    
    def __init__(self, lam: float):
        super().__init__(name=f"P({lam:.4f})")
        self.lam = lam
    
    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j * self.lam)]], dtype=complex)


class UGate(SingleQubitGate):
    """General single-qubit unitary gate U(θ, φ, λ)."""
    
    def __init__(self, theta: float, phi: float, lam: float):
        super().__init__(name=f"U({theta:.4f},{phi:.4f},{lam:.4f})")
        self.theta = theta
        self.phi = phi
        self.lam = lam
    
    @property
    def matrix(self) -> np.ndarray:
        cos_half = np.cos(self.theta / 2)
        sin_half = np.sin(self.theta / 2)
        return np.array([
            [cos_half, -np.exp(1j * self.lam) * sin_half],
            [np.exp(1j * self.phi) * sin_half, 
             np.exp(1j * (self.phi + self.lam)) * cos_half]
        ], dtype=complex)


# ============== Two-Qubit Gates ==============

class CNOTGate(TwoQubitGate):
    """Controlled-NOT (CNOT) gate - flips target if control is |1⟩."""
    
    def __init__(self, control: int = 0, target: int = 1):
        super().__init__(name="CNOT")
        self.control = control
        self.target = target
        self._matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class CZGate(TwoQubitGate):
    """Controlled-Z (CZ) gate - applies Z on target if control is |1⟩."""
    
    def __init__(self, control: int = 0, target: int = 1):
        super().__init__(name="CZ")
        self.control = control
        self.target = target
        self._matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class SwapGate(TwoQubitGate):
    """SWAP gate - exchanges two qubits."""
    
    def __init__(self, qubit1: int = 0, qubit2: int = 1):
        super().__init__(name="SWAP")
        self.qubit1 = qubit1
        self.qubit2 = qubit2
        self._matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class iSwapGate(TwoQubitGate):
    """iSWAP gate - swaps with phase."""
    
    def __init__(self, qubit1: int = 0, qubit2: int = 1):
        super().__init__(name="iSWAP")
        self.qubit1 = qubit1
        self.qubit2 = qubit2
        self._matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class RXXGate(TwoQubitGate):
    """XX interaction gate."""
    
    def __init__(self, theta: float, qubit1: int = 0, qubit2: int = 1):
        super().__init__(name=f"RXX({theta:.4f})")
        self.theta = theta
        self.qubit1 = qubit1
        self.qubit2 = qubit2
    
    @property
    def matrix(self) -> np.ndarray:
        cos_half = np.cos(self.theta / 2)
        sin_half = np.sin(self.theta / 2)
        return np.array([
            [cos_half, 0, 0, -1j * sin_half],
            [0, cos_half, -1j * sin_half, 0],
            [0, -1j * sin_half, cos_half, 0],
            [-1j * sin_half, 0, 0, cos_half]
        ], dtype=complex)


class RZZGate(TwoQubitGate):
    """ZZ interaction gate."""
    
    def __init__(self, theta: float, qubit1: int = 0, qubit2: int = 1):
        super().__init__(name=f"RZZ({theta:.4f})")
        self.theta = theta
        self.qubit1 = qubit1
        self.qubit2 = qubit2
    
    @property
    def matrix(self) -> np.ndarray:
        exp_neg = np.exp(-1j * self.theta / 2)
        exp_pos = np.exp(1j * self.theta / 2)
        return np.array([
            [exp_neg, 0, 0, 0],
            [0, exp_pos, 0, 0],
            [0, 0, exp_pos, 0],
            [0, 0, 0, exp_neg]
        ], dtype=complex)


class CRXGate(TwoQubitGate):
    """Controlled-RX gate."""
    
    def __init__(self, theta: float, control: int = 0, target: int = 1):
        super().__init__(name=f"CRX({theta:.4f})")
        self.theta = theta
        self.control = control
        self.target = target
    
    @property
    def matrix(self) -> np.ndarray:
        cos_half = np.cos(self.theta / 2)
        sin_half = np.sin(self.theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cos_half, -1j * sin_half],
            [0, 0, -1j * sin_half, cos_half]
        ], dtype=complex)


class CRZGate(TwoQubitGate):
    """Controlled-RZ gate."""
    
    def __init__(self, theta: float, control: int = 0, target: int = 1):
        super().__init__(name=f"CRZ({theta:.4f})")
        self.theta = theta
        self.control = control
        self.target = target
    
    @property
    def matrix(self) -> np.ndarray:
        exp_neg = np.exp(-1j * self.theta / 2)
        exp_pos = np.exp(1j * self.theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, exp_neg, 0],
            [0, 0, 0, exp_pos]
        ], dtype=complex)


class CPhaseGate(TwoQubitGate):
    """Controlled-phase gate."""
    
    def __init__(self, theta: float, control: int = 0, target: int = 1):
        super().__init__(name=f"CP({theta:.4f})")
        self.theta = theta
        self.control = control
        self.target = target
    
    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1j * self.theta)]
        ], dtype=complex)


# ============== Three-Qubit Gates ==============

class ToffoliGate(ThreeQubitGate):
    """Toffoli gate (CCNOT) - flips target if both controls are |1⟩."""
    
    def __init__(self, control1: int = 0, control2: int = 1, target: int = 2):
        super().__init__(name="Toffoli")
        self.control1 = control1
        self.control2 = control2
        self.target = target
        self._matrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0]
        ], dtype=complex)
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class FredkinGate(ThreeQubitGate):
    """Fredkin gate (CSWAP) - swaps target qubits if control is |1⟩."""
    
    def __init__(self, control: int = 0, target1: int = 1, target2: int = 2):
        super().__init__(name="Fredkin")
        self.control = control
        self.target1 = target1
        self.target2 = target2
        self._matrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=complex)
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class SWAPGate(ThreeQubitGate):
    """Double-controlled SWAP (Fredkin variant)."""
    
    def __init__(self, control1: int = 0, control2: int = 1, target1: int = 2, target2: int = 3):
        super().__init__(name="SWAP3")
        self.control1 = control1
        self.control2 = control2
        self.target1 = target1
        self.target2 = target2


class GateFactory:
    """Factory for creating and manipulating quantum gates."""
    
    # Gate definitions for dagger operations
    DAGGER_MAP = {
        'H': HadamardGate,
        'S': SdgGate,
        'T': TdgGate,
        'X': PauliXGate,
        'Y': PauliYGate,
        'Z': PauliZGate,
        'I': IdentityGate,
    }
    
    # Commutation relations for optimization
    COMMUTE_GATES = {
        'X': ['X', 'Y', 'Z', 'I', 'H'],
        'Y': ['X', 'Y', 'Z', 'I', 'H'],
        'Z': ['Z', 'S', 'T', 'I'],
        'H': ['H', 'X', 'Y', 'Z'],
        'CNOT': ['CNOT', 'CZ', 'SWAP'],
        'CZ': ['CNOT', 'CZ', 'SWAP'],
        'SWAP': ['SWAP', 'CNOT', 'CZ'],
    }
    
    # Gate cancellation pairs
    CANCEL_PAIRS = [
        (('X', 'X'), IdentityGate),
        (('Y', 'Y'), IdentityGate),
        (('Z', 'Z'), IdentityGate),
        (('H', 'H'), IdentityGate),
        (('S', 'S'), IdentityGate),
        (('T', 'T'), SGate),
        (('S', 'Sdg'), IdentityGate),
        (('Sdg', 'S'), IdentityGate),
        (('T', 'Tdg'), IdentityGate),
        (('Tdg', 'T'), IdentityGate),
    ]
    
    @classmethod
    def dagger(cls, gate: Gate) -> Gate:
        """Create the dagger (adjoint) of a gate."""
        gate_name = gate.name.split('(')[0]  # Handle parameterized gates
        
        if gate_name in cls.DAGGER_MAP:
            return cls.DAGGER_MAP[gate_name]()
        
        # For parameterized gates, use conjugate transpose
        if isinstance(gate, RXGate):
            return RXGate(-gate.theta)
        elif isinstance(gate, RYGate):
            return RYGate(-gate.theta)
        elif isinstance(gate, RZGate):
            return RZGate(-gate.theta)
        elif isinstance(gate, PhaseGate):
            return PhaseGate(-gate.lam)
        elif isinstance(gate, UGate):
            return UGate(-gate.theta, -gate.lam, -gate.phi)
        elif isinstance(gate, (CNOTGate, CZGate, SwapGate, ToffoliGate, FredkinGate)):
            # These gates are self-adjoint or need special handling
            return type(gate)(**gate.__dict__)
        
        # Default: conjugate transpose
        return cls._create_from_matrix(gate.matrix.T.conj(), f"{gate.name}†", gate.num_qubits)
    
    @classmethod
    def _create_from_matrix(cls, matrix: np.ndarray, name: str, num_qubits: int) -> Gate:
        """Create a gate from its matrix representation."""
        # This is a simplified version - in practice, you'd need more sophisticated matching
        gate = GenericGate(num_qubits, name, matrix)
        return gate
    
    @classmethod
    def controlled(cls, gate: SingleQubitGate, control: int = 0, target: int = 1) -> np.ndarray:
        """Create a controlled version of a single-qubit gate."""
        I = np.eye(2, dtype=complex)
        matrix = gate.matrix
        
        # |0><0| ⊗ I + |1><1| ⊗ U
        controlled = np.kron(np.array([[1, 0], [0, 0]], dtype=complex), I) + \
                    np.kron(np.array([[0, 0], [0, 1]], dtype=complex), matrix)
        return controlled
    
    @classmethod
    def can_cancel(cls, gate1: Gate, gate2: Gate) -> bool:
        """Check if two gates can cancel each other."""
        name1 = gate1.name.split('(')[0]
        name2 = gate2.name.split('(')[0]
        return (name1, name2) in [(n[0], n[1]) for n, _ in cls.CANCEL_PAIRS]
    
    @classmethod
    def can_commute(cls, gate1: Gate, gate2: Gate) -> bool:
        """Check if two gates commute."""
        name1 = gate1.name.split('(')[0]
        name2 = gate2.name.split('(')[0]
        return name2 in cls.COMMUTE_GATES.get(name1, [])


class GenericGate(Gate):
    """A generic gate defined by an arbitrary unitary matrix."""
    
    def __init__(self, num_qubits: int, name: str, matrix: np.ndarray):
        super().__init__(num_qubits, name)
        self._matrix = matrix
    
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class MeasurementGate(Gate):
    """Measurement gate (non-unitary, for simulation)."""
    
    def __init__(self, qubit: int):
        super().__init__(num_qubits=1, name=f"M({qubit})")
        self.qubit = qubit
    
    @property
    def matrix(self) -> np.ndarray:
        # Measurement is not unitary, so we return identity
        # Real measurement is handled in the simulator
        return np.eye(2, dtype=complex)


# Convenience functions
def I() -> IdentityGate:
    """Identity gate."""
    return IdentityGate()

def X() -> PauliXGate:
    """Pauli-X (NOT) gate."""
    return PauliXGate()

def Y() -> PauliYGate:
    """Pauli-Y gate."""
    return PauliYGate()

def Z() -> PauliZGate:
    """Pauli-Z gate."""
    return PauliZGate()

def H() -> HadamardGate:
    """Hadamard gate."""
    return HadamardGate()

def S() -> SGate:
    """S (phase) gate."""
    return SGate()

def T() -> TGate:
    """T (pi/8) gate."""
    return TGate()

def Sdg() -> SdgGate:
    """S-dagger gate."""
    return SdgGate()

def Tdg() -> TdgGate:
    """T-dagger gate."""
    return TdgGate()

def RX(theta: float) -> RXGate:
    """X-rotation gate."""
    return RXGate(theta)

def RY(theta: float) -> RYGate:
    """Y-rotation gate."""
    return RYGate(theta)

def RZ(theta: float) -> RZGate:
    """Z-rotation gate."""
    return RZGate(theta)

def CNOT(control: int = 0, target: int = 1) -> CNOTGate:
    """Controlled-NOT gate."""
    return CNOTGate(control, target)

def CZ(control: int = 0, target: int = 1) -> CZGate:
    """Controlled-Z gate."""
    return CZGate(control, target)

def SWAP(qubit1: int = 0, qubit2: int = 1) -> SwapGate:
    """SWAP gate."""
    return SwapGate(qubit1, qubit2)

def iSWAP(qubit1: int = 0, qubit2: int = 1) -> iSwapGate:
    """iSWAP gate."""
    return iSwapGate(qubit1, qubit2)

def Toffoli(control1: int = 0, control2: int = 1, target: int = 2) -> ToffoliGate:
    """Toffoli (CCNOT) gate."""
    return ToffoliGate(control1, control2, target)

def Fredkin(control: int = 0, target1: int = 1, target2: int = 2) -> FredkinGate:
    """Fredkin (CSWAP) gate."""
    return FredkinGate(control, target1, target2)

def U(theta: float, phi: float, lam: float) -> UGate:
    """General unitary gate."""
    return UGate(theta, phi, lam)

def RXX(theta: float, qubit1: int = 0, qubit2: int = 1) -> RXXGate:
    """XX interaction gate."""
    return RXXGate(theta, qubit1, qubit2)

def RZZ(theta: float, qubit1: int = 0, qubit2: int = 1) -> RZZGate:
    """ZZ interaction gate."""
    return RZZGate(theta, qubit1, qubit2)
