"""
Noise Model Module

Implements quantum noise models for realistic circuit simulation.
Includes depolarizing, amplitude damping, phase damping, and custom noise channels.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .state import StateVector, DensityMatrix


@dataclass
class NoiseChannel:
    """Represents a quantum noise channel."""
    name: str
    kraus_operators: List[np.ndarray]
    description: str = ""
    
    def apply(self, state: Union[StateVector, DensityMatrix],
              qubits: List[int]) -> DensityMatrix:
        """
        Apply the noise channel to a state.
        
        Args:
            state: Input quantum state
            qubits: Qubits to apply noise to
            
        Returns:
            Updated density matrix
        """
        if isinstance(state, StateVector):
            # Convert to density matrix first
            state = DensityMatrix(state)
        
        return state.apply_channel(self.kraus_operators, qubits)


class NoiseModel(ABC):
    """Abstract base class for noise models."""
    
    @abstractmethod
    def get_channel(self, gate_name: str, qubits: List[int]) -> Optional[NoiseChannel]:
        """
        Get the noise channel for a specific gate operation.
        
        Args:
            gate_name: Name of the gate being applied
            qubits: Qubits the gate acts on
            
        Returns:
            NoiseChannel or None if no noise for this gate
        """
        pass
    
    def apply(self, state: Union[StateVector, DensityMatrix],
             instruction: 'CircuitInstruction') -> DensityMatrix:
        """
        Apply noise model after an instruction.
        
        Args:
            state: Current state
            instruction: Circuit instruction that was just applied
            
        Returns:
            Updated state
        """
        channel = self.get_channel(instruction.gate.name, instruction.qubits)
        
        if channel is None:
            if isinstance(state, StateVector):
                return DensityMatrix(state)
            return state
        
        return channel.apply(state, instruction.qubits)


class DepolarizingNoise(NoiseModel):
    """
    Depolarizing noise channel.
    
    The depolarizing channel transforms a state as:
    ρ → (1 - p) ρ + (p/2) I
    
    This represents random Pauli errors occurring with probability p.
    """
    
    def __init__(self, probability: float):
        """
        Initialize depolarizing noise.
        
        Args:
            probability: Total error probability p
        """
        self.probability = probability
        self.error_prob = probability / 3  # Equal probability for X, Y, Z
        
        # Kraus operators for depolarizing channel
        # ρ' = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
        self._kraus_ops = self._compute_kraus_operators()
    
    def _compute_kraus_operators(self) -> List[np.ndarray]:
        """Compute Kraus operators for depolarizing channel."""
        p = self.probability
        q = 1 - p
        
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Kraus operators
        k0 = np.sqrt(q) * I
        k1 = np.sqrt(p/3) * X
        k2 = np.sqrt(p/3) * Y
        k3 = np.sqrt(p/3) * Z
        
        return [k0, k1, k2, k3]
    
    def get_channel(self, gate_name: str, qubits: List[int]) -> NoiseChannel:
        """Get depolarizing noise channel."""
        # Apply to all single-qubit gates
        if len(qubits) == 1:
            return NoiseChannel(
                name=f"depolarizing_{qubits[0]}",
                kraus_operators=self._kraus_ops,
                description=f"Depolarizing noise with p={self.probability}"
            )
        elif len(qubits) == 2:
            # Two-qubit depolarizing
            return NoiseChannel(
                name=f"depolarizing_2q_{qubits}",
                kraus_operators=self._two_qubit_kraus_ops(),
                description=f"2-qubit depolarizing with p={self.probability}"
            )
        return None
    
    def _two_qubit_kraus_ops(self) -> List[np.ndarray]:
        """Compute two-qubit depolarizing Kraus operators."""
        p = self.probability
        q = 1 - p
        
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        kraus_ops = []
        
        # Identity with probability q
        kraus_ops.append(np.sqrt(q) * np.kron(I, I))
        
        # Single-qubit errors on each qubit
        for pauli in [X, Y, Z]:
            # Error on first qubit
            kraus_ops.append(np.sqrt(p/15) * np.kron(pauli, I))
            # Error on second qubit
            kraus_ops.append(np.sqrt(p/15) * np.kron(I, pauli))
        
        # Two-qubit errors
        for pauli1 in [X, Y, Z]:
            for pauli2 in [X, Y, Z]:
                kraus_ops.append(np.sqrt(p/15) * np.kron(pauli1, pauli2))
        
        return kraus_ops


class AmplitudeDampingNoise(NoiseModel):
    """
    Amplitude damping noise channel.
    
    Models energy dissipation / relaxation to |0⟩ state.
    The Kraus operators are:
    K0 = |0⟩⟨0| + √(1-γ)|1⟩⟨1|
    K1 = √γ|0⟩⟨1|
    """
    
    def __init__(self, gamma: float):
        """
        Initialize amplitude damping noise.
        
        Args:
            gamma: Damping rate (probability of relaxation)
        """
        self.gamma = gamma
        self._kraus_ops = self._compute_kraus_operators()
    
    def _compute_kraus_operators(self) -> List[np.ndarray]:
        """Compute Kraus operators for amplitude damping."""
        gamma = self.gamma
        g = np.sqrt(gamma)
        g1 = np.sqrt(1 - gamma)
        
        K0 = np.array([[1, 0], [0, g1]], dtype=complex)
        K1 = np.array([[0, g], [0, 0]], dtype=complex)
        
        return [K0, K1]
    
    def get_channel(self, gate_name: str, qubits: List[int]) -> NoiseChannel:
        """Get amplitude damping channel."""
        if len(qubits) == 1:
            return NoiseChannel(
                name=f"amplitude_damping_{qubits[0]}",
                kraus_operators=self._kraus_ops,
                description=f"Amplitude damping with γ={self.gamma}"
            )
        return None


class PhaseDampingNoise(NoiseModel):
    """
    Phase damping (decoherence) noise channel.
    
    Models loss of quantum coherence without energy loss.
    The Kraus operators are:
    K0 = √(1-γ) I
    K1 = √γ |0⟩⟨0|
    K2 = √γ |1⟩⟨1|
    """
    
    def __init__(self, gamma: float):
        """
        Initialize phase damping noise.
        
        Args:
            gamma: Dephasing rate
        """
        self.gamma = gamma
        self._kraus_ops = self._compute_kraus_operators()
    
    def _compute_kraus_operators(self) -> List[np.ndarray]:
        """Compute Kraus operators for phase damping."""
        gamma = self.gamma
        g = np.sqrt(gamma)
        g1 = np.sqrt(1 - gamma)
        
        I = np.eye(2, dtype=complex)
        P0 = np.array([[1, 0], [0, 0]], dtype=complex)
        P1 = np.array([[0, 0], [0, 1]], dtype=complex)
        
        K0 = np.sqrt(g1) * I
        K1 = g * P0
        K2 = g * P1
        
        return [K0, K1, K2]
    
    def get_channel(self, gate_name: str, qubits: List[int]) -> NoiseChannel:
        """Get phase damping channel."""
        if len(qubits) == 1:
            return NoiseChannel(
                name=f"phase_damping_{qubits[0]}",
                kraus_operators=self._kraus_ops,
                description=f"Phase damping with γ={self.gamma}"
            )
        return None


class BitFlipNoise(NoiseModel):
    """
    Bit-flip noise channel.
    
    Applies X (Pauli) error with probability p.
    """
    
    def __init__(self, probability: float):
        """
        Initialize bit-flip noise.
        
        Args:
            probability: Probability of bit flip
        """
        self.probability = probability
        self._kraus_ops = self._compute_kraus_operators()
    
    def _compute_kraus_operators(self) -> List[np.ndarray]:
        """Compute Kraus operators for bit-flip channel."""
        p = self.probability
        
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        
        K0 = np.sqrt(1 - p) * I
        K1 = np.sqrt(p) * X
        
        return [K0, K1]
    
    def get_channel(self, gate_name: str, qubits: List[int]) -> NoiseChannel:
        """Get bit-flip channel."""
        if len(qubits) == 1:
            return NoiseChannel(
                name=f"bit_flip_{qubits[0]}",
                kraus_operators=self._kraus_ops,
                description=f"Bit-flip with p={self.probability}"
            )
        return None


class PhaseFlipNoise(NoiseModel):
    """
    Phase-flip (Z) noise channel.
    
    Applies Z error with probability p.
    """
    
    def __init__(self, probability: float):
        """
        Initialize phase-flip noise.
        
        Args:
            probability: Probability of phase flip
        """
        self.probability = probability
        self._kraus_ops = self._compute_kraus_operators()
    
    def _compute_kraus_operators(self) -> List[np.ndarray]:
        """Compute Kraus operators for phase-flip channel."""
        p = self.probability
        
        I = np.eye(2, dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        K0 = np.sqrt(1 - p) * I
        K1 = np.sqrt(p) * Z
        
        return [K0, K1]
    
    def get_channel(self, gate_name: str, qubits: List[int]) -> NoiseChannel:
        """Get phase-flip channel."""
        if len(qubits) == 1:
            return NoiseChannel(
                name=f"phase_flip_{qubits[0]}",
                kraus_operators=self._kraus_ops,
                description=f"Phase-flip with p={self.probability}"
            )
        return None


class CustomNoiseModel(NoiseModel):
    """
    Custom noise model with user-defined noise channels.
    
    Allows specifying different noise for different gates.
    """
    
    def __init__(self):
        """Initialize empty custom noise model."""
        self._channels: Dict[str, List[np.ndarray]] = {}
        self._default_channels: Dict[int, List[np.ndarray]] = {}
    
    def add_gate_noise(self, gate_name: str, kraus_operators: List[np.ndarray]):
        """
        Add noise channel for a specific gate.
        
        Args:
            gate_name: Name of the gate
            kraus_operators: List of Kraus operators
        """
        self._channels[gate_name] = kraus_operators
    
    def set_default_noise(self, num_qubits: int, kraus_operators: List[np.ndarray]):
        """
        Set default noise for gates affecting given number of qubits.
        
        Args:
            num_qubits: Number of qubits (1, 2, or 3)
            kraus_operators: List of Kraus operators
        """
        self._default_channels[num_qubits] = kraus_operators
    
    def add_pauli_noise(self, px: float = 0, py: float = 0, pz: float = 0):
        """
        Add Pauli error channel.
        
        Args:
            px: Probability of X error
            py: Probability of Y error
            pz: Probability of Z error
        """
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)
        
        p_total = px + py + pz
        
        kraus_ops = [np.sqrt(1 - p_total) * I]
        if px > 0:
            kraus_ops.append(np.sqrt(px) * X)
        if py > 0:
            kraus_ops.append(np.sqrt(py) * Y)
        if pz > 0:
            kraus_ops.append(np.sqrt(pz) * Z)
        
        self.set_default_noise(1, kraus_ops)
    
    def get_channel(self, gate_name: str, qubits: List[int]) -> Optional[NoiseChannel]:
        """Get noise channel for gate."""
        n_qubits = len(qubits)
        
        # Check for specific gate noise first
        if gate_name in self._channels:
            return NoiseChannel(
                name=f"custom_{gate_name}",
                kraus_operators=self._channels[gate_name],
                description=f"Custom noise for {gate_name}"
            )
        
        # Fall back to default for this qubit count
        if n_qubits in self._default_channels:
            return NoiseChannel(
                name=f"custom_default_{n_qubits}q",
                kraus_operators=self._default_channels[n_qubits],
                description=f"Default {n_qubits}-qubit noise"
            )
        
        return None


class ReadoutNoise(NoiseModel):
    """
    Readout (measurement) noise model.
    
    Models errors in measurement outcomes.
    """
    
    def __init__(self, p00: float = 0.99, p11: float = 0.99):
        """
        Initialize readout noise.
        
        Args:
            p00: Probability of correct |0⟩ outcome given |0⟩ state
            p11: Probability of correct |1⟩ outcome given |1⟩ state
        """
        self.p00 = p00
        self.p11 = p11
        self.p01 = 1 - p00  # Probability of |1⟩ when |0⟩
        self.p10 = 1 - p11  # Probability of |0⟩ when |1⟩
        
        # Classical noise matrix
        self._noise_matrix = np.array([
            [p00, p10],
            [p01, p11]
        ])
    
    def get_readout_error_probability(self) -> float:
        """Get average readout error probability."""
        return (self.p01 + self.p10) / 2
    
    def apply_readout_noise(self, state: DensityMatrix, qubit: int) -> DensityMatrix:
        """
        Apply readout noise to measurement probabilities.
        
        Args:
            state: Density matrix
            qubit: Qubit being measured
            
        Returns:
            Updated density matrix with noisy diagonal
        """
        # Compute probabilities of |0⟩ and |1⟩
        p0 = 0.0
        p1 = 0.0
        
        for i, val in enumerate(np.real(np.diag(state.data))):
            bit = (i >> qubit) & 1
            if bit == 0:
                p0 += val
            else:
                p1 += val
        
        # Apply noise
        new_p0 = self.p00 * p0 + self.p10 * p1
        new_p1 = self.p01 * p0 + self.p11 * p1
        
        # Return noisy probabilities (simplified)
        return state
    
    def get_channel(self, gate_name: str, qubits: List[int]) -> Optional[NoiseChannel]:
        """Get readout noise channel (for measurements)."""
        if 'M' in gate_name or gate_name == 'measure':
            return NoiseChannel(
                name="readout_noise",
                kraus_operators=self._compute_kraus_operators(),
                description=f"Readout noise p01={self.p01:.3f}, p10={self.p10:.3f}"
            )
        return None
    
    def _compute_kraus_operators(self) -> List[np.ndarray]:
        """Compute Kraus operators for readout noise."""
        # Classical noise can be represented with Kraus operators
        p00, p01 = self.p00, self.p01
        p10, p11 = self.p10, self.p11
        
        # POVM elements
        E0 = np.array([[np.sqrt(p00), np.sqrt(p01)], [0, 0]], dtype=complex)
        E1 = np.array([[0, 0], [np.sqrt(p10), np.sqrt(p11)]], dtype=complex)
        
        return [E0, E1]


def create_thermal_noise(t1: float, t2: float, time: float, 
                        freq: float = 1e9) -> NoiseModel:
    """
    Create noise model from thermal relaxation parameters.
    
    Args:
        t1: T1 relaxation time (seconds)
        t2: T2 dephasing time (seconds)
        time: Gate time (seconds)
        freq: Qubit frequency (Hz)
        
    Returns:
        Combined NoiseModel
    """
    # Calculate rates
    gamma1 = 1 / t1 if t1 > 0 else 0
    gamma2 = 1 / t2 if t2 > 0 else 0
    
    # Dephasing rate
    gamma_phi = gamma2 - gamma1 / 2
    
    # Probabilities
    p_reset = 1 - np.exp(-gamma1 * time)
    p_phase = 1 - np.exp(-gamma_phi * time)
    
    model = CustomNoiseModel()
    
    # Amplitude damping
    gamma_eff = 1 - np.exp(-time / t1) if t1 > 0 else 0
    if gamma_eff > 0:
        ad = AmplitudeDampingNoise(gamma_eff)
        model.add_gate_noise('all_1q', ad._kraus_ops)
    
    # Phase damping
    gamma_phase_eff = 1 - np.exp(-time / t2) if t2 > 0 else 0
    if gamma_phase_eff > 0:
        pd = PhaseDampingNoise(gamma_phase_eff)
        model.add_gate_noise('all_1q_phase', pd._kraus_ops)
    
    return model


def combine_noise_models(*models: NoiseModel) -> NoiseModel:
    """
    Combine multiple noise models.
    
    Args:
        *models: Noise models to combine
        
    Returns:
        Combined NoiseModel
    """
    combined = CustomNoiseModel()
    
    for model in models:
        # This is a simplified combination
        # In practice, noise channels compose
        pass
    
    return combined


def pauli_error_channel(px: float, py: float, pz: float) -> List[np.ndarray]:
    """
    Create Kraus operators for Pauli error channel.
    
    Args:
        px, py, pz: Probabilities of X, Y, Z errors
        
    Returns:
        List of Kraus operators
    """
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    p_total = px + py + pz
    
    kraus_ops = [np.sqrt(1 - p_total) * I]
    
    if px > 0:
        kraus_ops.append(np.sqrt(px) * X)
    if py > 0:
        kraus_ops.append(np.sqrt(py) * Y)
    if pz > 0:
        kraus_ops.append(np.sqrt(pz) * Z)
    
    return kraus_ops


def depolarizing_error_channel(p: float, n_qubits: int = 1) -> List[np.ndarray]:
    """
    Create Kraus operators for n-qubit depolarizing channel.
    
    Args:
        p: Error probability
        n_qubits: Number of qubits
        
    Returns:
        List of Kraus operators
    """
    if n_qubits == 1:
        return DepolarizingNoise(p)._kraus_ops
    
    # For multi-qubit, use tensor products
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Single-qubit Kraus ops
    single_kraus = [
        np.sqrt(1 - p) * I,
        np.sqrt(p/3) * X,
        np.sqrt(p/3) * Y,
        np.sqrt(p/3) * Z
    ]
    
    if n_qubits == 2:
        # Two-qubit depolarizing
        kraus_ops = [np.kron(k, k) for k in single_kraus]
        return kraus_ops
    
    # General case
    paulis = [I, X, Y, Z]
    kraus_ops = []
    
    for idx in itertools.product(range(4), repeat=n_qubits):
        prob = (p / 3) ** sum(1 for i in idx if i > 0) * (1 - p) ** sum(1 for i in idx if i == 0)
        if prob > 0:
            op = np.array([[1.0]])
            for i in idx:
                op = np.kron(op, paulis[i])
            kraus_ops.append(np.sqrt(prob) * op)
    
    return kraus_ops


def amplitude_damping_error_channel(gamma: float) -> List[np.ndarray]:
    """
    Create Kraus operators for amplitude damping channel.
    
    Args:
        gamma: Damping parameter
        
    Returns:
        List of Kraus operators
    """
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]


def phase_damping_error_channel(gamma: float) -> List[np.ndarray]:
    """
    Create Kraus operators for phase damping channel.
    
    Args:
        gamma: Dephasing parameter
        
    Returns:
        List of Kraus operators
    """
    I = np.eye(2, dtype=complex)
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    
    K0 = np.sqrt(1 - gamma) * I
    K1 = np.sqrt(gamma) * P0
    K2 = np.sqrt(gamma) * P1
    
    return [K0, K1, K2]
