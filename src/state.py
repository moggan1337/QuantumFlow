"""
Quantum State Module

Implements state vector and density matrix representations of quantum states.
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from functools import reduce


class StateVector:
    """
    Represents a pure quantum state as a state vector.
    
    The state vector is a complex-valued column vector of dimension 2^n,
    where n is the number of qubits.
    """
    
    def __init__(self, data: Union[np.ndarray, str, int], num_qubits: Optional[int] = None):
        """
        Initialize a state vector.
        
        Args:
            data: Can be:
                - numpy array: direct state vector data
                - string: basis state like "101" or "|101⟩"
                - int: basis state index (0 for |00...0⟩, etc.)
            num_qubits: Number of qubits (required if data is int)
        """
        if isinstance(data, np.ndarray):
            self._data = data.flatten().astype(complex)
        elif isinstance(data, str):
            # Parse basis state string like "101" or "|101⟩"
            clean = data.replace("|", "").replace("⟩", "").strip()
            if num_qubits is None:
                num_qubits = len(clean)
            self._data = self._basis_state(clean, num_qubits)
        elif isinstance(data, int):
            if num_qubits is None:
                raise ValueError("num_qubits must be provided for integer input")
            self._data = self._basis_state_index(data, num_qubits)
        else:
            raise TypeError(f"Invalid data type: {type(data)}")
        
        self._num_qubits = int(np.log2(len(self._data)))
        self._normalize()
    
    @staticmethod
    def _basis_state(bits: str, num_qubits: int) -> np.ndarray:
        """Create a basis state from bit string."""
        state = np.zeros(2**num_qubits, dtype=complex)
        index = int(bits, 2)
        state[index] = 1.0
        return state
    
    @staticmethod
    def _basis_state_index(index: int, num_qubits: int) -> np.ndarray:
        """Create a basis state from index."""
        state = np.zeros(2**num_qubits, dtype=complex)
        state[index] = 1.0
        return state
    
    def _normalize(self):
        """Normalize the state vector."""
        norm = np.linalg.norm(self._data)
        if norm > 0:
            self._data /= norm
    
    @property
    def data(self) -> np.ndarray:
        """Return the raw state vector data."""
        return self._data.copy()
    
    @property
    def num_qubits(self) -> int:
        """Return the number of qubits."""
        return self._num_qubits
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the Hilbert space."""
        return len(self._data)
    
    @property
    def probabilities(self) -> np.ndarray:
        """Return measurement probabilities for all computational basis states."""
        return np.abs(self._data)**2
    
    @property
    def entropy(self) -> float:
        """Return the von Neumann entropy."""
        probs = self.probabilities
        probs = probs[probs > 0]  # Avoid log(0)
        return -np.sum(probs * np.log2(probs))
    
    def amplitude(self, state: Union[str, int]) -> complex:
        """Get the amplitude for a specific basis state."""
        if isinstance(state, str):
            index = int(state.replace("|", "").replace("⟩", "").strip(), 2)
        else:
            index = state
        return self._data[index]
    
    def probs(self, indices: List[int]) -> np.ndarray:
        """Get probabilities for specific basis states."""
        return np.abs(self._data[indices])**2
    
    def measure(self, rng: Optional[np.random.Generator] = None) -> Tuple[int, 'StateVector']:
        """
        Measure the quantum state in the computational basis.
        
        Args:
            rng: Random number generator for reproducibility
            
        Returns:
            Tuple of (measurement outcome index, post-measurement state)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        probs = self.probabilities
        outcome = rng.choice(len(probs), p=probs)
        
        # Collapse to the measured state
        collapsed = np.zeros_like(self._data)
        collapsed[outcome] = 1.0
        
        return outcome, StateVector(collapsed)
    
    def measure_qubit(self, qubit: int, rng: Optional[np.random.Generator] = None) -> Tuple[int, 'StateVector']:
        """
        Measure a single qubit in the computational basis.
        
        Args:
            qubit: Index of the qubit to measure
            rng: Random number generator
            
        Returns:
            Tuple of (measurement outcome 0 or 1, post-measurement state)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Calculate probability of |0⟩ and |1⟩ for this qubit
        p0 = 0.0
        for i, amp in enumerate(self._data):
            # Check the qubit'th bit of the basis state index
            bit = (i >> qubit) & 1
            if bit == 0:
                p0 += np.abs(amp)**2
        
        # Sample measurement
        if rng.random() < p0:
            outcome = 0
        else:
            outcome = 1
        
        # Collapse the state
        collapsed = np.zeros_like(self._data)
        for i, amp in enumerate(self._data):
            bit = (i >> qubit) & 1
            if bit == outcome:
                collapsed[i] = amp / np.sqrt(p0 if outcome == 0 else (1 - p0))
        
        return outcome, StateVector(collapsed)
    
    def partial_trace(self, keep_qubits: List[int]) -> 'DensityMatrix':
        """
        Compute the partial trace over all qubits not in keep_qubits.
        
        Args:
            keep_qubits: List of qubit indices to keep
            
        Returns:
            DensityMatrix for the reduced system
        """
        # Reorder qubits to put kept qubits first
        n = self._num_qubits
        n_keep = len(keep_qubits)
        trace_qubits = [i for i in range(n) if i not in keep_qubits]
        
        # Reshape into tensor form
        shape = [2] * n
        tensor = self._data.reshape(shape)
        
        # Transpose to put kept qubits first
        order = keep_qubits + trace_qubits
        tensor = np.transpose(tensor, order)
        
        # Reshape and compute density matrix
        dim_keep = 2 ** n_keep
        dim_trace = 2 ** (n - n_keep)
        tensor = tensor.reshape((dim_keep, dim_trace))
        
        # Trace over traced qubits
        rho = np.tensordot(tensor, tensor.conj(), axes=([1], [1]))
        
        return DensityMatrix(rho)
    
    def apply_gate(self, gate_matrix: np.ndarray, qubits: List[int]) -> 'StateVector':
        """
        Apply a gate to specified qubits.
        
        Args:
            gate_matrix: Unitary matrix of the gate
            qubits: List of qubit indices to apply the gate to
            
        Returns:
            New StateVector with the gate applied
        """
        n = self._num_qubits
        dim = 2 ** n
        
        # Build the full unitary operator
        full_unitary = self._expand_unitary(gate_matrix, qubits, n)
        
        # Apply and normalize
        new_data = full_unitary @ self._data
        return StateVector(new_data)
    
    @staticmethod
    def _expand_unitary(unitary: np.ndarray, qubits: List[int], n_qubits: int) -> np.ndarray:
        """
        Expand a unitary operator to act on n qubits.
        
        Args:
            unitary: The operator to expand
            qubits: Which qubits it acts on (in order)
            n_qubits: Total number of qubits
        """
        dim = 2 ** n_qubits
        result = np.eye(dim, dtype=complex)
        
        # Process qubits in reverse order for correct tensor product
        for q in reversed(qubits):
            # Create single-qubit operator expanded to full space
            expanded = StateVector._expand_single_qubit(unitary, q, n_qubits)
            result = expanded @ result
        
        return result
    
    @staticmethod
    def _expand_single_qubit(single_qubit_op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
        """Expand a single-qubit operator to full Hilbert space."""
        if single_qubit_op.shape == (2, 2):
            dim = 2 ** n_qubits
            result = np.zeros((dim, dim), dtype=complex)
            
            for i in range(dim):
                for j in range(dim):
                    # Check if this matrix element is relevant
                    # Only connect basis states differing at position qubit
                    i_other = i >> qubit
                    j_other = j >> qubit
                    i_bit = (i >> qubit) & 1
                    j_bit = (j >> qubit) & 1
                    i_same = (i & ((1 << qubit) - 1)) | ((i >> (qubit + 1)) << qubit)
                    j_same = (j & ((1 << qubit) - 1)) | ((j >> (qubit + 1)) << qubit)
                    
                    if i_same == j_same:
                        result[i, j] = single_qubit_op[i_bit, j_bit]
            
            return result
        
        return single_qubit_op
    
    def expectation(self, observable: np.ndarray) -> complex:
        """
        Compute expectation value of an observable.
        
        Args:
            observable: Hermitian operator matrix
            
        Returns:
            Expectation value
        """
        return np.vdot(self._data, observable @ self._data)
    
    def fidelity(self, other: 'StateVector') -> float:
        """
        Compute fidelity with another pure state.
        
        Args:
            other: Another StateVector
            
        Returns:
            Fidelity (0 to 1)
        """
        if self._num_qubits != other._num_qubits:
            raise ValueError("States must have same number of qubits")
        return np.abs(np.vdot(self._data, other._data))**2
    
    def __repr__(self):
        return f"StateVector({self._num_qubits} qubits, norm={np.linalg.norm(self._data):.6f})"
    
    def __str__(self):
        """Human-readable string representation."""
        lines = [f"StateVector ({self._num_qubits} qubits):"]
        
        for i, amp in enumerate(self._data):
            if np.abs(amp) > 1e-10:
                bits = format(i, f'0{self._num_qubits}b')
                lines.append(f"  |{bits}⟩: {amp:.6f}")
        
        return "\n".join(lines)


class DensityMatrix:
    """
    Represents a quantum state as a density matrix.
    
    Supports both pure states (rank 1) and mixed states (general density matrices).
    """
    
    def __init__(self, data: Union[np.ndarray, StateVector]):
        """
        Initialize a density matrix.
        
        Args:
            data: Either a density matrix array or a StateVector (for pure states)
        """
        if isinstance(data, StateVector):
            # Convert state vector to density matrix
            self._data = np.outer(data.data, data.data.conj())
        elif isinstance(data, np.ndarray):
            self._data = data.astype(complex)
            self._normalize()
        else:
            raise TypeError(f"Invalid data type: {type(data)}")
        
        self._num_qubits = int(np.log2(len(self._data)))
    
    def _normalize(self):
        """Ensure the density matrix is properly normalized (trace 1)."""
        trace = np.trace(self._data)
        if np.abs(trace) > 1e-10:
            self._data /= trace
    
    @property
    def data(self) -> np.ndarray:
        """Return the raw density matrix data."""
        return self._data.copy()
    
    @property
    def num_qubits(self) -> int:
        """Return the number of qubits."""
        return self._num_qubits
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the Hilbert space."""
        return len(self._data)
    
    @property
    def purity(self) -> float:
        """Return the purity (Tr(ρ²))."""
        return np.real(np.trace(self._data @ self._data))
    
    @property
    def entropy(self) -> float:
        """Return the von Neumann entropy."""
        eigenvalues = np.linalg.eigvalsh(self._data)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    @property
    def is_pure(self) -> bool:
        """Check if the state is pure (purity = 1)."""
        return np.isclose(self.purity, 1.0)
    
    @property
    def is_valid(self) -> bool:
        """Check if this is a valid density matrix."""
        # Check trace = 1
        if not np.isclose(np.trace(self._data), 1.0):
            return False
        # Check Hermiticity
        if not np.allclose(self._data, self._data.conj().T):
            return False
        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(self._data)
        if np.any(eigenvalues < -1e-10):
            return False
        return True
    
    @staticmethod
    def mixed(p: List[float], states: List[StateVector]) -> 'DensityMatrix':
        """
        Create a mixed state from an ensemble.
        
        Args:
            p: Probabilities for each state
            states: List of StateVectors
            
        Returns:
            DensityMatrix representing the mixture
        """
        rho = np.zeros((2**states[0].num_qubits, 2**states[0].num_qubits), dtype=complex)
        for prob, state in zip(p, states):
            rho += prob * np.outer(state.data, state.data.conj())
        return DensityMatrix(rho)
    
    @staticmethod
    def maximally_mixed(num_qubits: int) -> 'DensityMatrix':
        """Create a maximally mixed state."""
        dim = 2 ** num_qubits
        rho = np.eye(dim, dtype=complex) / dim
        return DensityMatrix(rho)
    
    def measure(self, rng: Optional[np.random.Generator] = None) -> Tuple[int, 'DensityMatrix']:
        """
        Measure in the computational basis.
        
        Args:
            rng: Random number generator
            
        Returns:
            Tuple of (measurement outcome, post-measurement state)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Probabilities are diagonal elements
        probs = np.real(np.diag(self._data))
        outcome = rng.choice(len(probs), p=probs)
        
        # Construct projector
        dim = len(self._data)
        projector = np.zeros((dim, dim), dtype=complex)
        projector[outcome, outcome] = 1.0
        
        # Apply projector and normalize
        collapsed = projector @ self._data @ projector
        if np.trace(collapsed) > 0:
            collapsed /= np.trace(collapsed)
        
        return outcome, DensityMatrix(collapsed)
    
    def partial_trace(self, trace_qubits: List[int]) -> 'DensityMatrix':
        """
        Compute partial trace over specified qubits.
        
        Args:
            trace_qubits: Qubit indices to trace out
            
        Returns:
            Reduced density matrix
        """
        n = self._num_qubits
        n_keep = n - len(trace_qubits)
        
        # Reshape into tensor
        shape = [2] * (2 * n)
        rho_tensor = self._data.reshape(shape)
        
        # Trace over each qubit
        for q in sorted(trace_qubits):
            # Contract over this qubit's bra and ket
            pass
        
        # Simplified implementation using numpy
        # This is a basic partial trace
        keep_qubits = [i for i in range(n) if i not in trace_qubits]
        
        # Create permutation for traced qubits
        perm = keep_qubits + trace_qubits
        rho_perm = np.transpose(self._data.reshape([2]*2*n), perm + [n+i for i in perm])
        
        dim_keep = 2 ** n_keep
        dim_trace = 2 ** len(trace_qubits)
        rho_perm = rho_perm.reshape((dim_keep, dim_trace, dim_keep, dim_trace))
        
        # Trace
        rho_reduced = np.trace(rho_perm, axis1=1, axis2=3)
        
        return DensityMatrix(rho_reduced)
    
    def apply_gate(self, gate_matrix: np.ndarray, qubits: List[int]) -> 'DensityMatrix':
        """
        Apply a gate to the density matrix.
        
        Args:
            gate_matrix: Unitary operator
            qubits: Qubit indices to apply to
            
        Returns:
            Updated density matrix
        """
        n = self._num_qubits
        dim = 2 ** n
        
        # Build full unitary
        full_unitary = StateVector._expand_unitary(gate_matrix, qubits, n)
        
        # Apply: ρ' = U ρ U†
        new_rho = full_unitary @ self._data @ full_unitary.conj().T
        
        return DensityMatrix(new_rho)
    
    def apply_channel(self, kraus_ops: List[np.ndarray], qubits: List[int]) -> 'DensityMatrix':
        """
        Apply a quantum channel defined by Kraus operators.
        
        Args:
            kraus_ops: List of Kraus operators
            qubits: Qubit indices the channel acts on
            
        Returns:
            Updated density matrix
        """
        n = self._num_qubits
        dim = 2 ** n
        
        # Expand each Kraus operator
        result = np.zeros_like(self._data)
        for kraus in kraus_ops:
            # Expand Kraus operator to full space
            full_kraus = StateVector._expand_unitary(kraus, qubits, n)
            result += full_kraus @ self._data @ full_kraus.conj().T
        
        return DensityMatrix(result)
    
    def expectation(self, observable: np.ndarray) -> complex:
        """Compute expectation value of an observable."""
        return np.trace(observable @ self._data)
    
    def fidelity(self, other: 'DensityMatrix') -> float:
        """
        Compute fidelity with another density matrix.
        
        Uses the formula F(ρ, σ) = (Tr√(√ρ σ √ρ))²
        """
        if self._num_qubits != other._num_qubits:
            raise ValueError("States must have same number of qubits")
        
        # Use SVD-based approach for numerical stability
        tmp = np.linalg.solve(self._data, other._data)
        eigvals = np.linalg.eigvalsh(tmp)
        eigvals = np.abs(eigvals)
        eigvals = np.sqrt(eigvals)
        
        return np.sum(eigvals)**2
    
    def purity_of_subsystem(self, qubits: List[int]) -> float:
        """
        Compute purity of a subsystem.
        
        Args:
            qubits: Indices of qubits to consider
            
        Returns:
            Purity of reduced density matrix
        """
        reduced = self.partial_trace([i for i in range(self._num_qubits) if i not in qubits])
        return reduced.purity
    
    def __repr__(self):
        return f"DensityMatrix({self._num_qubits} qubits, purity={self.purity:.4f})"
    
    def __str__(self):
        """Human-readable representation."""
        lines = [f"DensityMatrix ({self._num_qubits} qubits):"]
        lines.append(f"  Purity: {self.purity:.6f}")
        lines.append(f"  Entropy: {self.entropy:.6f}")
        return "\n".join(lines)


def bell_state(which: int = 0) -> StateVector:
    """
    Create a Bell state.
    
    Args:
        which: Which Bell state (0=|Φ+⟩, 1=|Φ-⟩, 2=|Ψ+⟩, 3=|Ψ-⟩)
    
    Returns:
        StateVector representing the Bell state
    """
    bell_states = {
        0: [1, 0, 0, 1],  # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        1: [1, 0, 0, -1], # |Φ-⟩ = (|00⟩ - |11⟩)/√2
        2: [0, 1, 1, 0],  # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
        3: [0, 1, -1, 0], # |Ψ-⟩ = (|01⟩ - |10⟩)/√2
    }
    return StateVector(np.array(bell_states[which], dtype=complex) / np.sqrt(2))


def ghz_state(num_qubits: int) -> StateVector:
    """Create a GHZ state (|00...0⟩ + |11...1⟩)/√2."""
    dim = 2 ** num_qubits
    data = np.zeros(dim, dtype=complex)
    data[0] = 1 / np.sqrt(2)
    data[-1] = 1 / np.sqrt(2)
    return StateVector(data)


def w_state(num_qubits: int) -> StateVector:
    """Create a W state (single excitation)."""
    dim = 2 ** num_qubits
    data = np.zeros(dim, dtype=complex)
    for i in range(num_qubits):
        index = 1 << i  # Basis state with single 1 at position i
        data[index] = 1 / np.sqrt(num_qubits)
    return StateVector(data)


def superposition(*amplitudes: complex) -> StateVector:
    """Create a superposition from amplitudes."""
    data = np.array(list(amplitudes), dtype=complex)
    return StateVector(data)
