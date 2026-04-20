"""
Variational Quantum Eigensolver (VQE) Module

Implements the VQE algorithm for finding ground state energies of Hamiltonians.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

from .circuit import QuantumCircuit
from .gates import H, X, Y, Z, RX, RY, RZ, CNOT, CZ, RZZ
from .simulator import Simulator, SimulationMethod


@dataclass
class VQEResult:
    """Result of VQE optimization."""
    energy: float
    parameters: np.ndarray
    expectation_values: Dict[str, float]
    iterations: int
    converged: bool
    circuit: QuantumCircuit
    energy_history: List[float]
    
    @property
    def optimal_circuit(self) -> QuantumCircuit:
        """Return the circuit with optimal parameters."""
        return self.circuit
    
    def __repr__(self):
        return (f"VQEResult(energy={self.energy:.6f}, "
                f"iterations={self.iterations}, converged={self.converged})")


class HamiltonianTerm:
    """Represents a single term in a Hamiltonian."""
    
    def __init__(self, coefficient: complex, paulis: List[str]):
        """
        Initialize a Hamiltonian term.
        
        Args:
            coefficient: Coefficient of the term (can be complex)
            paulis: List of Pauli operators ('I', 'X', 'Y', 'Z') for each qubit
        """
        self.coefficient = coefficient
        self.paulis = paulis
        self._num_qubits = len(paulis)
        
        # Validate
        for p in paulis:
            if p not in ['I', 'X', 'Y', 'Z']:
                raise ValueError(f"Invalid Pauli: {p}")
    
    @property
    def num_qubits(self) -> int:
        """Number of qubits this term acts on."""
        return self._num_qubits
    
    @property
    def is_identity(self) -> bool:
        """Check if this is just an identity term."""
        return all(p == 'I' for p in self.paulis)
    
    def __repr__(self):
        return f"{self.coefficient:.4f} * {''.join(self.paulis)}"


class Hamiltonian:
    """
    Represents a quantum Hamiltonian as a sum of Pauli terms.
    
    H = Σ c_i P_i where P_i are Pauli strings.
    """
    
    def __init__(self, terms: Optional[List[HamiltonianTerm]] = None):
        """
        Initialize Hamiltonian.
        
        Args:
            terms: List of HamiltonianTerm objects
        """
        self.terms = terms or []
        self._num_qubits = 0
        if self.terms:
            self._num_qubits = max(t.num_qubits for t in self.terms)
    
    def add_term(self, coefficient: complex, paulis: List[str]):
        """Add a term to the Hamiltonian."""
        self.terms.append(HamiltonianTerm(coefficient, paulis))
        self._num_qubits = max(self._num_qubits, len(paulis))
    
    @property
    def num_qubits(self) -> int:
        """Number of qubits needed."""
        return self._num_qubits
    
    def __add__(self, other: 'Hamiltonian') -> 'Hamiltonian':
        """Add two Hamiltonians."""
        return Hamiltonian(self.terms + other.terms)
    
    def __mul__(self, scalar: complex) -> 'Hamiltonian':
        """Multiply Hamiltonian by scalar."""
        new_terms = [
            HamiltonianTerm(t.coefficient * scalar, t.paulis)
            for t in self.terms
        ]
        return Hamiltonian(new_terms)
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'Hamiltonian':
        """
        Create Hamiltonian from matrix representation.
        
        Args:
            matrix: Hermitian matrix
            
        Returns:
            Hamiltonian in Pauli basis
        """
        n = int(np.log2(len(matrix)))
        
        # Pauli matrices
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        PAULIS = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        
        # Decompose matrix into Pauli basis
        terms = []
        
        for pauli_str in itertools.product(PAULIS.keys(), repeat=n):
            # Compute coefficient
            coeff = 0.0
            for idx in itertools.product([0, 1], repeat=n):
                # Get matrix element
                row = sum(idx[i] << i for i in range(n))
                col = sum(idx[i] << i for i in range(n))
                
                # Compute tr(M * P)
                prod = matrix[row, col]
                for q, (p, b) in enumerate(zip(pauli_str, idx)):
                    prod *= PAULIS[p][b, b]
                
                coeff += prod
            
            coeff /= 2**n
            
            if abs(coeff) > 1e-10:
                terms.append(HamiltonianTerm(coeff, list(pauli_str)))
        
        return cls(terms)
    
    def to_matrix(self) -> np.ndarray:
        """
        Convert Hamiltonian to matrix representation.
        
        Returns:
            Hermitian matrix
        """
        dim = 2 ** self._num_qubits
        matrix = np.zeros((dim, dim), dtype=complex)
        
        # Pauli matrices
        PAULIS = {
            'I': np.eye(2),
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]])
        }
        
        for term in self.terms:
            # Build Pauli string matrix
            pmatrix = PAULIS[term.paulis[0]]
            for p in term.paulis[1:]:
                pmatrix = np.kron(pmatrix, PAULIS[p])
            
            matrix += term.coefficient * pmatrix
        
        return matrix
    
    def __repr__(self):
        return f"Hamiltonian({len(self.terms)} terms, {self._num_qubits} qubits)"


class VariationalForm(ABC):
    """Abstract base class for variational forms (Ansätze)."""
    
    @abstractmethod
    def create_circuit(self, parameters: np.ndarray, 
                      circuit: Optional[QuantumCircuit] = None) -> QuantumCircuit:
        """
        Create parameterized circuit.
        
        Args:
            parameters: Variational parameters
            circuit: Optional existing circuit to add to
            
        Returns:
            Quantum circuit
        """
        pass
    
    @property
    @abstractmethod
    def num_parameters(self) -> int:
        """Number of parameters needed."""
        pass
    
    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits used."""
        pass


class HardwareEfficientAnsatz(VariationalForm):
    """
    Hardware-efficient ansatz that respects hardware connectivity.
    
    Consists of alternating layers of single-qubit rotations and 
    entangling layers.
    """
    
    def __init__(self, num_qubits: int, depth: int = 2,
                 entangling_gates: str = 'cx'):
        """
        Initialize hardware-efficient ansatz.
        
        Args:
            num_qubits: Number of qubits
            depth: Number of layers
            entangling_gates: Type of entangling gates ('cx', 'cz', 'cp')
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.entangling_gates = entangling_gates
    
    @property
    def num_parameters(self) -> int:
        """3 parameters per qubit per layer (for RY, RZ, RX)."""
        return self.num_qubits * self.depth * 3
    
    def create_circuit(self, parameters: np.ndarray,
                      circuit: Optional[QuantumCircuit] = None) -> QuantumCircuit:
        """Create the parameterized circuit."""
        if circuit is None:
            circuit = QuantumCircuit(self.num_qubits, "HEA")
        
        if len(parameters) != self.num_parameters:
            raise ValueError(f"Expected {self.num_parameters} parameters, "
                           f"got {len(parameters)}")
        
        idx = 0
        for layer in range(self.depth):
            # Single-qubit rotations
            for q in range(self.num_qubits):
                circuit.ry(q, parameters[idx]); idx += 1
                circuit.rz(q, parameters[idx]); idx += 1
                circuit.rx(q, parameters[idx]); idx += 1
            
            # Entangling layer
            for q in range(self.num_qubits - 1):
                if self.entangling_gates == 'cx':
                    circuit.cx(q, q + 1)
                elif self.entangling_gates == 'cz':
                    circuit.cz(q, q + 1)
                elif self.entangling_gates == 'cp':
                    circuit.cp(np.pi / 4, q, q + 1)
        
        return circuit


class UnitaryCoupledClusterAnsatz(VariationalForm):
    """
    Unitary Coupled Cluster ansatz for molecular systems.
    
    Approximates the UCC wavefunction using exponential of excitation operators.
    """
    
    def __init__(self, num_qubits: int, num_electrons: int,
                 excitations: Optional[List[Tuple[int, ...]]] = None):
        """
        Initialize UCC ansatz.
        
        Args:
            num_qubits: Total number of qubits
            num_electrons: Number of electrons
            excitations: List of excitation tuples (default: single excitations)
        """
        self.num_qubits = num_qubits
        self.num_electrons = num_electrons
        
        # Default to single excitations
        if excitations is None:
            self.excitations = self._generate_single_excitations()
        else:
            self.excitations = excitations
    
    def _generate_single_excitations(self) -> List[Tuple[int, int]]:
        """Generate single excitation operators."""
        excitations = []
        # HOMO -> LUMO excitations
        for i in range(self.num_electrons):
            for a in range(self.num_electrons, self.num_qubits):
                excitations.append((i, a))
        return excitations
    
    @property
    def num_parameters(self) -> int:
        """Number of excitation operators."""
        return len(self.excitations)
    
    def create_circuit(self, parameters: np.ndarray,
                      circuit: Optional[QuantumCircuit] = None) -> QuantumCircuit:
        """Create UCC circuit."""
        if circuit is None:
            circuit = QuantumCircuit(self.num_qubits, "UCC")
        
        # Initialize in Hartree-Fock state
        for i in range(self.num_electrons):
            circuit.x(i)
        
        # Apply excitations
        for theta, excitation in zip(parameters, self.excitations):
            i, a = excitation
            # Single excitation: e^{θ(a^†i - i†a)}
            # Approximated as: RYRY + CNOT decomposition
            circuit.x(i)  # |0⟩ -> |1⟩
            circuit.cx(i, a)
            circuit.rz(a, theta)
            circuit.cx(i, a)
            circuit.x(i)  # Back to |0⟩
        
        return circuit


class QAOAVariationalForm(VariationalForm):
    """
    QAOA-style ansatz for optimization problems.
    
    Alternates between problem Hamiltonian and mixing Hamiltonian.
    """
    
    def __init__(self, num_qubits: int, p: int = 1):
        """
        Initialize QAOA ansatz.
        
        Args:
            num_qubits: Number of qubits
            p: Number of QAOA layers
        """
        self.num_qubits = num_qubits
        self.p = p
    
    @property
    def num_parameters(self) -> int:
        """2p parameters (gamma and beta per layer)."""
        return 2 * self.p
    
    def create_circuit(self, parameters: np.ndarray,
                      circuit: Optional[QuantumCircuit] = None,
                      cost_hamiltonian: Optional[Hamiltonian] = None) -> QuantumCircuit:
        """Create QAOA circuit."""
        if len(parameters) != self.num_parameters:
            raise ValueError(f"Expected {self.num_parameters} parameters")
        
        if circuit is None:
            circuit = QuantumCircuit(self.num_qubits, "QAOA")
        
        # Initial superposition
        for q in range(self.num_qubits):
            circuit.h(q)
        
        # QAOA layers
        for p_idx in range(self.p):
            gamma = parameters[2 * p_idx]
            beta = parameters[2 * p_idx + 1]
            
            # Cost Hamiltonian (problem-specific)
            if cost_hamiltonian is not None:
                for term in cost_hamiltonian.terms:
                    if all(pauli == 'Z' or pauli == 'I' for pauli in term.paulis):
                        # Z term can be directly applied
                        for q, p in enumerate(term.paulis):
                            if p == 'Z':
                                circuit.rz(q, gamma * term.coefficient)
                    elif term.paulis.count('Z') == 2:
                        # ZZ interaction
                        qs = [q for q, p in enumerate(term.paulis) if p == 'Z']
                        circuit.rzz(qs[0], qs[1], gamma * term.coefficient)
            
            # Mixing Hamiltonian (X)
            for q in range(self.num_qubits):
                circuit.rx(q, beta)


class VQE:
    """
    Variational Quantum Eigensolver implementation.
    
    Finds the ground state energy of a Hamiltonian using variational optimization.
    
    Example:
        hamiltonian = Hamiltonian([...])
        ansatz = HardwareEfficientAnsatz(4, depth=2)
        
        vqe = VQE(ansatz, hamiltonian)
        result = vqe.run()
        print(f"Ground state energy: {result.energy}")
    """
    
    def __init__(self, variational_form: VariationalForm,
                 hamiltonian: Hamiltonian,
                 simulator: Optional[Simulator] = None,
                 optimizer: Optional[Callable] = None):
        """
        Initialize VQE.
        
        Args:
            variational_form: Parameterized quantum circuit (Ansatz)
            hamiltonian: Hamiltonian whose ground state we want
            simulator: Quantum simulator (default: state vector)
            optimizer: Classical optimizer function
        """
        self.variational_form = variational_form
        self.hamiltonian = hamiltonian
        self.simulator = simulator or Simulator(method=SimulationMethod.STATE_VECTOR)
        
        # Default optimizer: COBYLA
        self.optimizer = optimizer or self._cobyla_optimizer
    
    def _cobyla_optimizer(self, fun: Callable, x0: np.ndarray,
                         bounds: Optional[List[Tuple[float, float]]] = None,
                         max_iter: int = 1000) -> Tuple[np.ndarray, float]:
        """Simple gradient-free optimizer (simplified)."""
        x = x0.copy()
        best_x = x.copy()
        best_val = fun(x)
        
        step_size = 0.1
        tolerance = 1e-6
        
        for _ in range(max_iter):
            improved = False
            
            # Random direction search
            for _ in range(10):
                direction = np.random.randn(len(x))
                direction /= np.linalg.norm(direction)
                
                # Try positive direction
                x_plus = x + step_size * direction
                val_plus = fun(x_plus)
                
                if val_plus < best_val:
                    best_val = val_plus
                    best_x = x_plus.copy()
                    improved = True
                
                # Try negative direction
                x_minus = x - step_size * direction
                val_minus = fun(x_minus)
                
                if val_minus < best_val:
                    best_val = val_minus
                    best_x = x_minus.copy()
                    improved = True
            
            if improved:
                x = best_x.copy()
            else:
                step_size *= 0.9
            
            if step_size < tolerance:
                break
        
        return best_x, best_val
    
    def _gradient_descent(self, fun: Callable, x0: np.ndarray,
                         lr: float = 0.1, max_iter: int = 1000,
                         tol: float = 1e-6) -> Tuple[np.ndarray, float]:
        """Gradient descent optimizer."""
        x = x0.copy().astype(float)
        best_x = x.copy()
        best_val = fun(x)
        
        for i in range(max_iter):
            # Compute gradient numerically
            eps = 1e-5
            grad = np.zeros_like(x)
            for j in range(len(x)):
                x_plus = x.copy()
                x_plus[j] += eps
                grad[j] = (fun(x_plus) - fun(x)) / eps
            
            # Update
            x_new = x - lr * grad
            
            val = fun(x_new)
            if val < best_val:
                best_val = val
                best_x = x_new.copy()
            
            if np.linalg.norm(grad) < tol:
                break
            
            x = x_new
        
        return best_x, best_val
    
    def _compute_expectation(self, parameters: np.ndarray) -> float:
        """
        Compute expectation value of Hamiltonian for given parameters.
        
        Args:
            parameters: Variational parameters
            
        Returns:
            Expectation value <ψ(θ)|H|ψ(θ)⟩
        """
        # Create circuit with parameters
        circuit = self.variational_form.create_circuit(parameters)
        
        # Simulate
        result = self.simulator.run(circuit)
        
        # Compute expectation value
        energy = 0.0
        expectation_values = {}
        
        # Pauli matrices
        PAULIS = {
            'I': np.eye(2),
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]])
        }
        
        for term in self.hamiltonian.terms:
            # Build full Pauli operator
            if term.is_identity:
                energy += np.real(term.coefficient)
                expectation_values['I' * term.num_qubits] = 1.0
                continue
            
            # Tensor product of Pauli matrices
            full_op = PAULIS[term.paulis[0]]
            for p in term.paulis[1:]:
                full_op = np.kron(full_op, PAULIS[p])
            
            # Compute expectation
            exp_val = result.expectation(full_op)
            energy += np.real(term.coefficient * exp_val)
            
            # Store for debugging
            term_key = ''.join(term.paulis)
            expectation_values[term_key] = np.real(exp_val)
        
        return energy
    
    def run(self, initial_parameters: Optional[np.ndarray] = None,
           max_iterations: int = 1000,
           tol: float = 1e-6,
           verbose: bool = False) -> VQEResult:
        """
        Run VQE optimization.
        
        Args:
            initial_parameters: Starting parameters (random if None)
            max_iterations: Maximum optimization iterations
            tol: Convergence tolerance
            verbose: Print progress
            
        Returns:
            VQEResult with optimal parameters and energy
        """
        # Initialize parameters
        if initial_parameters is None:
            initial_parameters = np.random.randn(self.variational_form.num_parameters) * 0.01
        else:
            initial_parameters = np.array(initial_parameters)
        
        # Energy history
        energy_history = []
        
        # Optimization callback
        def callback(params, energy):
            energy_history.append(energy)
            if verbose:
                print(f"Iteration {len(energy_history)}: E = {energy:.6f}")
        
        # Objective function
        def objective(params):
            energy = self._compute_expectation(params)
            callback(params, energy)
            return energy
        
        # Run optimization
        try:
            optimal_params, optimal_energy = self.optimizer(
                objective,
                initial_parameters,
                max_iter=max_iterations
            )
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            optimal_params = initial_parameters
            optimal_energy = self._compute_expectation(initial_parameters)
        
        # Final circuit
        optimal_circuit = self.variational_form.create_circuit(optimal_params)
        
        # Compute final expectation values
        expectation_values = {}
        PAULIS = {
            'I': np.eye(2),
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]])
        }
        
        result = self.simulator.run(optimal_circuit)
        
        for term in self.hamiltonian.terms:
            if term.is_identity:
                expectation_values['I' * term.num_qubits] = 1.0
                continue
            
            full_op = PAULIS[term.paulis[0]]
            for p in term.paulis[1:]:
                full_op = np.kron(full_op, PAULIS[p])
            
            exp_val = result.expectation(full_op)
            expectation_values[''.join(term.paulis)] = np.real(exp_val)
        
        # Check convergence
        converged = len(energy_history) < max_iterations
        if len(energy_history) >= 2:
            converged = abs(energy_history[-1] - energy_history[-2]) < tol
        
        return VQEResult(
            energy=optimal_energy,
            parameters=optimal_params,
            expectation_values=expectation_values,
            iterations=len(energy_history),
            converged=converged,
            circuit=optimal_circuit,
            energy_history=energy_history
        )


# Helper functions for common Hamiltonians

def hydrogen_molecule_hamiltonian(bond_length: float = 0.735) -> Hamiltonian:
    """
    Create Hamiltonian for H2 molecule (simplified).
    
    Args:
        bond_length: Internuclear distance in Angstroms
        
    Returns:
        Hamiltonian for H2
    """
    # Simplified Hamiltonian coefficients (STO-3G basis)
    # These are example coefficients
    h = Hamiltonian()
    
    h.add_term(-0.8105, ['I', 'I'])
    h.add_term(0.1692, ['Z', 'I'])
    h.add_term(-0.2228, ['I', 'Z'])
    h.add_term(0.1686, ['Z', 'Z'])
    h.add_term(0.1205, ['X', 'X'])
    h.add_term(0.1205, ['Y', 'Y'])
    
    return h


def heisenberg_hamiltonian(num_qubits: int, jx: float = 1.0,
                           jy: float = 1.0, jz: float = 1.0) -> Hamiltonian:
    """
    Create Heisenberg model Hamiltonian.
    
    H = J Σ (Jx X_i X_j + Jy Y_i Y_j + Jz Z_i Z_j)
    
    Args:
        num_qubits: Number of qubits
        jx, jy, jz: Coupling strengths
        
    Returns:
        Heisenberg Hamiltonian
    """
    h = Hamiltonian()
    
    for i in range(num_qubits - 1):
        paulis_i = ['I'] * num_qubits
        paulis_j = ['I'] * num_qubits
        
        # XX term
        paulis_xx = ['I'] * num_qubits
        paulis_xx[i] = 'X'
        paulis_xx[i + 1] = 'X'
        h.add_term(jx, paulis_xx)
        
        # YY term
        paulis_yy = ['I'] * num_qubits
        paulis_yy[i] = 'Y'
        paulis_yy[i + 1] = 'Y'
        h.add_term(jy, paulis_yy)
        
        # ZZ term
        paulis_zz = ['I'] * num_qubits
        paulis_zz[i] = 'Z'
        paulis_zz[i + 1] = 'Z'
        h.add_term(jz, paulis_zz)
    
    return h


def transverse_field_ising(num_qubits: int, j: float = 1.0,
                          h: float = 1.0) -> Hamiltonian:
    """
    Create transverse-field Ising Hamiltonian.
    
    H = -J Σ Z_i Z_{i+1} - h Σ X_i
    
    Args:
        num_qubits: Number of qubits
        j: Coupling strength
        h: Transverse field strength
        
    Returns:
        Ising Hamiltonian
    """
    hamiltonian = Hamiltonian()
    
    # ZZ interactions
    for i in range(num_qubits - 1):
        paulis = ['I'] * num_qubits
        paulis[i] = 'Z'
        paulis[i + 1] = 'Z'
        hamiltonian.add_term(-j, paulis)
    
    # Transverse field (X terms)
    for i in range(num_qubits):
        paulis = ['I'] * num_qubits
        paulis[i] = 'X'
        hamiltonian.add_term(-h, paulis)
    
    return hamiltonian


import itertools
