"""
Quantum Approximate Optimization Algorithm (QAOA) Module

Implements QAOA for solving combinatorial optimization problems.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

from .circuit import QuantumCircuit
from .gates import H, X, Y, Z, RX, RY, RZ, CNOT, CZ, RZZ, SWAP
from .simulator import Simulator, SimulationMethod
from .vqe import Hamiltonian, HamiltonianTerm, VQE, VariationalForm


@dataclass
class QAOAResult:
    """Result of QAOA optimization."""
    optimal_parameters: np.ndarray
    optimal_value: float
    expectation_history: List[float]
    samples: Dict[str, int]
    num_iterations: int
    converged: bool
    circuit: QuantumCircuit
    
    @property
    def most_likely_solution(self) -> str:
        """Get the most probable solution state."""
        if not self.samples:
            return ""
        return max(self.samples.items(), key=lambda x: x[1])[0]
    
    @property
    def success_probability(self) -> float:
        """Probability of measuring the optimal solution."""
        if not self.samples:
            return 0.0
        total = sum(self.samples.values())
        return self.samples.get(self.most_likely_solution, 0) / total
    
    def __repr__(self):
        return (f"QAOAResult(value={self.optimal_value:.4f}, "
                f"solution={self.most_likely_solution}, "
                f"prob={self.success_probability:.2%})")


class ProblemType(Enum):
    """Type of optimization problem."""
    MAXCUT = "maxcut"
    SAT = "sat"
    QUADRATIC = "quadratic"
    QUBO = "qubo"
    ISING = "ising"


class OptimizationProblem:
    """
    Abstract representation of an optimization problem.
    
    Subclasses implement specific problem types.
    """
    
    def __init__(self, num_variables: int):
        """
        Initialize optimization problem.
        
        Args:
            num_variables: Number of binary variables
        """
        self.num_variables = num_variables
    
    def cost_function(self, assignment: List[int]) -> float:
        """
        Evaluate cost for a given assignment.
        
        Args:
            assignment: Binary assignment (0 or 1 for each variable)
            
        Returns:
            Cost value (to minimize) or negative for maximization
        """
        raise NotImplementedError
    
    def to_hamiltonian(self) -> Hamiltonian:
        """
        Convert problem to cost Hamiltonian for QAOA.
        
        Returns:
            Hamiltonian whose ground state encodes the solution
        """
        raise NotImplementedError
    
    def to_qubo_matrix(self) -> np.ndarray:
        """
        Get QUBO (Quadratic Unconstrained Binary Optimization) matrix.
        
        Returns:
            Q matrix such that cost = x^T Q x
        """
        raise NotImplementedError


class MaxCutProblem(OptimizationProblem):
    """
    Maximum Cut problem.
    
    Given a graph, find a partition that maximizes the number of
    crossing edges.
    
    Cost function: Number of edges with endpoints in different partitions
    """
    
    def __init__(self, edges: List[Tuple[int, int]]):
        """
        Initialize MaxCut problem.
        
        Args:
            edges: List of edges as (u, v) tuples
        """
        vertices = set()
        for u, v in edges:
            vertices.add(u)
            vertices.add(v)
        
        super().__init__(num_variables=max(vertices) + 1 if vertices else 0)
        self.edges = edges
    
    def cost_function(self, assignment: List[int]) -> float:
        """Compute number of crossing edges."""
        cut_size = 0
        for u, v in self.edges:
            if assignment[u] != assignment[v]:
                cut_size += 1
        return -cut_size  # Negative for maximization
    
    def to_hamiltonian(self) -> Hamiltonian:
        """Create cost Hamiltonian for MaxCut."""
        h = Hamiltonian()
        
        for u, v in self.edges:
            # Cost term: (1 - Z_u)(1 - Z_v)/4 = (1 - Z_u - Z_v + Z_u Z_v)/4
            # We maximize, so we minimize -C = -Σ (1 - Z_i Z_j)/2
            paulis = ['I'] * self.num_variables
            paulis[u] = 'Z'
            paulis[v] = 'Z'
            h.add_term(0.5, paulis)
            
            # Linear Z terms
            paulis_z = ['I'] * self.num_variables
            paulis_z[u] = 'Z'
            h.add_term(-0.5, paulis_z)
            
            paulis_z = ['I'] * self.num_variables
            paulis_z[v] = 'Z'
            h.add_term(-0.5, paulis_z)
        
        return h
    
    def to_qubo_matrix(self) -> np.ndarray:
        """Get QUBO matrix for MaxCut."""
        Q = np.zeros((self.num_variables, self.num_variables))
        
        for u, v in self.edges:
            # Off-diagonal: edge contributes to u,v
            Q[u, v] = 1
            Q[v, u] = 1
        
        return Q
    
    def optimal_solution(self) -> Tuple[float, List[List[int]]]:
        """Find optimal solution by brute force (for small instances)."""
        best_cost = float('-inf')
        best_assignments = []
        
        for bits in range(2 ** self.num_variables):
            assignment = [(bits >> i) & 1 for i in range(self.num_variables)]
            cost = -self.cost_function(assignment)  # Convert to maximization
            
            if cost > best_cost:
                best_cost = cost
                best_assignments = [assignment]
            elif cost == best_cost:
                best_assignments.append(assignment)
        
        return best_cost, best_assignments


class QuadraticProblem(OptimizationProblem):
    """
    General Quadratic Unconstrained Binary Optimization (QUBO).
    
    Minimizes: x^T Q x where x ∈ {0,1}^n
    """
    
    def __init__(self, Q: np.ndarray):
        """
        Initialize QUBO problem.
        
        Args:
            Q: QUBO matrix (n x n)
        """
        super().__init__(num_variables=len(Q))
        self.Q = np.array(Q)
    
    def cost_function(self, assignment: List[int]) -> float:
        """Compute x^T Q x."""
        x = np.array(assignment)
        return float(x @ self.Q @ x)
    
    def to_hamiltonian(self) -> Hamiltonian:
        """Convert QUBO to Ising Hamiltonian."""
        h = Hamiltonian()
        n = self.num_variables
        
        # QUBO to Ising: x_i = (1 - z_i)/2
        # Cost = Σ Q_ij (1 - z_i)(1 - z_j)/4
        
        # Linear terms
        for i in range(n):
            coeff = sum(self.Q[i, j] for j in range(n)) / 4
            if abs(coeff) > 1e-10:
                paulis = ['I'] * n
                paulis[i] = 'Z'
                h.add_term(coeff, paulis)
        
        # Quadratic terms
        for i in range(n):
            for j in range(i + 1, n):
                if abs(self.Q[i, j]) > 1e-10:
                    coeff = self.Q[i, j] / 4
                    paulis = ['I'] * n
                    paulis[i] = 'Z'
                    paulis[j] = 'Z'
                    h.add_term(coeff, paulis)
        
        # Constant term
        const = sum(self.Q[i, j] for i in range(n) for j in range(n)) / 4
        h.add_term(const, ['I'] * n)
        
        return h
    
    def to_qubo_matrix(self) -> np.ndarray:
        """Return the QUBO matrix."""
        return self.Q.copy()


class IsingProblem(OptimizationProblem):
    """
    Ising model optimization problem.
    
    Minimizes: Σ J_ij σ_i σ_j + Σ h_i σ_i
    where σ_i ∈ {-1, +1}
    """
    
    def __init__(self, j_matrix: Optional[np.ndarray] = None,
                 h_vector: Optional[np.ndarray] = None):
        """
        Initialize Ising problem.
        
        Args:
            j_matrix: Coupling matrix J_ij (n x n)
            h_vector: External field h_i (n,)
        """
        n = len(j_matrix) if j_matrix is not None else len(h_vector)
        super().__init__(num_variables=n)
        self.J = np.array(j_matrix) if j_matrix is not None else np.zeros((n, n))
        self.h = np.array(h_vector) if h_vector is not None else np.zeros(n)
    
    def cost_function(self, assignment: List[int]) -> float:
        """
        Compute Ising energy.
        
        Args:
            assignment: Spins as {-1, +1}
        """
        spins = np.array([1 if a else -1 for a in assignment])
        
        # Interaction term
        interaction = 0.0
        n = len(spins)
        for i in range(n):
            for j in range(i + 1, n):
                interaction += self.J[i, j] * spins[i] * spins[j]
        
        # External field term
        field = np.dot(self.h, spins)
        
        return interaction + field
    
    def to_hamiltonian(self) -> Hamiltonian:
        """Convert Ising to qubit Hamiltonian."""
        h = Hamiltonian()
        n = self.num_variables
        
        # ZZ interactions
        for i in range(n):
            for j in range(i + 1, n):
                if abs(self.J[i, j]) > 1e-10:
                    paulis = ['I'] * n
                    paulis[i] = 'Z'
                    paulis[j] = 'Z'
                    h.add_term(self.J[i, j] / 4, paulis)
        
        # Linear Z terms
        for i in range(n):
            coeff = self.J[i, i] / 2 if i < len(np.diag(self.J)) else 0
            coeff += self.h[i] / 2
            if abs(coeff) > 1e-10:
                paulis = ['I'] * n
                paulis[i] = 'Z'
                h.add_term(coeff, paulis)
        
        # Constant
        const = sum(self.J[i, j] for i in range(n) for j in range(n)) / 4
        const += sum(self.h) / 2
        h.add_term(const, ['I'] * n)
        
        return h


class QAOA:
    """
    Quantum Approximate Optimization Algorithm.
    
    QAOA approximates the solution to combinatorial optimization problems
    by preparing a parameterized quantum state and optimizing the parameters
    to maximize (or minimize) the expected cost.
    
    Example:
        # MaxCut problem
        problem = MaxCutProblem([(0,1), (1,2), (2,3)])
        qaoa = QAOA(problem, p=3)
        result = qaoa.run()
        print(f"Best cut: {result.optimal_value}")
        print(f"Partition: {result.most_likely_solution}")
    """
    
    def __init__(self, problem: OptimizationProblem,
                 p: int = 1,
                 simulator: Optional[Simulator] = None,
                 optimizer: Optional[Callable] = None):
        """
        Initialize QAOA.
        
        Args:
            problem: Optimization problem to solve
            p: Number of QAOA layers (depth parameter)
            simulator: Quantum simulator
            optimizer: Classical optimizer function
        """
        self.problem = problem
        self.p = p
        self.simulator = simulator or Simulator(method=SimulationMethod.STATE_VECTOR)
        self.optimizer = optimizer or self._default_optimizer
    
    def _default_optimizer(self, fun: Callable, x0: np.ndarray,
                         max_iter: int = 500) -> Tuple[np.ndarray, float]:
        """Default gradient-free optimizer."""
        x = x0.copy()
        best_x = x.copy()
        best_val = fun(x)
        
        step = 0.1
        tolerance = 1e-6
        
        for _ in range(max_iter):
            improved = False
            
            # Random direction search
            for _ in range(5):
                direction = np.random.randn(len(x))
                direction /= np.linalg.norm(direction)
                
                # Try both directions
                for sign in [1, -1]:
                    x_new = x + sign * step * direction
                    val = fun(x_new)
                    
                    if val < best_val:
                        best_val = val
                        best_x = x_new.copy()
                        improved = True
            
            if improved:
                x = best_x.copy()
            else:
                step *= 0.95
            
            if step < tolerance:
                break
        
        return best_x, best_val
    
    def create_circuit(self, parameters: np.ndarray,
                      initial_state: Optional[QuantumCircuit] = None
                      ) -> QuantumCircuit:
        """
        Create QAOA circuit with given parameters.
        
        Args:
            parameters: Parameters [gamma_0, beta_0, gamma_1, beta_1, ...]
            initial_state: Optional initial circuit (default: superposition)
            
        Returns:
            QAOA circuit
        """
        if len(parameters) != 2 * self.p:
            raise ValueError(f"Expected {2 * self.p} parameters, got {len(parameters)}")
        
        circuit = initial_state or QuantumCircuit(self.problem.num_variables, "QAOA")
        
        # Get cost Hamiltonian
        cost_h = self.problem.to_hamiltonian()
        
        # Initial superposition
        if initial_state is None:
            for q in range(self.problem.num_variables):
                circuit.h(q)
        
        # QAOA layers
        for p_idx in range(self.p):
            gamma = parameters[2 * p_idx]
            beta = parameters[2 * p_idx + 1]
            
            # Cost unitary exp(-i gamma C)
            self._apply_cost_unitary(circuit, cost_h, gamma)
            
            # Mixing unitary exp(-i beta B)
            self._apply_mixing_unitary(circuit, beta)
        
        return circuit
    
    def _apply_cost_unitary(self, circuit: QuantumCircuit,
                           hamiltonian: Hamiltonian,
                           gamma: float):
        """Apply cost Hamiltonian unitary."""
        n = self.problem.num_variables
        
        for term in hamiltonian.terms:
            if term.is_identity:
                continue
            
            # Count Z operators
            z_indices = [i for i, p in enumerate(term.paulis) if p == 'Z']
            
            if len(z_indices) == 0:
                # Identity - no action
                continue
            elif len(z_indices) == 1:
                # Single Z rotation
                i = z_indices[0]
                circuit.rz(i, 2 * gamma * np.real(term.coefficient))
            elif len(z_indices) == 2:
                # ZZ interaction
                i, j = z_indices
                # RZZ = exp(-i theta zz/4) Rz_i Rz_j CNOT CNOT
                # Or use controlled rotations
                circuit.rzz(i, j, 2 * gamma * np.real(term.coefficient))
            else:
                # Higher-order - approximate with CNOT cascade
                # RZ on last Z with CNOTs to control
                last_z = z_indices[-1]
                for i in z_indices[:-1]:
                    circuit.cx(i, last_z)
                circuit.rz(last_z, 2 * gamma * np.real(term.coefficient))
                for i in reversed(z_indices[:-1]):
                    circuit.cx(i, last_z)
    
    def _apply_mixing_unitary(self, circuit: QuantumCircuit, beta: float):
        """Apply mixing Hamiltonian unitary."""
        for q in range(self.problem.num_variables):
            circuit.rx(q, 2 * beta)
    
    def _compute_expectation(self, parameters: np.ndarray) -> float:
        """Compute expected cost value."""
        circuit = self.create_circuit(parameters)
        result = self.simulator.run(circuit)
        
        # Get probabilities
        probs = result.state.probabilities
        
        # Compute expected cost
        expectation = 0.0
        for i, prob in enumerate(probs):
            if prob > 1e-10:
                # Decode basis state
                bits = [(i >> q) & 1 for q in range(self.problem.num_variables)]
                cost = self.problem.cost_function(bits)
                expectation += prob * cost
        
        return expectation
    
    def run(self, initial_parameters: Optional[np.ndarray] = None,
           max_iterations: int = 500,
           tol: float = 1e-6,
           shots: int = 1024,
           verbose: bool = False) -> QAOAResult:
        """
        Run QAOA optimization.
        
        Args:
            initial_parameters: Starting parameters (random if None)
            max_iterations: Maximum optimization iterations
            tol: Convergence tolerance
            shots: Number of measurement shots for final sampling
            verbose: Print progress
            
        Returns:
            QAOAResult with optimal parameters and solution
        """
        # Initialize parameters
        if initial_parameters is None:
            initial_parameters = np.random.uniform(0, 2 * np.pi, 2 * self.p)
        else:
            initial_parameters = np.array(initial_parameters)
        
        # Track history
        history = []
        
        # Objective function
        def objective(params):
            val = self._compute_expectation(params)
            history.append(val)
            if verbose:
                print(f"  E = {val:.4f}")
            return val
        
        if verbose:
            print("Optimizing QAOA parameters...")
        
        # Optimize
        optimal_params, optimal_value = self.optimizer(
            objective,
            initial_parameters,
            max_iter=max_iterations
        )
        
        # Final circuit
        optimal_circuit = self.create_circuit(optimal_params)
        
        # Sample solutions
        samples = {}
        for _ in range(shots):
            result = self.simulator.run(optimal_circuit)
            outcome, _ = result.state.measure()
            bits = format(outcome, f'0{self.problem.num_variables}b')
            samples[bits] = samples.get(bits, 0) + 1
        
        # Check convergence
        converged = len(history) < max_iterations
        if len(history) >= 2:
            converged = abs(history[-1] - history[-2]) < tol
        
        return QAOAResult(
            optimal_parameters=optimal_params,
            optimal_value=optimal_value,
            expectation_history=history,
            samples=samples,
            num_iterations=len(history),
            converged=converged,
            circuit=optimal_circuit
        )
    
    def sample_wavefunction(self, parameters: np.ndarray,
                           shots: int = 1024) -> Dict[str, int]:
        """
        Sample from the QAOA wavefunction.
        
        Args:
            parameters: QAOA parameters
            shots: Number of samples
            
        Returns:
            Dictionary of measurement outcomes and counts
        """
        circuit = self.create_circuit(parameters)
        result = self.simulator.run(circuit)
        
        samples = {}
        for _ in range(shots):
            outcome, _ = result.state.measure()
            bits = format(outcome, f'0{self.problem.num_variables}b')
            samples[bits] = samples.get(bits, 0) + 1
        
        return samples


# Utility functions

def create_random_qubo(n: int, density: float = 0.5,
                      diagonal_bias: float = 0.0) -> QuadraticProblem:
    """
    Create a random QUBO problem.
    
    Args:
        n: Number of variables
        density: Fraction of non-zero entries
        diagonal_bias: Bias added to diagonal
        
    Returns:
        Random QuadraticProblem
    """
    Q = np.random.randn(n, n)
    
    # Symmetrize
    Q = (Q + Q.T) / 2
    
    # Random zero pattern
    mask = np.random.rand(n, n) > density
    Q[mask] = 0
    
    # Diagonal bias
    np.fill_diagonal(Q, np.diag(Q) + diagonal_bias)
    
    return QuadraticProblem(Q)


def create_random_maxcut(n: int, edge_probability: float = 0.5,
                        seed: Optional[int] = None) -> MaxCutProblem:
    """
    Create a random MaxCut problem.
    
    Args:
        n: Number of vertices
        edge_probability: Probability of each edge
        seed: Random seed
        
    Returns:
        Random MaxCutProblem
    """
    if seed is not None:
        np.random.seed(seed)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < edge_probability:
                edges.append((i, j))
    
    return MaxCutProblem(edges)


def maxcut_to_isings(hamiltonian: Hamiltonian) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert MaxCut Hamiltonian to Ising form.
    
    Returns:
        Tuple of (J matrix, h vector)
    """
    n = hamiltonian.num_qubits
    J = np.zeros((n, n))
    h = np.zeros(n)
    
    for term in hamiltonian.terms:
        if term.is_identity:
            continue
        
        z_indices = [i for i, p in enumerate(term.paulis) if p == 'Z']
        
        if len(z_indices) == 2:
            i, j = z_indices
            J[i, j] = np.real(term.coefficient)
            J[j, i] = np.real(term.coefficient)
        elif len(z_indices) == 1:
            i = z_indices[0]
            h[i] += np.real(term.coefficient)
    
    return J, h


def evaluate_qaoa_fidelity(qaoa: QAOA, parameters: np.ndarray,
                          optimal_assignment: List[int]) -> float:
    """
    Evaluate how well QAOA approximates the optimal solution.
    
    Args:
        qaoa: QAOA instance
        parameters: QAOA parameters
        optimal_assignment: Known optimal assignment
        
    Returns:
        Probability of measuring optimal assignment
    """
    circuit = qaoa.create_circuit(parameters)
    result = qaoa.simulator.run(circuit)
    
    # Find index of optimal state
    optimal_index = sum(optimal_assignment[i] << i for i in range(len(optimal_assignment)))
    
    return result.state.probabilities[optimal_index]


def qaoa_performance_ratio(qaoa: QAOA, result: QAOAResult,
                           problem: OptimizationProblem) -> float:
    """
    Compute QAOA approximation ratio.
    
    Args:
        qaoa: QAOA instance
        result: QAOA result
        problem: Original problem
        
    Returns:
        Approximation ratio (1.0 = optimal)
    """
    # Get expected value from QAOA
    qaoa_value = -result.optimal_value  # Convert back to maximization
    
    # Get optimal value
    if isinstance(problem, MaxCutProblem):
        optimal_value, _ = problem.optimal_solution()
    else:
        # Brute force for small problems
        best = float('-inf')
        for bits in range(2 ** problem.num_variables):
            assignment = [(bits >> i) & 1 for i in range(problem.num_variables)]
            cost = -problem.cost_function(assignment)  # Maximize
            best = max(best, cost)
        optimal_value = best
    
    if abs(optimal_value) < 1e-10:
        return 1.0
    
    return qaoa_value / optimal_value


# Import for type hints
from .vqe import VariationalForm
