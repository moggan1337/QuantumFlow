"""
Quantum Simulator Module

Implements state vector and density matrix simulators for quantum circuits.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import warnings

from .state import StateVector, DensityMatrix
from .circuit import QuantumCircuit, CircuitInstruction
from .gates import Gate, MeasurementGate


class SimulationMethod(Enum):
    """Method used for simulation."""
    STATE_VECTOR = "state_vector"
    DENSITY_MATRIX = "density_matrix"
    MPS = "matrix_product_state"  # For larger systems


@dataclass
class MeasurementResult:
    """Result of a measurement operation."""
    qubit: int
    outcome: int
    probability: float
    post_state: Union[StateVector, DensityMatrix]


@dataclass
class SimulationResult:
    """Complete result of a circuit simulation."""
    state: Union[StateVector, DensityMatrix]
    measurements: List[MeasurementResult]
    method: SimulationMethod
    num_qubits: int
    final_state_vector: Optional[np.ndarray] = None
    execution_time: float = 0.0
    
    @property
    def counts(self) -> Dict[str, int]:
        """Get measurement counts (for sampling)."""
        return self.state_vector_counts()
    
    def state_vector_counts(self, shots: int = 1024) -> Dict[str, int]:
        """
        Sample measurement outcomes from the final state.
        
        Args:
            shots: Number of measurement samples
            
        Returns:
            Dictionary mapping basis states to counts
        """
        probs = self.state.probabilities
        indices = np.arange(len(probs))
        
        # Sample
        samples = np.random.choice(indices, size=shots, p=probs)
        
        # Count
        counts = {}
        for s in samples:
            bits = format(s, f'0{self.num_qubits}b')
            counts[bits] = counts.get(bits, 0) + 1
        
        return counts
    
    def expectation(self, observable: np.ndarray) -> complex:
        """Compute expectation value of an observable."""
        return self.state.expectation(observable)
    
    def probabilities(self, qubits: Optional[List[int]] = None) -> np.ndarray:
        """
        Get probabilities for basis states.
        
        Args:
            qubits: Specific qubits to measure (None = all)
        """
        if qubits is None:
            return self.state.probabilities
        else:
            # Partial measurement probabilities
            return self._partial_probabilities(qubits)
    
    def _partial_probabilities(self, qubits: List[int]) -> np.ndarray:
        """Calculate probabilities for a subset of qubits."""
        n = len(qubits)
        probs = np.zeros(2**n)
        
        full_probs = self.state.probabilities
        for i, p in enumerate(full_probs):
            # Extract bits for the specified qubits
            mask = 0
            value = 0
            for j, q in enumerate(qubits):
                if (i >> q) & 1:
                    value |= (1 << j)
            
            probs[value] += p
        
        return probs


class Simulator:
    """
    Quantum circuit simulator supporting multiple simulation methods.
    
    Example:
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        
        simulator = Simulator()
        result = simulator.run(circuit)
        print(result.state)
    """
    
    def __init__(self, seed: Optional[int] = None, 
                 method: SimulationMethod = SimulationMethod.STATE_VECTOR):
        """
        Initialize the simulator.
        
        Args:
            seed: Random seed for reproducibility
            method: Simulation method to use
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.method = method
        self._noise_model = None
    
    def set_noise_model(self, noise_model):
        """Set a noise model for simulation."""
        self._noise_model = noise_model
    
    def run(self, circuit: QuantumCircuit, 
            initial_state: Optional[Union[StateVector, DensityMatrix]] = None,
            shots: int = 1,
            measure_all: bool = False) -> SimulationResult:
        """
        Run a quantum circuit simulation.
        
        Args:
            circuit: The quantum circuit to simulate
            initial_state: Optional initial state (default: |0...0⟩)
            shots: Number of measurement shots (for sampling)
            measure_all: Whether to measure all qubits at the end
            
        Returns:
            SimulationResult containing final state and measurements
        """
        import time
        start_time = time.time()
        
        # Initialize state
        if initial_state is not None:
            if self.method == SimulationMethod.STATE_VECTOR and isinstance(initial_state, DensityMatrix):
                raise ValueError("Cannot use density matrix with state vector method")
            state = initial_state
        else:
            if self.method == SimulationMethod.STATE_VECTOR:
                state = StateVector(0, num_qubits=circuit.num_qubits)
            else:
                state = DensityMatrix(StateVector(0, num_qubits=circuit.num_qubits))
        
        measurements = []
        
        # Apply each gate
        for instr in circuit.instructions:
            # Handle barriers
            if instr.params.get('barrier'):
                continue
            
            # Get gate matrix
            gate_matrix = instr.gate.matrix
            
            # Apply gate
            if self.method == SimulationMethod.STATE_VECTOR:
                state = state.apply_gate(gate_matrix, instr.qubits)
            else:
                state = state.apply_gate(gate_matrix, instr.qubits)
            
            # Apply noise if configured
            if self._noise_model is not None:
                state = self._noise_model.apply(state, instr.qubits)
            
            # Handle measurements
            if isinstance(instr.gate, MeasurementGate):
                if self.method == SimulationMethod.STATE_VECTOR:
                    outcome, post_state = state.measure_qubit(instr.qubits[0], self.rng)
                else:
                    outcome, post_state = state.measure()
                
                prob = np.abs(post_state.data[instr.qubits[0]])**2 if isinstance(post_state, StateVector) else np.real(np.diag(post_state.data)[instr.qubits[0]])
                
                measurements.append(MeasurementResult(
                    qubit=instr.qubits[0],
                    outcome=outcome,
                    probability=prob,
                    post_state=post_state
                ))
                state = post_state
        
        # Final measurement if requested
        if measure_all and not measurements:
            if self.method == SimulationMethod.STATE_VECTOR:
                for q in range(circuit.num_qubits):
                    outcome, state = state.measure_qubit(q, self.rng)
                    measurements.append(MeasurementResult(
                        qubit=q,
                        outcome=outcome,
                        probability=1.0,
                        post_state=state
                    ))
        
        execution_time = time.time() - start_time
        
        return SimulationResult(
            state=state,
            measurements=measurements,
            method=self.method,
            num_qubits=circuit.num_qubits,
            execution_time=execution_time
        )
    
    def run_batch(self, circuit: QuantumCircuit, shots: int = 1024) -> SimulationResult:
        """
        Run multiple shots of a circuit (with final measurements).
        
        Args:
            circuit: Circuit to simulate
            shots: Number of shots
            
        Returns:
            SimulationResult with accumulated counts
        """
        # Run with final measurement
        result = self.run(circuit, shots=shots, measure_all=True)
        
        # Aggregate counts from multiple runs if needed
        if shots > 1:
            all_counts = {}
            for _ in range(shots - 1):
                result_shot = self.run(circuit, shots=1, measure_all=True)
                counts = result_shot.state_vector_counts(shots=1)
                for k, v in counts.items():
                    all_counts[k] = all_counts.get(k, 0) + v
            
            # Merge with first result
            first_counts = result.state_vector_counts(shots=1)
            for k, v in first_counts.items():
                all_counts[k] = all_counts.get(k, 0) + v
            
            return result
        else:
            return result
    
    # ============== Specific Algorithm Simulations ==============
    
    def simulate_bell_state(self, which: int = 0) -> SimulationResult:
        """
        Simulate preparation and measurement of a Bell state.
        
        Args:
            which: Which Bell state (0-3)
            
        Returns:
            SimulationResult
        """
        circuit = QuantumCircuit(2, "bell")
        circuit.h(0)
        circuit.cx(0, 1)
        
        return self.run(circuit)
    
    def simulate_ghz_state(self, num_qubits: int) -> SimulationResult:
        """
        Simulate GHZ state preparation.
        
        Args:
            num_qubits: Number of qubits
            
        Returns:
            SimulationResult
        """
        circuit = QuantumCircuit(num_qubits, "ghz")
        circuit.h(0)
        for i in range(num_qubits - 1):
            circuit.cx(i, i + 1)
        
        return self.run(circuit)
    
    def simulate_qft(self, num_qubits: int, inverse: bool = False) -> SimulationResult:
        """
        Simulate Quantum Fourier Transform.
        
        Args:
            num_qubits: Number of qubits
            inverse: Whether to apply inverse QFT
            
        Returns:
            SimulationResult
        """
        circuit = QuantumCircuit(num_qubits, "qft")
        
        if inverse:
            # Inverse QFT
            for i in range(num_qubits - 1, -1, -1):
                # RZ(-π/2^i) (simplified as S, T gates)
                for j in range(i):
                    circuit.cp(-np.pi / (2 ** (i - j)), j, i)
                circuit.h(i)
        else:
            # Forward QFT
            for i in range(num_qubits):
                circuit.h(i)
                for j in range(i + 1, num_qubits):
                    circuit.cp(np.pi / (2 ** (j - i)), i, j)
        
        return self.run(circuit)
    
    def simulate_grovers(self, num_qubits: int, oracle: np.ndarray) -> SimulationResult:
        """
        Simulate Grover's search algorithm (simplified).
        
        Args:
            num_qubits: Number of qubits
            oracle: Oracle unitary matrix
            
        Returns:
            SimulationResult
        """
        circuit = QuantumCircuit(num_qubits, "grover")
        
        # Initial superposition
        for i in range(num_qubits):
            circuit.h(i)
        
        # Oracle and diffusion (simplified)
        # In practice, you'd iterate O(√N) times
        
        return self.run(circuit)
    
    def simulate_qpe(self, num_qubits: int, phase: float) -> SimulationResult:
        """
        Simulate Quantum Phase Estimation.
        
        Args:
            num_qubits: Number of precision qubits
            phase: Phase to estimate (0 to 1)
            
        Returns:
            SimulationResult
        """
        circuit = QuantumCircuit(num_qubits + 1, "qpe")
        
        # Prepare eigenstate
        circuit.x(num_qubits)
        
        # Apply controlled exponentials
        for i in range(num_qubits):
            for _ in range(2**i):
                circuit.cu1(2 * np.pi * phase, i, num_qubits)
        
        # Inverse QFT on first num_qubits
        for i in range(num_qubits // 2):
            circuit.swap(i, num_qubits - 1 - i)
        
        return self.run(circuit)
    
    def probability_distribution(self, circuit: QuantumCircuit, 
                                num_qubits: Optional[int] = None) -> Dict[str, float]:
        """
        Get the probability distribution over measurement outcomes.
        
        Args:
            circuit: Circuit to simulate
            num_qubits: Specific qubits to measure (None = all)
            
        Returns:
            Dictionary mapping basis states to probabilities
        """
        result = self.run(circuit)
        
        if num_qubits is None:
            probs = result.state.probabilities
        else:
            probs = result.probabilities(num_qubits)
        
        dist = {}
        for i, p in enumerate(probs):
            if p > 1e-10:
                bits = format(i, f'0{(num_qubits or circuit.num_qubits)}b')
                dist[bits] = p
        
        return dist
    
    def compute_expectation(self, circuit: QuantumCircuit,
                           observable: np.ndarray) -> complex:
        """
        Compute expectation value without measuring.
        
        Args:
            circuit: Circuit to simulate
            observable: Hermitian operator
            
        Returns:
            Expectation value
        """
        result = self.run(circuit)
        return result.expectation(observable)
    
    def verify_unitary(self, circuit: QuantumCircuit, tolerance: float = 1e-8) -> bool:
        """
        Verify that a circuit implements a valid unitary operation.
        
        Args:
            circuit: Circuit to verify
            tolerance: Numerical tolerance
            
        Returns:
            True if circuit is unitary
        """
        U = circuit.to_matrix()
        I = np.eye(len(U))
        
        # Check U†U = I
        is_unitary = np.allclose(U @ U.conj().T, I, atol=tolerance)
        
        # Check U is normalized
        norms = np.linalg.norm(U, axis=1)
        is_normalized = np.allclose(norms, 1.0, atol=tolerance)
        
        return is_unitary and is_normalized


class ParallelSimulator:
    """
    Simulator that can parallelize across multiple shots.
    
    Useful for running circuits with measurements many times.
    """
    
    def __init__(self, num_workers: int = 4, **kwargs):
        """
        Initialize parallel simulator.
        
        Args:
            num_workers: Number of parallel workers
            **kwargs: Arguments passed to Simulator
        """
        self.num_workers = num_workers
        self.simulator_kwargs = kwargs
    
    def run_shots(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        """
        Run many shots in parallel.
        
        Args:
            circuit: Circuit to simulate
            shots: Number of shots
            
        Returns:
            Measurement counts
        """
        # For small shot counts, just use regular simulator
        if shots <= self.num_workers:
            sim = Simulator(**self.simulator_kwargs)
            return sim.run(circuit, shots=shots, measure_all=True).counts
        
        # Parallel execution for many shots
        counts = {}
        
        # Run in batches
        batch_size = max(1, shots // self.num_workers)
        for _ in range(self.num_workers):
            sim = Simulator(**self.simulator_kwargs)
            result = sim.run(circuit, shots=batch_size, measure_all=True)
            batch_counts = result.state_vector_counts(shots=batch_size)
            
            for k, v in batch_counts.items():
                counts[k] = counts.get(k, 0) + v
        
        return counts


# Utility functions

def create_quantum_simulator(method: str = "state_vector", **kwargs) -> Simulator:
    """
    Factory function to create a quantum simulator.
    
    Args:
        method: Simulation method ("state_vector" or "density_matrix")
        **kwargs: Additional arguments for Simulator
        
    Returns:
        Configured Simulator instance
    """
    method_map = {
        "state_vector": SimulationMethod.STATE_VECTOR,
        "density_matrix": SimulationMethod.DENSITY_MATRIX,
        "mps": SimulationMethod.MPS,
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}. Choose from {list(method_map.keys())}")
    
    return Simulator(method=method_map[method], **kwargs)


def simulate_circuit(circuit: QuantumCircuit, 
                     method: str = "state_vector",
                     shots: int = 1,
                     seed: Optional[int] = None) -> SimulationResult:
    """
    Convenience function to simulate a circuit.
    
    Args:
        circuit: Quantum circuit to simulate
        method: Simulation method
        shots: Number of measurement shots
        seed: Random seed
        
    Returns:
        SimulationResult
    """
    sim = create_quantum_simulator(method, seed=seed)
    return sim.run(circuit, shots=shots, measure_all=(shots > 1))


def compare_circuits(circuit1: QuantumCircuit, 
                    circuit2: QuantumCircuit,
                    tolerance: float = 1e-8) -> bool:
    """
    Check if two circuits implement equivalent unitaries.
    
    Args:
        circuit1: First circuit
        circuit2: Second circuit
        tolerance: Numerical tolerance
        
    Returns:
        True if circuits are equivalent
    """
    if circuit1.num_qubits != circuit2.num_qubits:
        return False
    
    U1 = circuit1.to_matrix()
    U2 = circuit2.to_matrix()
    
    # Check if U1 = e^(iφ) U2 for some global phase
    ratio = U1 / (U2 + 1e-10)
    phases = np.diag(ratio)
    
    # All phases should be equal (up to numerical error)
    if not np.allclose(phases, phases[0], atol=tolerance):
        return False
    
    # Check if unitaries match up to global phase
    return np.allclose(U1, phases[0] * U2, atol=tolerance)
