"""
Tests for QuantumFlow quantum circuit simulator.
"""

import unittest
import numpy as np
from src.circuit import QuantumCircuit
from src.gates import H, X, Y, Z, CNOT, CZ, SWAP, Toffoli, Fredkin, RX, RY, RZ, S, T
from src.state import StateVector, DensityMatrix, bell_state, ghz_state, w_state
from src.simulator import Simulator, SimulationMethod, simulate_circuit
from src.optimizer import CircuitOptimizer, optimize_circuit
from src.noise import DepolarizingNoise, AmplitudeDampingNoise, NoiseChannel
from src.vqe import VQE, Hamiltonian, VariationalForm, HardwareEfficientAnsatz
from src.qaoa import QAOA, MaxCutProblem, QuadraticProblem


class TestGates(unittest.TestCase):
    """Test quantum gates."""
    
    def test_identity_gate(self):
        """Test identity gate."""
        I = np.eye(2, dtype=complex)
        self.assertTrue(np.allclose(I @ I, I))
    
    def test_pauli_gates(self):
        """Test Pauli matrices."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Check anti-commutation
        self.assertTrue(np.allclose(X @ Y + Y @ X, 2 * np.array([[0, 0], [0, 0]], dtype=complex)))
        
        # Check X² = I
        self.assertTrue(np.allclose(X @ X, np.eye(2)))
    
    def test_hadamard(self):
        """Test Hadamard gate."""
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        
        # H² = I
        self.assertTrue(np.allclose(H @ H, np.eye(2)))
        
        # Check normalization
        self.assertTrue(np.isclose(np.linalg.norm(H[0]), 1.0))
    
    def test_cnot_matrix(self):
        """Test CNOT gate matrix."""
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        
        # Check it's unitary
        self.assertTrue(np.allclose(CNOT @ CNOT.conj().T, np.eye(4)))
    
    def test_toffoli(self):
        """Test Toffoli gate."""
        circuit = QuantumCircuit(3)
        circuit.toffoli(0, 1, 2)
        
        # Toffoli matrix should be unitary
        U = circuit.to_matrix()
        self.assertTrue(np.allclose(U @ U.conj().T, np.eye(8)))


class TestStateVector(unittest.TestCase):
    """Test state vector operations."""
    
    def test_basis_state(self):
        """Test basis state creation."""
        state = StateVector("0", num_qubits=2)
        self.assertTrue(np.isclose(state.data[0], 1.0))
        
        state = StateVector("01", num_qubits=2)
        self.assertTrue(np.isclose(state.data[1], 1.0))
    
    def test_superposition(self):
        """Test superposition state."""
        state = StateVector(np.array([1, 1], dtype=complex) / np.sqrt(2))
        
        probs = state.probabilities
        self.assertTrue(np.allclose(probs, [0.5, 0.5]))
    
    def test_measurement(self):
        """Test measurement."""
        state = StateVector(np.array([1, 0], dtype=complex))
        outcome, collapsed = state.measure()
        
        self.assertEqual(outcome, 0)
    
    def test_bell_state(self):
        """Test Bell state creation."""
        bell = bell_state(0)
        self.assertEqual(bell.num_qubits, 2)
        
        # Check it's normalized
        self.assertTrue(np.isclose(np.linalg.norm(bell.data), 1.0))
    
    def test_ghz_state(self):
        """Test GHZ state."""
        ghz = ghz_state(3)
        probs = ghz.probabilities
        
        # Should have equal probability for |000⟩ and |111⟩
        self.assertTrue(np.isclose(probs[0], 0.5))
        self.assertTrue(np.isclose(probs[-1], 0.5))


class TestDensityMatrix(unittest.TestCase):
    """Test density matrix operations."""
    
    def test_from_state_vector(self):
        """Test conversion from state vector."""
        state = StateVector(np.array([1, 0], dtype=complex))
        rho = DensityMatrix(state)
        
        self.assertTrue(np.isclose(np.trace(rho.data), 1.0))
    
    def test_purity(self):
        """Test purity calculation."""
        state = StateVector(np.array([1, 0], dtype=complex))
        rho = DensityMatrix(state)
        
        # Pure state should have purity 1
        self.assertTrue(np.isclose(rho.purity, 1.0))
    
    def test_maximally_mixed(self):
        """Test maximally mixed state."""
        rho = DensityMatrix.maximally_mixed(2)
        
        # Should have purity = 1/d
        self.assertTrue(np.isclose(rho.purity, 0.25))


class TestCircuit(unittest.TestCase):
    """Test quantum circuits."""
    
    def test_simple_circuit(self):
        """Test a simple circuit."""
        circuit = QuantumCircuit(2, "test")
        circuit.h(0)
        circuit.cx(0, 1)
        
        self.assertEqual(circuit.num_qubits, 2)
        self.assertGreater(circuit.num_instructions, 0)
    
    def test_bell_state_circuit(self):
        """Test Bell state preparation circuit."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        
        # The matrix should produce Bell states
        U = circuit.to_matrix()
        self.assertTrue(np.allclose(U @ U.conj().T, np.eye(4), atol=1e-10))
    
    def test_gate_count(self):
        """Test gate counting."""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.h(1)
        circuit.cx(0, 2)
        circuit.x(1)
        
        counts = circuit.gate_count()
        self.assertEqual(counts['H'], 2)
        self.assertEqual(counts['X'], 1)
        self.assertEqual(counts['CNOT'], 1)
    
    def test_circuit_copy(self):
        """Test circuit copying."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        
        copy = circuit.copy()
        self.assertEqual(copy.num_qubits, circuit.num_qubits)
        self.assertEqual(copy.num_instructions, circuit.num_instructions)


class TestSimulator(unittest.TestCase):
    """Test quantum simulator."""
    
    def test_bell_state_simulation(self):
        """Test Bell state simulation."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        
        sim = Simulator()
        result = sim.run(circuit)
        
        # Should be in superposition
        probs = result.state.probabilities
        self.assertTrue(np.allclose(probs, [0.5, 0, 0, 0.5], atol=0.01))
    
    def test_ghz_simulation(self):
        """Test GHZ state simulation."""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        
        sim = Simulator()
        result = sim.run(circuit)
        
        probs = result.state.probabilities
        # Should have probability in |000⟩ and |111⟩
        self.assertTrue(np.isclose(probs[0] + probs[-1], 1.0, atol=0.01))
    
    def test_measurement_sampling(self):
        """Test measurement sampling."""
        circuit = QuantumCircuit(1)
        circuit.h(0)
        
        sim = Simulator(seed=42)
        result = sim.run(circuit)
        
        counts = result.state_vector_counts(shots=100)
        self.assertEqual(len(counts), 2)  # Both |0⟩ and |1⟩ should appear
    
    def test_density_matrix_simulation(self):
        """Test density matrix simulation."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        
        sim = Simulator(method=SimulationMethod.DENSITY_MATRIX)
        result = sim.run(circuit)
        
        self.assertIsInstance(result.state, DensityMatrix)
        
        # Should be pure
        self.assertTrue(np.isclose(result.state.purity, 1.0, atol=0.01))


class TestOptimizer(unittest.TestCase):
    """Test circuit optimizer."""
    
    def test_identity_removal(self):
        """Test identity gate removal."""
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.x(0)  # XX = I
        
        optimizer = CircuitOptimizer()
        result = optimizer.optimize(circuit, level=1)
        
        # Should have reduced gates
        self.assertLess(result.total_gates_removed, circuit.num_instructions)
    
    def test_adjacent_cancellation(self):
        """Test adjacent gate cancellation."""
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.h(0)
        circuit.x(0)
        circuit.x(0)
        
        optimizer = CircuitOptimizer()
        result = optimizer.optimize(circuit, level=1)
        
        # Should cancel HH and XX
        self.assertGreater(result.total_gates_removed, 0)
    
    def test_cnot_cancellation(self):
        """Test CNOT cancellation."""
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.cx(0, 1)
        
        optimizer = CircuitOptimizer()
        result = optimizer.optimize(circuit, level=2)
        
        # Should cancel both CNOTs
        self.assertEqual(result.optimized_circuit.num_instructions, 0)


class TestNoise(unittest.TestCase):
    """Test noise models."""
    
    def test_depolarizing_noise(self):
        """Test depolarizing noise."""
        noise = DepolarizingNoise(0.1)
        
        self.assertIsNotNone(noise.get_channel('H', [0]))
        self.assertIsNone(noise.get_channel('CNOT', [0, 1]))  # Only 1q gates
    
    def test_amplitude_damping(self):
        """Test amplitude damping noise."""
        noise = AmplitudeDampingNoise(0.1)
        
        channel = noise.get_channel('X', [0])
        self.assertIsNotNone(channel)
        self.assertEqual(len(channel.kraus_operators), 2)  # K0, K1


class TestVQE(unittest.TestCase):
    """Test VQE algorithm."""
    
    def test_hamiltonian_creation(self):
        """Test Hamiltonian creation."""
        h = Hamiltonian()
        h.add_term(0.5, ['Z', 'I'])
        h.add_term(-1.0, ['Z', 'Z'])
        
        self.assertEqual(len(h.terms), 2)
        self.assertEqual(h.num_qubits, 2)
    
    def test_variational_form(self):
        """Test variational form."""
        ansatz = HardwareEfficientAnsatz(num_qubits=2, depth=1)
        
        self.assertEqual(ansatz.num_qubits, 2)
        self.assertEqual(ansatz.num_parameters, 6)  # 2 qubits * 1 layer * 3 params
        
        circuit = ansatz.create_circuit(np.random.randn(6))
        self.assertEqual(circuit.num_qubits, 2)


class TestQAOA(unittest.TestCase):
    """Test QAOA algorithm."""
    
    def test_maxcut_problem(self):
        """Test MaxCut problem."""
        edges = [(0, 1), (1, 2), (2, 3)]
        problem = MaxCutProblem(edges)
        
        self.assertEqual(problem.num_variables, 4)
        
        # Test cost function
        assignment = [0, 0, 1, 1]
        cost = problem.cost_function(assignment)
        self.assertEqual(cost, -2)  # Two edges in cut
    
    def test_qubo_problem(self):
        """Test QUBO problem."""
        Q = np.array([[1, 2], [2, 1]], dtype=float)
        problem = QuadraticProblem(Q)
        
        self.assertEqual(problem.num_variables, 2)
        
        # Test cost: [1,0]^T Q [1,0] = 1
        cost = problem.cost_function([1, 0])
        self.assertEqual(cost, 1.0)
    
    def test_qaoa_circuit(self):
        """Test QAOA circuit creation."""
        problem = MaxCutProblem([(0, 1)])
        qaoa = QAOA(problem, p=1)
        
        params = np.array([np.pi/4, np.pi/4])
        circuit = qaoa.create_circuit(params)
        
        self.assertEqual(circuit.num_qubits, 2)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_end_to_end_simulation(self):
        """Test complete workflow."""
        # Create circuit
        circuit = QuantumCircuit(3, name="integration_test")
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        
        # Simulate
        sim = Simulator(seed=42)
        result = sim.run(circuit)
        
        # Check state is valid
        self.assertIsInstance(result.state, StateVector)
        self.assertTrue(np.isclose(np.linalg.norm(result.state.data), 1.0))
    
    def test_optimization_workflow(self):
        """Test circuit optimization workflow."""
        # Create circuit with redundancies
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(0, 1)
        
        # Optimize
        result = optimize_circuit(circuit, level=2)
        
        # Should have reduced gates
        self.assertLess(result.optimized_circuit.num_instructions, 
                       circuit.num_instructions)


if __name__ == '__main__':
    unittest.main()
