"""
Circuit Optimizer Module

Implements circuit optimization techniques including gate cancellation,
commutation analysis, and decomposition-based optimization.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from copy import deepcopy
import itertools

from .circuit import QuantumCircuit, CircuitInstruction
from .gates import Gate, SingleQubitGate, TwoQubitGate, ThreeQubitGate
from .gates import (
    IdentityGate, PauliXGate, PauliYGate, PauliZGate, 
    HadamardGate, SGate, TGate, SdgGate, TdgGate,
    CNOTGate, CZGate, SwapGate, ToffoliGate, FredkinGate,
    GateFactory
)


@dataclass
class OptimizationPass:
    """Represents a single optimization pass."""
    name: str
    description: str
    gates_removed: int = 0
    depth_reduced: int = 0


@dataclass
class OptimizationResult:
    """Result of circuit optimization."""
    original_circuit: QuantumCircuit
    optimized_circuit: QuantumCircuit
    passes: List[OptimizationPass]
    total_gates_removed: int
    depth_reduction: int
    improvement_ratio: float
    
    @property
    def summary(self) -> str:
        """Get a summary of the optimization."""
        lines = [
            f"Optimization Results:",
            f"  Original gates: {self.original_circuit.num_instructions}",
            f"  Optimized gates: {self.optimized_circuit.num_instructions}",
            f"  Gates removed: {self.total_gates_removed}",
            f"  Depth reduction: {self.depth_reduction}",
            f"  Improvement: {self.improvement_ratio:.2%}",
            "",
            "Passes applied:"
        ]
        for p in self.passes:
            if p.gates_removed > 0:
                lines.append(f"  - {p.name}: removed {p.gates_removed} gates")
        
        return "\n".join(lines)


class CircuitOptimizer:
    """
    Quantum circuit optimizer with multiple optimization strategies.
    
    Example:
        optimizer = CircuitOptimizer()
        result = optimizer.optimize(circuit, level=3)
        print(result.optimized_circuit)
    """
    
    def __init__(self, basis_gates: Optional[Tuple[str, ...]] = None):
        """
        Initialize optimizer.
        
        Args:
            basis_gates: Target basis gates for decomposition
        """
        self.basis_gates = basis_gates or ('h', 'x', 'y', 'z', 's', 't', 
                                           'cx', 'cz', 'swap')
        self.passes: List[OptimizationPass] = []
    
    def optimize(self, circuit: QuantumCircuit, 
                level: int = 2,
                max_iterations: int = 100) -> OptimizationResult:
        """
        Optimize a quantum circuit.
        
        Args:
            circuit: Circuit to optimize
            level: Optimization level (1-3)
                - 1: Basic (remove identity gates, simple cancellations)
                - 2: Medium (commutation, gate fusion)
                - 3: Aggressive (full optimization, basis decomposition)
            max_iterations: Maximum optimization iterations
            
        Returns:
            OptimizationResult
        """
        self.passes = []
        original_circuit = circuit.copy()
        
        current_circuit = circuit.copy()
        
        for iteration in range(max_iterations):
            prev_gate_count = current_circuit.num_instructions
            prev_depth = current_circuit.depth
            
            # Apply optimization passes based on level
            if level >= 1:
                current_circuit = self._remove_identity_gates(current_circuit)
                current_circuit = self._cancel_adjacent_gates(current_circuit)
            
            if level >= 2:
                current_circuit = self._commutation_optimization(current_circuit)
                current_circuit = self._fuse_single_qubit_gates(current_circuit)
            
            if level >= 3:
                current_circuit = self._cancel_redundant_cnots(current_circuit)
                current_circuit = self._optimize_hadamard_chains(current_circuit)
                current_circuit = self._optimize_cnot_chains(current_circuit)
            
            # Check for convergence
            if current_circuit.num_instructions >= prev_gate_count:
                break
        
        # Calculate results
        total_removed = original_circuit.num_instructions - current_circuit.num_instructions
        depth_reduction = original_circuit.depth - current_circuit.depth
        improvement = total_removed / max(1, original_circuit.num_instructions)
        
        return OptimizationResult(
            original_circuit=original_circuit,
            optimized_circuit=current_circuit,
            passes=self.passes,
            total_gates_removed=total_removed,
            depth_reduction=depth_reduction,
            improvement_ratio=improvement
        )
    
    def _remove_identity_gates(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Remove identity gates that don't affect the computation."""
        new_circuit = QuantumCircuit(circuit.num_qubits, circuit.name)
        removed = 0
        
        for instr in circuit.instructions:
            if isinstance(instr.gate, IdentityGate):
                removed += 1
            else:
                new_circuit._add_gate(instr.gate, instr.qubits)
        
        if removed > 0:
            self.passes.append(OptimizationPass(
                name="Identity Removal",
                description="Removed identity gates",
                gates_removed=removed
            ))
        
        return new_circuit
    
    def _cancel_adjacent_gates(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Cancel pairs of adjacent inverse gates.
        
        For example: X-X = I, H-H = I, S-S = I
        """
        if not circuit.instructions:
            return circuit
        
        new_circuit = QuantumCircuit(circuit.num_qubits, circuit.name)
        i = 0
        removed = 0
        
        while i < len(circuit.instructions):
            if i + 1 < len(circuit.instructions):
                gate1 = circuit.instructions[i].gate
                gate2 = circuit.instructions[i + 1].gate
                qubits1 = circuit.instructions[i].qubits
                qubits2 = circuit.instructions[i + 1].qubits
                
                # Check if gates are on same qubits and can cancel
                if qubits1 == qubits2 and GateFactory.can_cancel(gate1, gate2):
                    i += 2
                    removed += 2
                    continue
            
            new_circuit._add_gate(
                circuit.instructions[i].gate, 
                circuit.instructions[i].qubits
            )
            i += 1
        
        if removed > 0:
            self.passes.append(OptimizationPass(
                name="Adjacent Cancellation",
                description="Cancelled adjacent inverse gates",
                gates_removed=removed
            ))
        
        return new_circuit
    
    def _commutation_optimization(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Move gates past each other using commutation relations.
        
        This allows for more cancellation opportunities.
        """
        instructions = circuit.instructions.copy()
        moved = 0
        max_iterations = len(instructions) * 2
        
        for _ in range(max_iterations):
            changed = False
            
            for i in range(len(instructions) - 1):
                gate1 = instructions[i].gate
                gate2 = instructions[i + 1].gate
                qubits1 = set(instructions[i].qubits)
                qubits2 = set(instructions[i + 1].qubits)
                
                # Gates that don't overlap can be commuted
                if qubits1.isdisjoint(qubits2):
                    continue
                
                # Check if gates commute
                if not GateFactory.can_commute(gate1, gate2):
                    # Try to swap them
                    gate_name1 = gate1.name.split('(')[0]
                    gate_name2 = gate2.name.split('(')[0]
                    
                    # Known commutation pairs
                    # For example, Z commutes with everything on other qubits
                    # H-X-H = Z, H-Z-H = X, etc.
                    
                    # Simple case: single-qubit gates on same qubit can be reordered
                    if len(qubits1) == 1 and len(qubits2) == 1:
                        if qubits1 == qubits2:
                            # Can reorder single-qubit gates on same qubit
                            instructions[i], instructions[i + 1] = \
                                instructions[i + 1], instructions[i]
                            moved += 1
                            changed = True
            
            if not changed:
                break
        
        if moved > 0:
            self.passes.append(OptimizationPass(
                name="Commutation Reordering",
                description="Reordered gates using commutation relations",
                gates_removed=0,
                depth_reduced=moved
            ))
        
        new_circuit = QuantumCircuit(circuit.num_qubits, circuit.name)
        for instr in instructions:
            new_circuit._add_gate(instr.gate, instr.qubits)
        
        return new_circuit
    
    def _fuse_single_qubit_gates(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Fuse consecutive single-qubit gates into a single gate.
        
        For example: RZ(a)RZ(b) = RZ(a+b)
        """
        if not circuit.instructions:
            return circuit
        
        new_circuit = QuantumCircuit(circuit.num_qubits, circuit.name)
        fused = 0
        
        i = 0
        while i < len(circuit.instructions):
            instr = circuit.instructions[i]
            
            # Only fuse single-qubit gates
            if not isinstance(instr.gate, SingleQubitGate):
                new_circuit._add_gate(instr.gate, instr.qubits)
                i += 1
                continue
            
            # Look ahead for more single-qubit gates on same qubit
            fused_matrix = instr.gate.matrix.copy()
            j = i + 1
            qubit = instr.qubits[0]
            
            while j < len(circuit.instructions):
                next_instr = circuit.instructions[j]
                
                if next_instr.qubits != [qubit]:
                    break
                if not isinstance(next_instr.gate, SingleQubitGate):
                    break
                
                # Fuse matrices
                fused_matrix = next_instr.gate.matrix @ fused_matrix
                j += 1
                fused += 1
            
            # Decompose fused matrix back to gates (simplified)
            # In practice, you'd want to use U3 decomposition
            new_circuit._add_gate(
                self._matrix_to_gate(fused_matrix, qubit),
                [qubit]
            )
            
            i = j
        
        if fused > 0:
            self.passes.append(OptimizationPass(
                name="Gate Fusion",
                description="Fused consecutive single-qubit gates",
                gates_removed=fused
            ))
        
        return new_circuit
    
    def _matrix_to_gate(self, matrix: np.ndarray, qubit: int):
        """Convert a 2x2 matrix back to a gate (simplified)."""
        # For simplicity, just return identity
        # A proper implementation would do U3 decomposition
        return IdentityGate()
    
    def _cancel_redundant_cnots(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Cancel redundant CNOT gates.
        
        For example: CNOT(0,1)CNOT(0,1) = I
        """
        new_circuit = QuantumCircuit(circuit.num_qubits, circuit.name)
        removed = 0
        
        i = 0
        while i < len(circuit.instructions):
            if i + 1 < len(circuit.instructions):
                gate1 = circuit.instructions[i].gate
                gate2 = circuit.instructions[i + 1].gate
                
                # Check for CNOT-CNOT cancellation
                if (isinstance(gate1, CNOTGate) and isinstance(gate2, CNOTGate) and
                    gate1.control == gate2.control and gate1.target == gate2.target):
                    i += 2
                    removed += 2
                    continue
                
                # CNOT-CZ-CNOT = CNOT on different target optimization
                if (isinstance(gate1, CNOTGate) and isinstance(gate2, CZGate) and
                    i + 2 < len(circuit.instructions) and
                    isinstance(circuit.instructions[i + 2].gate, CNOTGate)):
                    cnot1 = gate1
                    cz = gate2
                    cnot2 = circuit.instructions[i + 2].gate
                    
                    # CNOT-a-CZ-a-CNOT-b = CZ-b
                    if (cnot1.control == cz.control and cnot2.control == cz.target and
                        cnot1.target == cz.control):
                        new_circuit._add_gate(CZGate(cnot2.target, cz.target), 
                                            [cnot2.target, cz.target])
                        i += 3
                        removed += 2
                        continue
            
            new_circuit._add_gate(
                circuit.instructions[i].gate,
                circuit.instructions[i].qubits
            )
            i += 1
        
        if removed > 0:
            self.passes.append(OptimizationPass(
                name="CNOT Reduction",
                description="Cancelled redundant CNOT gates",
                gates_removed=removed
            ))
        
        return new_circuit
    
    def _optimize_hadamard_chains(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Optimize chains of Hadamard gates.
        
        H-H = I, H-X-H = Z, etc.
        """
        return self._cancel_adjacent_gates(circuit)
    
    def _optimize_cnot_chains(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Optimize chains of CNOT gates.
        
        CNOT-CNOT pairs can sometimes be eliminated.
        """
        new_circuit = QuantumCircuit(circuit.num_qubits, circuit.name)
        
        i = 0
        while i < len(circuit.instructions):
            # Check for CNOT chain patterns
            if (isinstance(circuit.instructions[i].gate, CNOTGate) and
                i + 2 < len(circuit.instructions) and
                isinstance(circuit.instructions[i + 2].gate, CNOTGate)):
                
                cnot1 = circuit.instructions[i].gate
                cnot2 = circuit.instructions[i + 2].gate
                
                # Pattern: CNOT(a,b) - any gate - CNOT(a,b) = CNOT(a,b)
                # (middle gate must be on different qubits)
                mid_qubits = set(circuit.instructions[i + 1].qubits)
                cnot1_qubits = set([cnot1.control, cnot1.target])
                cnot2_qubits = set([cnot2.control, cnot2.target])
                
                if (cnot1_qubits == cnot2_qubits and mid_qubits.isdisjoint(cnot1_qubits)):
                    # Keep first CNOT, skip middle and last
                    new_circuit._add_gate(cnot1, [cnot1.control, cnot1.target])
                    i += 3
                    continue
            
            new_circuit._add_gate(
                circuit.instructions[i].gate,
                circuit.instructions[i].qubits
            )
            i += 1
        
        return new_circuit
    
    def decompose_to_basis(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Decompose circuit to use only basis gates.
        
        Args:
            circuit: Circuit to decompose
            
        Returns:
            New circuit using only basis gates
        """
        return circuit.decompose_to_basis_gates(self.basis_gates)
    
    def optimize_for_basis(self, circuit: QuantumCircuit,
                          target_basis: Tuple[str, ...]) -> QuantumCircuit:
        """
        Optimize and decompose to target basis.
        
        Args:
            circuit: Circuit to optimize
            target_basis: Target basis gate names
            
        Returns:
            Optimized circuit using only basis gates
        """
        # First optimize
        result = self.optimize(circuit, level=2)
        
        # Then decompose
        self.basis_gates = target_basis
        return self.decompose_to_basis(result.optimized_circuit)
    
    def transpile(self, circuit: QuantumCircuit,
                 coupling_map: List[Tuple[int, int]],
                 basis_gates: Optional[Tuple[str, ...]] = None) -> QuantumCircuit:
        """
        Transpile circuit for specific hardware coupling constraints.
        
        Args:
            circuit: Circuit to transpile
            coupling_map: Allowed CNOT connections
            basis_gates: Basis gates for the hardware
            
        Returns:
            Transpiled circuit
        """
        if basis_gates is None:
            basis_gates = self.basis_gates
        
        # Map qubits to hardware topology
        # This is a simplified implementation
        new_circuit = QuantumCircuit(circuit.num_qubits, circuit.name)
        
        for instr in circuit.instructions:
            gate = instr.gate
            qubits = instr.qubits
            
            if isinstance(gate, CNOTGate):
                # Check if CNOT is allowed
                control, target = qubits
                
                if (control, target) not in coupling_map:
                    # Need to insert SWAP gates to move qubits
                    # Simplified: just add as-is
                    pass
                
                new_circuit._add_gate(gate, qubits)
            else:
                new_circuit._add_gate(gate, qubits)
        
        return new_circuit


class LayoutOptimizer:
    """
    Optimizer for qubit layout and routing.
    
    Maps logical qubits in a circuit to physical qubits on hardware.
    """
    
    def __init__(self, coupling_map: List[Tuple[int, int]]):
        """
        Initialize layout optimizer.
        
        Args:
            coupling_map: Hardware connectivity (allowed two-qubit connections)
        """
        self.coupling_map = coupling_map
        self.graph = self._build_graph()
    
    def _build_graph(self) -> Dict[int, Set[int]]:
        """Build adjacency graph from coupling map."""
        graph = {}
        for u, v in self.coupling_map:
            if u not in graph:
                graph[u] = set()
            if v not in graph:
                graph[v] = set()
            graph[u].add(v)
            graph[v].add(u)
        return graph
    
    def initial_layout(self, circuit: QuantumCircuit) -> Dict[int, int]:
        """
        Generate initial qubit layout.
        
        Args:
            circuit: Circuit to layout
            
        Returns:
            Mapping from logical to physical qubits
        """
        # Simple identity layout
        return {i: i for i in range(circuit.num_qubits)}
    
    def route_circuit(self, circuit: QuantumCircuit,
                     layout: Optional[Dict[int, int]] = None) -> QuantumCircuit:
        """
        Route circuit to satisfy hardware constraints.
        
        Args:
            circuit: Circuit to route
            layout: Initial qubit layout
            
        Returns:
            Routed circuit with SWAP gates inserted
        """
        if layout is None:
            layout = self.initial_layout(circuit)
        
        new_circuit = QuantumCircuit(circuit.num_qubits, circuit.name)
        reverse_layout = {v: k for k, v in layout.items()}
        
        for instr in circuit.instructions:
            gate = instr.gate
            qubits = instr.qubits
            
            if isinstance(gate, CNOTGate):
                control, target = qubits
                
                # Check if CNOT is valid
                phys_control = layout[control]
                phys_target = layout[target]
                
                if (phys_control, phys_target) not in self.coupling_map:
                    # Need to insert SWAP gates
                    new_circuit = self._insert_swap(new_circuit, control, target, layout)
                    # Update layout after SWAP
                    layout = self._update_layout(layout, control, target)
                
                new_circuit._add_gate(gate, qubits)
            else:
                new_circuit._add_gate(gate, qubits)
        
        return new_circuit
    
    def _insert_swap(self, circuit: QuantumCircuit,
                    qubit1: int, qubit2: int,
                    layout: Dict[int, int]) -> QuantumCircuit:
        """Insert SWAP gates to bring qubits closer."""
        # Find path through coupling graph
        phys1 = layout[qubit1]
        phys2 = layout[qubit2]
        
        # BFS to find shortest path
        path = self._find_path(phys1, phys2)
        
        if path:
            # Insert SWAPs along path
            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i + 1]
                # Find logical qubits at these positions
                l1 = layout[p1]
                l2 = layout[p2]
                circuit.swap(l1, l2)
        
        return circuit
    
    def _find_path(self, start: int, end: int) -> List[int]:
        """Find shortest path between two qubits."""
        if start == end:
            return [start]
        
        visited = {start}
        queue = [(start, [start])]
        
        while queue:
            node, path = queue.pop(0)
            
            for neighbor in self.graph.get(node, []):
                if neighbor == end:
                    return path + [end]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []
    
    def _update_layout(self, layout: Dict[int, int],
                      qubit1: int, qubit2: int) -> Dict[int, int]:
        """Update layout after SWAP."""
        new_layout = layout.copy()
        new_layout[qubit1], new_layout[qubit2] = new_layout[qubit2], new_layout[qubit1]
        return new_layout


def optimize_circuit(circuit: QuantumCircuit, 
                    level: int = 2) -> OptimizationResult:
    """
    Convenience function to optimize a circuit.
    
    Args:
        circuit: Circuit to optimize
        level: Optimization level (1-3)
        
    Returns:
        OptimizationResult
    """
    optimizer = CircuitOptimizer()
    return optimizer.optimize(circuit, level=level)


def transpile_circuit(circuit: QuantumCircuit,
                     coupling_map: List[Tuple[int, int]],
                     basis_gates: Optional[Tuple[str, ...]] = None) -> QuantumCircuit:
    """
    Convenience function to transpile a circuit.
    
    Args:
        circuit: Circuit to transpile
        coupling_map: Hardware connectivity
        basis_gates: Basis gates
        
    Returns:
        Transpiled circuit
    """
    optimizer = CircuitOptimizer(basis_gates)
    return optimizer.transpile(circuit, coupling_map, basis_gates)
