"""
Quantum Circuit Module

Implements quantum circuits with gate operations and circuit manipulation.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from copy import deepcopy

from .gates import Gate, SingleQubitGate, TwoQubitGate, ThreeQubitGate
from .gates import H, X, Y, Z, CNOT, CZ, SWAP, Toffoli, Fredkin
from .gates import S, T, RX, RY, RZ, IdentityGate


@dataclass
class CircuitInstruction:
    """A single instruction in a quantum circuit."""
    gate: Gate
    qubits: List[int]
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        q = ",".join(map(str, self.qubits))
        return f"{self.gate.name} on qubit(s) [{q}]"


class QuantumCircuit:
    """
    A quantum circuit consisting of gates applied to qubits.
    
    Example:
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
    """
    
    def __init__(self, num_qubits: int, name: str = "circuit"):
        """
        Initialize a quantum circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
            name: Optional name for the circuit
        """
        self._num_qubits = num_qubits
        self._name = name
        self._instructions: List[CircuitInstruction] = []
        self._cregs: Dict[str, int] = {}  # Classical registers
        self._qubit_labels: Dict[int, str] = {}  # Labeled qubits
    
    @property
    def num_qubits(self) -> int:
        """Return the number of qubits."""
        return self._num_qubits
    
    @property
    def num_instructions(self) -> int:
        """Return the number of instructions."""
        return len(self._instructions)
    
    @property
    def depth(self) -> int:
        """
        Calculate the circuit depth (approximate, assumes no parallel execution).
        For exact depth, use the optimized circuit.
        """
        return len(self._instructions)
    
    @property
    def instructions(self) -> List[CircuitInstruction]:
        """Return all instructions in order."""
        return self._instructions.copy()
    
    @property
    def name(self) -> str:
        """Return the circuit name."""
        return self._name
    
    @name.setter
    def name(self, value: str):
        """Set the circuit name."""
        self._name = value
    
    def copy(self) -> 'QuantumCircuit':
        """Create a deep copy of this circuit."""
        new_circuit = QuantumCircuit(self._num_qubits, self._name)
        new_circuit._instructions = deepcopy(self._instructions)
        new_circuit._cregs = deepcopy(self._cregs)
        new_circuit._qubit_labels = deepcopy(self._qubit_labels)
        return new_circuit
    
    # ============== Gate Methods ==============
    
    def h(self, qubit: int) -> 'QuantumCircuit':
        """Apply Hadamard gate."""
        return self._add_gate(H(), [qubit])
    
    def x(self, qubit: int) -> 'QuantumCircuit':
        """Apply Pauli-X (NOT) gate."""
        return self._add_gate(X(), [qubit])
    
    def y(self, qubit: int) -> 'QuantumCircuit':
        """Apply Pauli-Y gate."""
        return self._add_gate(Y(), [qubit])
    
    def z(self, qubit: int) -> 'QuantumCircuit':
        """Apply Pauli-Z gate."""
        return self._add_gate(Z(), [qubit])
    
    def s(self, qubit: int) -> 'QuantumCircuit':
        """Apply S (phase) gate."""
        return self._add_gate(S(), [qubit])
    
    def t(self, qubit: int) -> 'QuantumCircuit':
        """Apply T gate."""
        return self._add_gate(T(), [qubit])
    
    def rx(self, qubit: int, theta: float) -> 'QuantumCircuit':
        """Apply RX rotation."""
        return self._add_gate(RX(theta), [qubit])
    
    def ry(self, qubit: int, theta: float) -> 'QuantumCircuit':
        """Apply RY rotation."""
        return self._add_gate(RY(theta), [qubit])
    
    def rz(self, qubit: int, theta: float) -> 'QuantumCircuit':
        """Apply RZ rotation."""
        return self._add_gate(RZ(theta), [qubit])
    
    def cx(self, control: int, target: int) -> 'QuantumCircuit':
        """Apply CNOT (controlled-X) gate."""
        return self._add_gate(CNOT(control, target), [control, target])
    
    def cz(self, control: int, target: int) -> 'QuantumCircuit':
        """Apply controlled-Z gate."""
        return self._add_gate(CZ(control, target), [control, target])
    
    def swap(self, qubit1: int, qubit2: int) -> 'QuantumCircuit':
        """Apply SWAP gate."""
        return self._add_gate(SWAP(qubit1, qubit2), [qubit1, qubit2])
    
    def toffoli(self, control1: int, control2: int, target: int) -> 'QuantumCircuit':
        """Apply Toffoli (CCNOT) gate."""
        return self._add_gate(Toffoli(control1, control2, target), 
                              [control1, control2, target])
    
    def fredkin(self, control: int, target1: int, target2: int) -> 'QuantumCircuit':
        """Apply Fredkin (CSWAP) gate."""
        return self._add_gate(Fredkin(control, target1, target2),
                              [control, target1, target2])
    
    def barrier(self, *qubits: int) -> 'QuantumCircuit':
        """Add a barrier (no-op for simulation, helps visualization)."""
        self._instructions.append(CircuitInstruction(
            gate=IdentityGate(),
            qubits=list(qubits) if qubits else list(range(self._num_qubits)),
            params={'barrier': True}
        ))
        return self
    
    def _add_gate(self, gate: Gate, qubits: List[int]) -> 'QuantumCircuit':
        """Internal method to add a gate."""
        # Validate qubits
        for q in qubits:
            if q < 0 or q >= self._num_qubits:
                raise ValueError(f"Qubit index {q} out of range [0, {self._num_qubits})")
        
        self._instructions.append(CircuitInstruction(gate=gate, qubits=qubits))
        return self
    
    def add_gate(self, gate: Gate, *qubits: int) -> 'QuantumCircuit':
        """
        Add a generic gate to the circuit.
        
        Args:
            gate: The gate to add
            qubits: Qubit indices
            
        Returns:
            self for method chaining
        """
        return self._add_gate(gate, list(qubits))
    
    # ============== Composite Gates ==============
    
    def add_circuit(self, other: 'QuantumCircuit', qubit_map: Optional[Dict[int, int]] = None) -> 'QuantumCircuit':
        """
        Append another circuit to this one.
        
        Args:
            other: Circuit to append
            qubit_map: Optional mapping from other circuit's qubits to this circuit's
        """
        if other._num_qubits > self._num_qubits:
            raise ValueError("Cannot append larger circuit to smaller one")
        
        if qubit_map is None:
            qubit_map = {i: i for i in range(other._num_qubits)}
        
        for instr in other._instructions:
            new_qubits = [qubit_map[q] for q in instr.qubits]
            self._add_gate(instr.gate, new_qubits)
        
        return self
    
    # ============== Gate Decompositions ==============
    
    def decompose_toffoli(self, control1: int, control2: int, target: int) -> 'QuantumCircuit':
        """
        Decompose Toffoli gate into single-qubit and CNOT gates.
        
        Uses the standard decomposition with 6 CNOTs and several single-qubit gates.
        """
        # Standard Toffoli decomposition
        self.h(target)
        self.cx(control2, target)
        self.tdg(target)
        self.cx(control1, target)
        self.t(target)
        self.cx(control2, target)
        self.tdg(target)
        self.cx(control1, target)
        self.t(target)
        self.t(target)
        self.cx(control2, target)
        self.h(target)
        self.cx(control1, control2)
        self.t(control1)
        self.tdg(control2)
        self.cx(control1, control2)
        return self
    
    def decompose_fredkin(self, control: int, target1: int, target2: int) -> 'QuantumCircuit':
        """
        Decompose Fredkin gate into Toffoli gates.
        
        Uses: CSWAP = CCNOT with one target swapped using CNOTs.
        """
        # Standard Fredkin decomposition
        self.cx(target2, target1)
        self.toffoli(control, target1, target2)
        self.cx(target2, target1)
        return self
    
    def decompose_to_basis_gates(self, basis: Tuple[str, ...] = ('h', 't', 'tdg', 'cx')) -> 'QuantumCircuit':
        """
        Decompose circuit to use only basis gates.
        
        Args:
            basis: Tuple of basis gate names
            
        Returns:
            New circuit decomposed to basis gates
        """
        new_circuit = QuantumCircuit(self._num_qubits, f"{self._name}_decomposed")
        
        # Track decomposition rules
        decomposition_rules = {
            'x': [('h', [0]), ('z', [0]), ('h', [0])],
            'y': [('s', [0]), ('h', [0]), ('z', [0]), ('h', [0])],
            's': [('s', [0])],
            't': [('t', [0])],
            'sdg': [('tdg', [0]), ('tdg', [0]), ('tdg', [0])],
            'tdg': [('tdg', [0])],
            'rx': [],  # Would need decomposition
            'ry': [],  # Would need decomposition
            'rz': [],  # Would need decomposition
            'swap': [('cx', [0, 1]), ('cx', [1, 0]), ('cx', [0, 1])],
            'cz': [('h', [1]), ('cx', [0, 1]), ('h', [1])],
            'toffoli': None,  # Use method
            'fredkin': None,  # Use method
        }
        
        for instr in self._instructions:
            gate_name = instr.gate.name.lower()
            qubits = instr.qubits
            
            # Handle special decompositions
            if gate_name == 'toffoli':
                new_circuit.decompose_toffoli(*qubits)
            elif gate_name == 'fredkin':
                new_circuit.decompose_fredkin(*qubits)
            elif gate_name in decomposition_rules:
                rules = decomposition_rules[gate_name]
                if rules:
                    for gate_nm, qs in rules:
                        if hasattr(new_circuit, gate_nm):
                            getattr(new_circuit, gate_nm)(*qs)
                        else:
                            # Generic gate addition
                            pass
                else:
                    # Gate needs special handling
                    new_circuit._add_gate(instr.gate, qubits)
            else:
                new_circuit._add_gate(instr.gate, qubits)
        
        return new_circuit
    
    # ============== Circuit Analysis ==============
    
    def gate_count(self) -> Dict[str, int]:
        """Count occurrences of each gate type."""
        counts = {}
        for instr in self._instructions:
            name = instr.gate.name
            counts[name] = counts.get(name, 0) + 1
        return counts
    
    def qubit_usage(self) -> Dict[int, int]:
        """Count how many gates act on each qubit."""
        usage = {i: 0 for i in range(self._num_qubits)}
        for instr in self._instructions:
            for q in instr.qubits:
                usage[q] += 1
        return usage
    
    def get_qubits_acted_on(self) -> set:
        """Get set of all qubits that have gates applied."""
        qubits = set()
        for instr in self._instructions:
            qubits.update(instr.qubits)
        return qubits
    
    # ============== Matrix Representation ==============
    
    def to_matrix(self) -> np.ndarray:
        """
        Compute the unitary matrix for the entire circuit.
        
        Note: This grows as 2^n × 2^n, so it's only practical for small circuits.
        """
        dim = 2 ** self._num_qubits
        unitary = np.eye(dim, dtype=complex)
        
        for instr in self._instructions:
            gate_matrix = instr.gate.matrix
            qubits = instr.qubits
            
            # Build full unitary for this gate
            full_gate = self._expand_gate(gate_matrix, qubits)
            unitary = full_gate @ unitary
        
        return unitary
    
    def _expand_gate(self, gate_matrix: np.ndarray, qubits: List[int]) -> np.ndarray:
        """Expand a gate to full circuit unitary."""
        dim = 2 ** self._num_qubits
        result = np.eye(dim, dtype=complex)
        
        for q in qubits:
            expanded = np.eye(dim, dtype=complex)
            for i in range(dim):
                for j in range(dim):
                    # Check if this element is affected by the gate on qubit q
                    i_q = (i >> q) & 1
                    j_q = (j >> q) & 1
                    i_mask = (dim - 1) ^ (1 << q)
                    i_rest = i & i_mask
                    j_rest = j & i_mask
                    
                    if i_rest == j_rest:
                        i_other = (i >> (q + 1)) | ((i & ((1 << q) - 1)))
                        j_other = (j >> (q + 1)) | ((j & ((1 << q) - 1)))
                        if i_other == j_other:
                            new_i = i_rest | (i_q << q)
                            new_j = j_rest | (j_q << q)
                            expanded[new_i, new_j] = gate_matrix[i_q, j_q]
            
            result = expanded @ result
        
        return result
    
    # ============== Drawing ==============
    
    def draw(self, output: str = 'text', scale: float = 1.0) -> str:
        """
        Generate a text representation of the circuit.
        
        Args:
            output: Output format ('text' or 'latex')
            scale: Scale factor for visualization
            
        Returns:
            String representation of the circuit
        """
        if output == 'text':
            return self._draw_text()
        elif output == 'latex':
            return self._draw_latex()
        else:
            raise ValueError(f"Unknown output format: {output}")
    
    def _draw_text(self) -> str:
        """Generate ASCII art circuit diagram."""
        lines = []
        
        # Header
        lines.append(f"Circuit: {self._name} ({self._num_qubits} qubits, depth={self.depth})")
        lines.append("─" * 60)
        
        # Qubit lines
        for q in range(self._num_qubits):
            lines.append(f"q{q}: ─")
        
        lines.append("─" * 60)
        
        # Build gate representation per time step
        # Simple linear representation
        gate_strs = []
        current_qubits = {q: [] for q in range(self._num_qubits)}
        
        for instr in self._instructions:
            qubits = instr.qubits
            name = instr.gate.name
            
            if len(qubits) == 1:
                current_qubits[qubits[0]].append(name)
            elif len(qubits) == 2:
                current_qubits[qubits[0]].append(f"●──{name}──●")
                current_qubits[qubits[1]].append(f"•")
            else:
                # Multi-qubit gate
                for i, q in enumerate(qubits):
                    if i == 0:
                        current_qubits[q].append(f"●──{name}──")
                    elif i == len(qubits) - 1:
                        current_qubits[q].append(f"•")
                    else:
                        current_qubits[q].append("•")
        
        # Simple circuit diagram
        lines = [f"q[{q}] │" for q in range(self._num_qubits)]
        
        # Determine columns needed
        gate_symbols = {
            'H': '─[H]─',
            'X': '─[X]─',
            'Y': '─[Y]─',
            'Z': '─[Z]─',
            'S': '─[S]─',
            'T': '─[T]─',
            'CNOT': '─●──',
            'CZ': '─@──',
            'SWAP': '─×──',
            'Toffoli': '─▣──',
            'I': '────',
        }
        
        for instr in self._instructions:
            name = instr.gate.name
            qubits = sorted(instr.qubits)
            
            symbol = gate_symbols.get(name, f'─[{name[:2]}]─')
            
            for q in range(self._num_qubits):
                if q in qubits:
                    lines[q] += symbol
                else:
                    lines[q] += '────'
        
        return "\n".join(lines)
    
    def _draw_latex(self) -> str:
        """Generate LaTeX code for circuit visualization using Qcircuit."""
        lines = [
            r"\begin{Qcircuit}",
            rf"@C={1.0}em @R={0.7}em",
        ]
        
        # Gate commands
        for instr in self._instructions:
            qubits = sorted(instr.qubits)
            name = instr.gate.name
            
            # Map gate names to LaTeX commands
            latex_gates = {
                'H': r'\gate{H}',
                'X': r'\gate{X}',
                'Y': r'\gate{Y}',
                'Z': r'\gate{Z}',
                'S': r'\gate{S}',
                'T': r'\gate{T}',
                'CNOT': r'\control\qw',
                'CZ': r'\control\qw',
                'SWAP': r'\qswap\qw',
                'I': r'\qw',
            }
            
            cmd = latex_gates.get(name, f'\\gate{{{name}}}')
            
            for q in range(self._num_qubits):
                if q == qubits[0]:
                    lines.append(f"\\lstick{{q_{q}}} & {cmd} & \\qw")
                elif q in qubits[1:]:
                    lines.append(f" & \\targ{{}} & \\qw")
                else:
                    lines.append(f" & \\qw & \\qw")
        
        lines.append(r"\end{Qcircuit}")
        return "\n".join(lines)
    
    # ============== Serialization ==============
    
    def to_dict(self) -> Dict:
        """Convert circuit to dictionary for serialization."""
        return {
            'name': self._name,
            'num_qubits': self._num_qubits,
            'instructions': [
                {
                    'gate': instr.gate.name,
                    'qubits': instr.qubits,
                    'params': instr.params
                }
                for instr in self._instructions
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QuantumCircuit':
        """Create circuit from dictionary."""
        circuit = cls(data['num_qubits'], data.get('name', 'circuit'))
        
        # Gate mapping
        from . import gates as g
        
        for instr_data in data.get('instructions', []):
            gate_name = instr_data['gate']
            qubits = instr_data['qubits']
            
            # Find and add the gate
            if hasattr(g, gate_name):
                gate_func = getattr(g, gate_name)
                if callable(gate_func):
                    try:
                        circuit.add_gate(gate_func(*qubits))
                    except TypeError:
                        circuit.add_gate(gate_func())
            else:
                # Generic gate
                pass
        
        return circuit
    
    def __repr__(self):
        return f"QuantumCircuit({self._num_qubits} qubits, {len(self._instructions)} gates)"
    
    def __str__(self):
        """Human-readable circuit description."""
        lines = [f"QuantumCircuit: {self._name}"]
        lines.append(f"  Qubits: {self._num_qubits}")
        lines.append(f"  Depth: {self.depth}")
        lines.append(f"  Gates: {len(self._instructions)}")
        lines.append("")
        lines.append("Instructions:")
        
        for i, instr in enumerate(self._instructions):
            q_str = ",".join(map(str, instr.qubits))
            lines.append(f"  {i:3d}: {instr.gate.name:10s} on [{q_str}]")
        
        return "\n".join(lines)
