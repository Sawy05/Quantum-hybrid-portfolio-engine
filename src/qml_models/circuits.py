"""Parametric Quantum Circuits, Encodings (Angle, IQP), and Variational Ansätze."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pennylane as qml


class QuantumCircuitFactory:
    """Factory for building quantum feature encodings and variational ansätze."""

    def __init__(
        self,
        n_qubits: int = 4,
        circuit_depth: int = 3,
        encoding_type: Literal["angle", "iqp"] = "angle",
        entanglement_topology: Literal["ring", "all-to-all"] = "ring",
        diff_method: str = "parameter-shift",
        device_name: str = "default.qubit",
    ):
        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        self.encoding_type = encoding_type
        self.entanglement_topology = entanglement_topology
        self.diff_method = diff_method
        self.device_name = device_name
        self.dev = qml.device(device_name, wires=n_qubits)

    @staticmethod
    def angle_embedding(features: Any, wires: List[int], rotation: str = "Y") -> None:
        """Applies single-qubit rotation angle embedding: R_x, R_y, or R_z."""
        qml.AngleEmbedding(features=features, wires=wires, rotation=rotation)

    @staticmethod
    def iqp_embedding(features: Any, wires: List[int], reps: int = 1) -> None:
        """Instantaneous Quantum Polynomial (IQP) embedding: H -> Rz(x_i) -> CNOT-Rz(x_i*x_j)-CNOT."""
        n_w = len(wires)
        for _ in range(reps):
            for w in wires:
                qml.Hadamard(wires=w)
            for i, w in enumerate(wires):
                qml.RZ(features[i], wires=w)
            for i in range(n_w):
                for j in range(i + 1, n_w):
                    qml.CNOT(wires=[wires[i], wires[j]])
                    qml.RZ(features[i] * features[j], wires=wires[j])
                    qml.CNOT(wires=[wires[i], wires[j]])

    def variational_ansatz(self, weights: Any, wires: List[int]) -> None:
        """Strongly entangling layers with configurable ring or all-to-all CNOT topology."""
        n_w = len(wires)
        # weights shape: (depth, n_qubits, 3)
        depth = weights.shape[0]

        for layer in range(depth):
            # 1. Arbitrary single-qubit rotations: RZ -> RY -> RZ or Rot(phi, theta, omega)
            for i, w in enumerate(wires):
                qml.Rot(
                    weights[layer, i, 0],
                    weights[layer, i, 1],
                    weights[layer, i, 2],
                    wires=w,
                )

            # 2. Entanglement Topology
            if self.entanglement_topology == "all-to-all":
                for i in range(n_w):
                    for j in range(i + 1, n_w):
                        qml.CNOT(wires=[wires[i], wires[j]])
            else:  # ring topology
                for i in range(n_w):
                    target = (i + 1) % n_w
                    qml.CNOT(wires=[wires[i], wires[target]])

    def weight_shape(self) -> Tuple[int, int, int]:
        """Returns the tensor shape of trainable variational weights."""
        return (self.circuit_depth, self.n_qubits, 3)

    def create_vqc_qnode(
        self,
        wires: Optional[List[int]] = None,
        interface: str = "torch",
    ):
        """Constructs and returns an autograd/torch-differentiable QNode."""
        target_wires = wires if wires is not None else list(range(self.n_qubits))

        @qml.qnode(self.dev, interface=interface, diff_method=self.diff_method)
        def vqc_circuit(inputs, weights):
            if self.encoding_type == "iqp":
                self.iqp_embedding(inputs, target_wires)
            else:
                self.angle_embedding(inputs, target_wires, rotation="Y")

            self.variational_ansatz(weights, target_wires)
            return [qml.expval(qml.PauliZ(w)) for w in target_wires]

        return vqc_circuit

    def get_circuit_specs(self) -> Dict[str, Any]:
        """Calculates gate counts, depth, and parameter footprint for diagnostics."""
        dummy_inputs = np.zeros(self.n_qubits)
        dummy_weights = np.zeros(self.weight_shape())

        @qml.qnode(self.dev)
        def dummy_circuit():
            if self.encoding_type == "iqp":
                self.iqp_embedding(dummy_inputs, list(range(self.n_qubits)))
            else:
                self.angle_embedding(dummy_inputs, list(range(self.n_qubits)), rotation="Y")
            self.variational_ansatz(dummy_weights, list(range(self.n_qubits)))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        specs = qml.specs(dummy_circuit)()
        specs_dict = specs.to_dict() if hasattr(specs, "to_dict") else {}
        resources = specs_dict.get("resources", {})

        gate_counts = {}
        depth = self.circuit_depth * 2
        if isinstance(resources, dict):
            gate_counts = {str(k): int(v) for k, v in resources.get("gate_types", {}).items()}
            depth = int(resources.get("depth", depth))

        param_count = int(np.prod(self.weight_shape()))

        return {
            "n_qubits": self.n_qubits,
            "circuit_depth": depth,
            "encoding_type": self.encoding_type,
            "entanglement_topology": self.entanglement_topology,
            "trainable_parameters": param_count,
            "weight_shape": list(self.weight_shape()),
            "diff_method": self.diff_method,
            "gate_types": gate_counts,
        }
