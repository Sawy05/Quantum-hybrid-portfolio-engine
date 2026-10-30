"""Barren Plateau diagnostic tool tracking gradient variance decay across qubit scales and depths."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pennylane as qml


class BarrenPlateauAnalyzer:
    """Diagnoses barren plateaus by evaluating the parameter-shift gradient variance Var(d<H>/d theta).

    References:
        McClean et al. (2018): Barren plateaus in quantum neural network training landscapes.
        Skolik et al. (2021): Layerwise learning for quantum neural networks.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    @staticmethod
    def _parameter_shift_gradient(
        circuit_fn,
        weights: np.ndarray,
        target_idx: Tuple[int, int, int],
    ) -> float:
        """Evaluates exact analytical gradient via Parameter-Shift Rule:

        d<H>/d theta = 0.5 * [<H>(theta + pi/2) - <H>(theta - pi/2)]
        """
        shift = np.pi / 2.0

        weights_plus = weights.copy()
        weights_plus[target_idx] += shift

        weights_minus = weights.copy()
        weights_minus[target_idx] -= shift

        val_plus = float(circuit_fn(weights_plus))
        val_minus = float(circuit_fn(weights_minus))

        return 0.5 * (val_plus - val_minus)

    def measure_gradient_variance(
        self,
        n_qubits: int,
        circuit_depth: int,
        n_samples: int = 40,
        strategy: str = "random",  # "random" or "identity_block"
    ) -> float:
        """Computes Var(d<H>/d theta_0) over random or pre-trained parameter initializations."""
        dev = qml.device("default.qubit", wires=n_qubits)
        rng = np.random.default_rng(self.seed + n_qubits * 100 + circuit_depth)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(w):
            depth = w.shape[0]
            for layer in range(depth):
                for i in range(n_qubits):
                    qml.Rot(w[layer, i, 0], w[layer, i, 1], w[layer, i, 2], wires=i)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            # Observable: local Pauli-Z on first two qubits
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        gradients: List[float] = []
        target_param = (0, 0, 1)  # Depth 0, Qubit 0, Rotation Y

        for _ in range(n_samples):
            if strategy == "identity_block":
                # Layer-wise pre-training initialization: initial small perturbation near identity
                w = rng.normal(loc=0.0, scale=0.08, size=(circuit_depth, n_qubits, 3))
            else:  # Uniform random Haar-like initialization in [0, 2*pi]
                w = rng.uniform(0.0, 2.0 * np.pi, size=(circuit_depth, n_qubits, 3))

            grad = self._parameter_shift_gradient(circuit, w, target_param)
            gradients.append(grad)

        var_grad = float(np.var(gradients))
        return var_grad

    def scan_qubit_scaling(
        self,
        qubit_range: Optional[List[int]] = None,
        circuit_depth: int = 3,
        n_samples: int = 30,
    ) -> Dict[str, Any]:
        """Scans gradient variance over increasing qubit counts for both random and layerwise initialization."""
        qubits = qubit_range or [4, 6, 8, 10]
        random_vars: List[float] = []
        identity_vars: List[float] = []

        for n in qubits:
            v_rand = self.measure_gradient_variance(
                n_qubits=n,
                circuit_depth=circuit_depth,
                n_samples=n_samples,
                strategy="random",
            )
            v_ident = self.measure_gradient_variance(
                n_qubits=n,
                circuit_depth=circuit_depth,
                n_samples=n_samples,
                strategy="identity_block",
            )
            random_vars.append(v_rand)
            identity_vars.append(v_ident)

        return {
            "qubit_counts": qubits,
            "circuit_depth": circuit_depth,
            "random_init_variances": random_vars,
            "identity_init_variances": identity_vars,
        }
