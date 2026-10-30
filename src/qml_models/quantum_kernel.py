"""Quantum Kernel Estimators (Fidelity & Projected Quantum Kernel) and SVM Classifiers."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np
import pennylane as qml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


class QuantumKernelEstimator:
    """Computes Quantum State Overlaps |<psi(x)|psi(x')>|^2 and Projected Quantum Kernels."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        kernel_type: Literal["fidelity", "projected"] = "fidelity",
        gamma: float = 1.0,
        device_name: str = "default.qubit",
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.dev = qml.device(device_name, wires=n_qubits)

        self._build_circuits()

    def _feature_map(self, x: np.ndarray) -> None:
        """Applies layered Angle + Entanglement Feature Map U(x)."""
        wires = list(range(self.n_qubits))
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
                qml.RZ(x[i] * 0.5, wires=i)
            # Entanglement
            for i in range(self.n_qubits):
                qml.CNOT(wires=[wires[i], wires[(i + 1) % self.n_qubits]])

    def _build_circuits(self) -> None:
        """Compiles PennyLane QNodes for kernel evaluation."""

        # 1. Fidelity Kernel QNode: |<psi(x)|psi(x')>|^2 = P(|00...0>) under U(x) U^\dagger(x')
        @qml.qnode(self.dev)
        def fidelity_circuit(x1, x2):
            self._feature_map(x1)
            qml.adjoint(self._feature_map)(x2)
            return qml.probs(wires=range(self.n_qubits))

        self._fidelity_qnode = fidelity_circuit

        # 2. Projected Quantum Kernel QNode: computes Pauli <X>, <Y>, <Z> for single qubits
        @qml.qnode(self.dev)
        def projected_qnode(x):
            self._feature_map(x)
            obs = []
            for w in range(self.n_qubits):
                obs.append(qml.expval(qml.PauliX(w)))
                obs.append(qml.expval(qml.PauliY(w)))
                obs.append(qml.expval(qml.PauliZ(w)))
            return obs

        self._projected_qnode = projected_qnode

    def evaluate_pair(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Computes kernel similarity K(x1, x2) in [0, 1]."""
        if self.kernel_type == "fidelity":
            probs = self._fidelity_qnode(x1, x2)
            return float(probs[0])  # Probability of |0...0>
        else:  # Projected Quantum Kernel
            obs1 = np.array(self._projected_qnode(x1))
            obs2 = np.array(self._projected_qnode(x2))
            dist_sq = float(np.sum((obs1 - obs2) ** 2))
            return float(np.exp(-self.gamma * dist_sq))

    def compute_gram_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculates the Gram matrix K_{ij} = K(x_i, x_j)."""
        is_symmetric = X2 is None
        X2_eval = X1 if is_symmetric else X2
        assert X2_eval is not None

        n1 = len(X1)
        n2 = len(X2_eval)
        gram = np.zeros((n1, n2), dtype=np.float64)

        if is_symmetric:
            for i in range(n1):
                gram[i, i] = 1.0  # Unit fidelity for identical states
                for j in range(i + 1, n1):
                    val = self.evaluate_pair(X1[i], X1[j])
                    gram[i, j] = val
                    gram[j, i] = val
        else:
            for i in range(n1):
                for j in range(n2):
                    gram[i, j] = self.evaluate_pair(X1[i], X2_eval[j])

        return gram


class QuantumKernelClassifier:
    """Scikit-Learn compatible Support Vector Classifier powered by Quantum Kernels."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        kernel_type: Literal["fidelity", "projected"] = "fidelity",
        c_param: float = 1.0,
        gamma: float = 1.0,
    ):
        self.n_qubits = n_qubits
        self.kernel_estimator = QuantumKernelEstimator(
            n_qubits=n_qubits,
            n_layers=n_layers,
            kernel_type=kernel_type,
            gamma=gamma,
        )
        self.c_param = c_param
        self.svm = SVC(kernel="precomputed", C=c_param)
        self.X_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> QuantumKernelClassifier:
        """Fits the SVM on the Quantum Gram Matrix."""
        self.X_train = np.copy(X)
        K_train = self.kernel_estimator.compute_gram_matrix(X)
        self.svm.fit(K_train, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for query samples."""
        if self.X_train is None:
            raise ValueError("Classifier must be fitted before predict.")
        K_test = self.kernel_estimator.compute_gram_matrix(X, self.X_train)
        return self.svm.predict(K_test)

    def evaluate_benchmark(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Compares Quantum Kernel SVM against Classical RBF SVM."""
        # 1. Fit Quantum Kernel SVM
        self.fit(X_train, y_train)
        q_preds = self.predict(X_test)
        q_acc = float(accuracy_score(y_test, q_preds))
        q_f1 = float(f1_score(y_test, q_preds, average="macro"))

        # 2. Classical RBF SVM Baseline
        classical_svm = SVC(kernel="rbf", C=self.c_param)
        classical_svm.fit(X_train, y_train)
        c_preds = classical_svm.predict(X_test)
        c_acc = float(accuracy_score(y_test, c_preds))
        c_f1 = float(f1_score(y_test, c_preds, average="macro"))

        return {
            "quantum_kernel": {
                "accuracy": q_acc,
                "macro_f1": q_f1,
                "support_vectors_count": int(len(self.svm.support_)),
            },
            "classical_rbf": {
                "accuracy": c_acc,
                "macro_f1": c_f1,
                "support_vectors_count": int(len(classical_svm.support_)),
            },
        }
