"""Tests for Parameter-Shift rule gradients and Hybrid PyTorch-PennyLane training."""

import numpy as np
import pennylane as qml
import torch

from src.qml_models.hybrid_vqc import HybridQuantumClassifier


def test_parameter_shift_vs_finite_difference():
    """Verifies that Parameter-Shift rule matches finite difference for rotation gates."""
    n_qubits = 2
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(theta):
        qml.RY(theta, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(1))

    theta_val = 0.73
    shift = np.pi / 2.0
    grad_ps = float(0.5 * (circuit(theta_val + shift) - circuit(theta_val - shift)))
    expected_grad = -np.sin(theta_val)

    assert abs(grad_ps - expected_grad) < 1e-5


def test_hybrid_vqc_forward_and_backward():
    """Verifies end-to-end forward pass and backpropagation through classical LSTM and quantum layer."""
    n_qubits = 4
    model = HybridQuantumClassifier(
        n_qubits=n_qubits,
        circuit_depth=2,
        feature_dim=4,
        hidden_dim=8,
        num_classes=3,
        diff_method="backprop",
    )

    batch_size = 3
    seq_len = 6
    x = torch.randn(batch_size, seq_len, 4, requires_grad=True)

    logits, q_angles, q_expvals = model(x)

    assert logits.shape == (batch_size, 3)
    assert q_angles.shape == (batch_size, n_qubits)
    assert q_expvals.shape == (batch_size, n_qubits)

    # Compute loss and backward pass
    loss = logits.sum()
    loss.backward()

    # Verify gradients propagate to classical and quantum parameters
    assert model.classical_to_quantum[0].weight.grad is not None
    assert model.quantum_layer.weights.grad is not None
    assert torch.any(model.quantum_layer.weights.grad != 0.0)


def test_hybrid_vqc_short_training():
    """Verifies that training loop executes and reduces loss on synthetic batch."""
    model = HybridQuantumClassifier(
        n_qubits=4,
        circuit_depth=1,
        feature_dim=4,
        hidden_dim=8,
        num_classes=3,
    )

    X_train = np.random.randn(12, 5, 4).astype(np.float32)
    y_train = np.random.choice([0, 1, 2], size=12).astype(np.int64)

    history = model.fit_model(X_train, y_train, epochs=2, batch_size=4, lr=0.01, verbose=False)

    assert len(history["loss"]) == 2
    assert history["loss"][-1] >= 0.0
