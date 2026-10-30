"""Tests for Zero-Noise Extrapolation (ZNE), unitary folding, and NISQ error mitigation."""

import numpy as np
import pennylane as qml

from src.noise_mitigation.noise_models import NISQNoiseModel
from src.noise_mitigation.zne_mitigator import ZeroNoiseExtrapolator


def test_unitary_folding_noiseless_identity():
    r"""Verifies that in noiseless simulation, U(U^\dagger U)^n produces identical expectation value to U."""
    dev = qml.device("default.qubit", wires=2)

    def ansatz():
        qml.RX(0.8, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(1.2, wires=1)

    @qml.qnode(dev)
    def original_circuit():
        ansatz()
        return qml.expval(qml.PauliZ(1))

    ref_val = float(original_circuit())

    for scale in [1, 3, 5]:

        @qml.qnode(dev)
        def folded_circuit():
            ZeroNoiseExtrapolator.fold_circuit_unitary(ansatz, scale_factor=scale)
            return qml.expval(qml.PauliZ(1))

        folded_val = float(folded_circuit())
        assert abs(folded_val - ref_val) < 1e-6, f"Scale {scale} diverged from noiseless identity."


def test_richardson_extrapolation_coefficients():
    """Verifies mathematical properties of Richardson extrapolation weights: sum(weights) == 1."""
    scales = [1, 3, 5]
    weights = ZeroNoiseExtrapolator.richardson_coefficients(scales)

    # In zero-noise extrapolation, sum of Richardson weights is mathematically 1
    assert abs(np.sum(weights) - 1.0) < 1e-6

    # Test exact recovery of linear decay: f(lambda) = a0 + a1 * lambda
    a0 = 0.75
    a1 = -0.04
    noisy_vals = [a0 + a1 * s for s in scales]

    zne = ZeroNoiseExtrapolator(scale_factors=scales, extrapolation_method="richardson")
    mitigated, _ = zne.extrapolate(scales, noisy_vals)

    assert abs(mitigated - a0) < 1e-6


def test_zne_noise_reduction():
    """Verifies that ZNE reduces estimation error on a simulated depolarizing channel."""
    noise_model = NISQNoiseModel(
        n_qubits=1, depolarizing_prob=0.03, phase_damping_prob=0.0, readout_error_prob=0.0
    )

    def simple_unitary():
        qml.RX(np.pi, wires=0)  # Rotates |0> to |1>, so <Z> = -1.0

    def obs():
        return qml.expval(qml.PauliZ(0))

    zne = ZeroNoiseExtrapolator(scale_factors=[1, 3, 5], extrapolation_method="polynomial")
    res = zne.run_mitigated_evaluation(
        circuit_fn=simple_unitary,
        observable_fn=obs,
        noise_model=noise_model,
        wires=[0],
    )

    true_val = -1.0
    unmit_err = abs(res.unmitigated_value - true_val)
    mit_err = abs(res.mitigated_value - true_val)

    # Mitigated error should be less than or equal to unmitigated error
    assert mit_err < unmit_err + 1e-3
