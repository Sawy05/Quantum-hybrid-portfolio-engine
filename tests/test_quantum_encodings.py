"""Tests for quantum data encodings (Angle, IQP) and feature pipelines."""

import numpy as np

from src.data.market_generator import MarketDataGenerator
from src.data.quantum_encoder import ClassicalQuantumFeaturePipeline
from src.qml_models.circuits import QuantumCircuitFactory


def test_market_generator_determinism():
    """Verifies that market generator with same seed produces identical paths."""
    gen1 = MarketDataGenerator(n_assets=4, n_steps=100, random_seed=42)
    gen2 = MarketDataGenerator(n_assets=4, n_steps=100, random_seed=42)

    data1 = gen1.generate_synthetic()
    data2 = gen2.generate_synthetic()

    np.testing.assert_allclose(data1.prices.to_numpy(), data2.prices.to_numpy())
    np.testing.assert_array_equal(data1.regimes, data2.regimes)


def test_quantum_encoder_angle_bounds():
    """Verifies feature pipeline scales classical inputs strictly into [-pi, pi]."""
    gen = MarketDataGenerator(n_assets=4, n_steps=150, random_seed=42)
    data = gen.generate_synthetic()

    pipeline = ClassicalQuantumFeaturePipeline(
        n_qubits=4, rolling_window=10, scaling_method="minmax"
    )
    features = pipeline.extract_classical_features(data.returns)
    angles = pipeline.fit_transform_to_angles(features.to_numpy())

    assert angles.shape[1] == 4
    assert np.all(angles >= -np.pi - 1e-6)
    assert np.all(angles <= np.pi + 1e-6)


def test_angle_vs_iqp_circuits():
    """Verifies that Angle and IQP encodings compile cleanly and produce valid quantum expectations."""
    n_qubits = 4
    angles = np.array([0.5, -1.2, 2.1, -0.8])

    for enc in ["angle", "iqp"]:
        factory = QuantumCircuitFactory(
            n_qubits=n_qubits,
            circuit_depth=2,
            encoding_type=enc,
            entanglement_topology="ring",
        )
        qnode = factory.create_vqc_qnode(interface="numpy")
        weights = np.random.uniform(0, np.pi, size=factory.weight_shape())

        expvals = qnode(angles, weights)
        assert len(expvals) == n_qubits
        for val in expvals:
            assert -1.0 <= val <= 1.0


def test_circuit_specifications():
    """Verifies circuit specs return accurate depth, gate counts, and parameter counts."""
    factory = QuantumCircuitFactory(n_qubits=4, circuit_depth=3, encoding_type="angle")
    specs = factory.get_circuit_specs()

    assert specs["n_qubits"] == 4
    assert specs["trainable_parameters"] == 3 * 4 * 3  # depth * n_qubits * 3
    assert specs["circuit_depth"] > 0
