"""Tests for QUBO formulation, Ising Hamiltonian transformation, and QAOA optimization."""

import numpy as np

from src.optimization.qaoa_solver import QAOAPortfolioSolver
from src.optimization.qubo_ising import (
    PortfolioQUBO,
    evaluate_ising_energy,
    ising_to_qubo,
    qubo_to_ising,
)


def test_qubo_to_ising_energy_equivalence():
    """Mathematically proves that E_QUBO(x) == E_Ising(s) for all 2^N bitstrings."""
    n_assets = 4
    mu = np.array([0.12, 0.15, 0.08, 0.10])
    cov = np.array(
        [
            [0.04, 0.01, 0.005, 0.01],
            [0.01, 0.05, 0.015, 0.02],
            [0.005, 0.015, 0.03, 0.008],
            [0.01, 0.02, 0.008, 0.06],
        ]
    )

    qubo = PortfolioQUBO(
        expected_returns=mu,
        covariance_matrix=cov,
        risk_aversion=0.5,
        penalty_coeff=8.0,
        cardinality_k=2,
    )
    ising = qubo.to_ising()

    # Test exhaustive set of all 2^4 = 16 bitstrings
    for i in range(2**n_assets):
        bitstring = [int(b) for b in format(i, f"0{n_assets}b")]
        # Mapping: x_i in {0, 1} -> s_i = 1 - 2*x_i in {+1, -1}
        spins = [1 - 2 * x for x in bitstring]

        qubo_energy = qubo.evaluate_bitstring(bitstring)
        ising_energy = evaluate_ising_energy(spins, ising)

        assert abs(qubo_energy - ising_energy) < 1e-7, (
            f"Energy mismatch for bitstring {bitstring}: QUBO={qubo_energy}, Ising={ising_energy}"
        )


def test_qubo_ising_roundtrip_invertibility():
    """Verifies that ising_to_qubo exactly inverts qubo_to_ising."""
    n = 3
    rng = np.random.default_rng(123)
    # Symmetric QUBO matrix
    A = rng.normal(size=(n, n))
    Q_orig = (A + A.T) / 2.0
    c_orig = 5.25

    ising = qubo_to_ising(Q_orig, c_orig)
    Q_rec, c_rec = ising_to_qubo(ising)

    np.testing.assert_allclose(Q_orig, Q_rec, atol=1e-7)
    assert abs(c_orig - c_rec) < 1e-7

    # Check that bitstring evaluations match exactly
    for i in range(2**n):
        x = np.array([int(b) for b in format(i, f"0{n}b")])
        e_orig = x.T @ Q_orig @ x + c_orig
        e_rec = x.T @ Q_rec @ x + c_rec
        assert abs(e_orig - e_rec) < 1e-7


def test_qaoa_solver_execution_and_approximation_ratio():
    """Verifies QAOA solver convergence and approximation ratio bounds."""
    n_assets = 3
    mu = np.array([0.15, 0.10, 0.08])
    cov = np.eye(3) * 0.04

    qubo = PortfolioQUBO(
        expected_returns=mu,
        covariance_matrix=cov,
        risk_aversion=0.5,
        penalty_coeff=5.0,
        cardinality_k=2,
    )
    ising = qubo.to_ising()

    solver = QAOAPortfolioSolver(
        ising=ising,
        p_steps=1,
        optimizer="COBYLA",
        max_iterations=30,
    )

    res = solver.solve(expected_returns=mu, covariance_matrix=cov)

    assert len(res.best_bitstring) == n_assets
    assert 0.0 <= res.approximation_ratio <= 1.0
    assert 0.0 <= res.probability_of_success <= 1.0
    assert len(res.weights) == n_assets
    assert abs(np.sum(res.weights) - 1.0) < 1e-6
