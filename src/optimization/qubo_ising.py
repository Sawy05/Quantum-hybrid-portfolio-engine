"""Markowitz Portfolio QUBO formulation and exact Ising Hamiltonian mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class IsingHamiltonian:
    """Represents an Ising Cost Hamiltonian: H = sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j + offset."""

    h: np.ndarray  # Shape (N,)
    J: np.ndarray  # Shape (N, N), upper triangular with zero diagonal
    offset: float
    n_qubits: int


class PortfolioQUBO:
    """Constructs Quadratic Unconstrained Binary Optimization (QUBO) for Markowitz Portfolio Selection.

    Objective:
        min_x  (lambda / 2) * x^T Sigma x - mu^T x + A * (sum_i x_i - K)^2
    where x_i in {0, 1}, lambda is risk aversion, K is cardinality target, A is constraint penalty.
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 0.5,
        penalty_coeff: float = 10.0,
        cardinality_k: int = 2,
    ):
        self.mu = np.asarray(expected_returns, dtype=np.float64)
        self.sigma = np.asarray(covariance_matrix, dtype=np.float64)
        self.risk_aversion = float(risk_aversion)
        self.penalty = float(penalty_coeff)
        self.k = int(cardinality_k)
        self.n_assets = len(self.mu)

        if self.sigma.shape != (self.n_assets, self.n_assets):
            raise ValueError(
                f"Covariance matrix shape {self.sigma.shape} does not match assets {self.n_assets}"
            )

        self.Q, self.constant = self._build_qubo_matrix()

    def _build_qubo_matrix(self) -> Tuple[np.ndarray, float]:
        """Expands the portfolio objective into Q matrix such that f(x) = x^T Q x + constant."""
        n = self.n_assets
        Q = np.zeros((n, n), dtype=np.float64)

        # 1. Quadratic risk component: (lambda / 2) * x^T Sigma x
        risk_term = 0.5 * self.risk_aversion * self.sigma

        # 2. Linear expected return component: - mu^T x (placed on diagonal since x_i^2 = x_i)
        # 3. Cardinality penalty: A * (sum_i x_i - K)^2
        #    = A * ( sum_i x_i^2 + 2 sum_{i<j} x_i x_j - 2K sum_i x_i + K^2 )
        #    = A * ( (1 - 2K) sum_i x_i + 2 sum_{i<j} x_i x_j ) + A * K^2
        constant = self.penalty * (self.k**2)

        for i in range(n):
            for j in range(n):
                if i == j:
                    Q[i, i] = risk_term[i, i] - self.mu[i] + self.penalty * (1.0 - 2.0 * self.k)
                else:
                    # Off-diagonal gets risk term + 2 * A (split equally or placed on upper triangular)
                    Q[i, j] = risk_term[i, j] + self.penalty

        return Q, constant

    def evaluate_bitstring(self, bitstring: np.ndarray | List[int]) -> float:
        """Evaluates QUBO objective value for a given binary selection vector x in {0, 1}^N."""
        x = np.asarray(bitstring, dtype=np.float64)
        return float(x.T @ self.Q @ x + self.constant)

    def to_ising(self) -> IsingHamiltonian:
        """Maps this QUBO problem to an Ising Hamiltonian H_C."""
        return qubo_to_ising(self.Q, self.constant)


def qubo_to_ising(Q: np.ndarray, constant: float = 0.0) -> IsingHamiltonian:
    """Exact algebraic transformation from binary QUBO variables x_i in {0, 1}

    to Pauli-Z spin operators Z_i in {-1, +1} via x_i = (1 - Z_i) / 2:

    H_C = sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j + offset * I
    """
    n = Q.shape[0]
    # Symmetrize off-diagonal terms W_ij = Q_ij + Q_ji
    W = Q + Q.T
    for i in range(n):
        W[i, i] = Q[i, i]

    h = np.zeros(n, dtype=np.float64)
    J = np.zeros((n, n), dtype=np.float64)

    # Couplings J_{ij} for i < j
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = (Q[i, j] + Q[j, i]) / 4.0

    # Linear fields h_i
    for i in range(n):
        sum_off_diag = np.sum(Q[i, :]) + np.sum(Q[:, i]) - 2.0 * Q[i, i]
        h[i] = -0.5 * Q[i, i] - 0.25 * sum_off_diag

    # Constant scalar offset
    # offset = constant + 0.5 * sum_i Q_ii + 0.25 * sum_{i<j} (Q_ij + Q_ji)
    sum_qii = np.sum(np.diag(Q))
    sum_qij = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sum_qij += Q[i, j] + Q[j, i]
    offset = float(constant + 0.5 * sum_qii + 0.25 * sum_qij)

    return IsingHamiltonian(h=h, J=J, offset=offset, n_qubits=n)


def ising_to_qubo(ising: IsingHamiltonian) -> Tuple[np.ndarray, float]:
    """Inverse transformation: recovers QUBO Q matrix and constant from Ising parameters."""
    n = ising.n_qubits
    Q = np.zeros((n, n), dtype=np.float64)

    # J_{ij} = (Q_{ij} + Q_{ji}) / 4 => Q_{ij} = 2 * J_{ij} (symmetric)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = 2.0 * ising.J[i, j]
            Q[j, i] = 2.0 * ising.J[i, j]

    # h_i = -0.5 * Q_ii - 0.25 * sum_{j!=i} (Q_ij + Q_ji)
    # Since Q_{ij} + Q_{ji} = 4 * J_{ij}, we have:
    # h_i = -0.5 * Q_ii - sum_{j!=i} J_ij => Q_ii = -2 * h_i - 2 * sum_{j!=i} J_ij
    for i in range(n):
        sum_j = np.sum(ising.J[i, :]) + np.sum(ising.J[:, i])
        Q[i, i] = -2.0 * ising.h[i] - 2.0 * sum_j

    # Constant recovery: offset = constant + 0.5 * sum_i Q_ii + 0.25 * sum_{i<j} (Q_ij + Q_ji)
    sum_qii = float(np.sum(np.diag(Q)))
    sum_qij = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sum_qij += Q[i, j] + Q[j, i]
    constant = float(ising.offset - 0.5 * sum_qii - 0.25 * sum_qij)

    return Q, constant


def evaluate_ising_energy(spins: np.ndarray | List[int], ising: IsingHamiltonian) -> float:
    """Evaluates Ising Hamiltonian energy for spin configuration s in {-1, +1}^N."""
    s = np.asarray(spins, dtype=np.float64)
    energy = float(ising.offset + np.sum(ising.h * s))
    for i in range(ising.n_qubits):
        for j in range(i + 1, ising.n_qubits):
            energy += float(ising.J[i, j] * s[i] * s[j])
    return energy
