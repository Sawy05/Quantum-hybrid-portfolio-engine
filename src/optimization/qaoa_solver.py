"""Quantum Approximate Optimization Algorithm (QAOA) for Combinatorial Portfolio Selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from src.optimization.qubo_ising import (
    IsingHamiltonian,
    evaluate_ising_energy,
)


@dataclass
class QAOAResult:
    """Telemetry and solution results from QAOA portfolio optimization."""

    best_bitstring: str
    best_energy: float
    ground_state_energy: float
    ground_state_bitstring: str
    approximation_ratio: float
    probability_of_success: float
    optimal_gamma: List[float]
    optimal_beta: List[float]
    optimization_history: List[float]
    weights: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float


class QAOAPortfolioSolver:
    """Solves the portfolio Ising Hamiltonian using p-step QAOA with PennyLane."""

    def __init__(
        self,
        ising: IsingHamiltonian,
        p_steps: int = 2,
        optimizer: Literal["COBYLA", "Adam", "SPSA"] = "COBYLA",
        max_iterations: int = 80,
        device_name: str = "default.qubit",
    ):
        self.ising = ising
        self.n_qubits = ising.n_qubits
        self.p = p_steps
        self.optimizer_name = optimizer
        self.max_iterations = max_iterations
        self.dev = qml.device(device_name, wires=self.n_qubits)

        # Precompute bitstring energies for fast expectation evaluation
        self.all_bitstrings = [format(i, f"0{self.n_qubits}b") for i in range(2**self.n_qubits)]
        self.bitstring_energies = np.array(
            [self._energy_from_bitstring(b) for b in self.all_bitstrings]
        )

        self._build_circuit()

    def _energy_from_bitstring(self, bitstring: str) -> float:
        """Converts binary string (e.g. '0110') to spins s_i in {+1, -1} (where 0->+1, 1->-1) and calculates Ising energy."""
        # 0 corresponds to spin +1 (x=0), 1 corresponds to spin -1 (x=1)
        spins = np.array([1 if ch == "0" else -1 for ch in bitstring], dtype=np.float64)
        return evaluate_ising_energy(spins, self.ising)

    def _build_circuit(self) -> None:
        """Builds QAOA variational circuit producing measurement probability distribution."""
        n = self.n_qubits
        ising = self.ising

        @qml.qnode(self.dev)
        def qaoa_circuit(gamma, beta):
            # 1. Initialize uniform superposition |+>^N
            for i in range(n):
                qml.Hadamard(wires=i)

            # 2. Alternating layers of Cost and Mixer unitaries
            for step in range(self.p):
                # Cost unitary U(H_C, gamma)
                # Single-qubit rotations for h_i Z_i
                for i in range(n):
                    if abs(ising.h[i]) > 1e-8:
                        qml.RZ(2.0 * gamma[step] * ising.h[i], wires=i)

                # Two-qubit ZZ interactions for J_{ij} Z_i Z_j
                for i in range(n):
                    for j in range(i + 1, n):
                        if abs(ising.J[i, j]) > 1e-8:
                            qml.CNOT(wires=[i, j])
                            qml.RZ(2.0 * gamma[step] * ising.J[i, j], wires=j)
                            qml.CNOT(wires=[i, j])

                # Mixer unitary U(H_B, beta)
                for i in range(n):
                    qml.RX(2.0 * beta[step], wires=i)

            return qml.probs(wires=range(n))

        self.circuit = qaoa_circuit

    def cost_function(self, params: np.ndarray) -> float:
        """Computes expectation value <H_C> = sum_z P(z) E_C(z)."""
        gamma = params[: self.p]
        beta = params[self.p :]
        probs = self.circuit(gamma, beta)
        expval = float(np.dot(probs, self.bitstring_energies))
        return expval

    def exact_classical_solver(self) -> Tuple[str, float, float]:
        """Exhaustive search minimum eigensolver over 2^N states.

        Returns (ground_bitstring, ground_energy, worst_energy).
        """
        min_idx = int(np.argmin(self.bitstring_energies))
        max_idx = int(np.argmax(self.bitstring_energies))
        return (
            self.all_bitstrings[min_idx],
            float(self.bitstring_energies[min_idx]),
            float(self.bitstring_energies[max_idx]),
        )

    def simulated_annealing_solver(
        self,
        n_steps: int = 1000,
        t_start: float = 10.0,
        t_end: float = 0.01,
        seed: int = 42,
    ) -> Tuple[str, float]:
        """Classical Simulated Annealing benchmark for Ising portfolio optimization."""
        rng = np.random.default_rng(seed)
        n = self.n_qubits
        current_state = rng.choice([0, 1], size=n)
        current_energy = self._energy_from_bitstring("".join(map(str, current_state)))

        best_state = current_state.copy()
        best_energy = current_energy

        temp_decay = (t_end / t_start) ** (1.0 / n_steps)
        T = t_start

        for _ in range(n_steps):
            # Propose flipping a random spin
            flip_idx = rng.integers(0, n)
            candidate_state = current_state.copy()
            candidate_state[flip_idx] = 1 - candidate_state[flip_idx]

            cand_energy = self._energy_from_bitstring("".join(map(str, candidate_state)))
            delta_e = cand_energy - current_energy

            if delta_e < 0 or rng.random() < np.exp(-delta_e / max(1e-8, T)):
                current_state = candidate_state
                current_energy = cand_energy

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_state = current_state.copy()

            T *= temp_decay

        best_str = "".join(map(str, best_state))
        return best_str, float(best_energy)

    def _spsa_optimize(
        self,
        init_params: np.ndarray,
        a: float = 0.1,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma_spsa: float = 0.101,
    ) -> Tuple[np.ndarray, List[float]]:
        """Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer."""
        params = init_params.copy()
        dim = len(params)
        history = []
        rng = np.random.default_rng(42)

        for k in range(1, self.max_iterations + 1):
            ak = a / (k**alpha)
            ck = c / (k**gamma_spsa)

            # Bernoulli +/- 1 perturbation vector
            delta = rng.choice([-1.0, 1.0], size=dim)

            params_plus = params + ck * delta
            params_minus = params - ck * delta

            loss_plus = self.cost_function(params_plus)
            loss_minus = self.cost_function(params_minus)

            grad = (loss_plus - loss_minus) / (2.0 * ck * delta)
            params = params - ak * grad

            current_loss = self.cost_function(params)
            history.append(current_loss)

        return params, history

    def solve(
        self,
        expected_returns: Optional[np.ndarray] = None,
        covariance_matrix: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02,
    ) -> QAOAResult:
        """Executes the full QAOA optimization loop and portfolio telemetry computation."""
        # Initial parameters heuristic: gamma in [0, pi], beta in [0, pi/2]
        np.random.seed(42)
        init_gamma = np.random.uniform(0.1, 0.8, size=self.p)
        init_beta = np.random.uniform(0.1, 0.8, size=self.p)
        init_params = np.concatenate([init_gamma, init_beta])

        history: List[float] = []

        if self.optimizer_name == "SPSA":
            optimal_params, history = self._spsa_optimize(init_params)
        else:  # COBYLA or Adam/SLSQP via scipy

            def callback(x):
                val = self.cost_function(x)
                history.append(val)

            res = minimize(
                self.cost_function,
                init_params,
                method="COBYLA" if self.optimizer_name == "COBYLA" else "Nelder-Mead",
                options={"maxiter": self.max_iterations},
                callback=callback,
            )
            optimal_params = res.x

        opt_gamma = optimal_params[: self.p]
        opt_beta = optimal_params[self.p :]

        # Get final state probability distribution
        final_probs = self.circuit(opt_gamma, opt_beta)
        most_probable_idx = int(np.argmax(final_probs))
        best_bitstring = self.all_bitstrings[most_probable_idx]
        best_energy = float(self.bitstring_energies[most_probable_idx])

        # Exact ground state benchmark
        ground_bitstring, ground_energy, worst_energy = self.exact_classical_solver()
        ground_idx = self.all_bitstrings.index(ground_bitstring)
        prob_success = float(final_probs[ground_idx])

        # Approximation ratio
        denom = worst_energy - ground_energy
        if abs(denom) > 1e-8:
            approx_ratio = float((worst_energy - best_energy) / denom)
        else:
            approx_ratio = 1.0
        approx_ratio = float(np.clip(approx_ratio, 0.0, 1.0))

        # Portfolio metrics for selected bitstring
        binary_weights = np.array([int(ch) for ch in best_bitstring], dtype=np.float64)
        sum_w = np.sum(binary_weights)
        if sum_w > 0:
            norm_weights = binary_weights / sum_w
        else:
            norm_weights = np.full(self.n_qubits, 1.0 / self.n_qubits)

        mu = expected_returns if expected_returns is not None else np.full(self.n_qubits, 0.10)
        cov = covariance_matrix if covariance_matrix is not None else np.eye(self.n_qubits) * 0.04

        exp_ret = float(norm_weights @ mu)
        exp_vol = float(np.sqrt(max(1e-8, norm_weights.T @ cov @ norm_weights)))
        sharpe = float((exp_ret - risk_free_rate) / exp_vol)

        return QAOAResult(
            best_bitstring=best_bitstring,
            best_energy=best_energy,
            ground_state_energy=ground_energy,
            ground_state_bitstring=ground_bitstring,
            approximation_ratio=approx_ratio,
            probability_of_success=prob_success,
            optimal_gamma=opt_gamma.tolist(),
            optimal_beta=opt_beta.tolist(),
            optimization_history=history,
            weights=norm_weights,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe,
        )
