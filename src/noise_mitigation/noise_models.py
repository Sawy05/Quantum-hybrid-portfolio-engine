"""NISQ noise models simulating depolarizing, phase damping, and readout bit-flip channels."""

from __future__ import annotations

from typing import List

import pennylane as qml


class NISQNoiseModel:
    """Simulates realistic NISQ hardware noise on PennyLane's default.mixed density matrix simulator."""

    def __init__(
        self,
        n_qubits: int = 4,
        depolarizing_prob: float = 0.02,
        phase_damping_prob: float = 0.01,
        readout_error_prob: float = 0.02,
    ):
        self.n_qubits = n_qubits
        self.depolarizing_prob = depolarizing_prob
        self.phase_damping_prob = phase_damping_prob
        self.readout_error_prob = readout_error_prob
        self.dev = qml.device("default.mixed", wires=n_qubits)

    def apply_gate_noise(self, wires: List[int]) -> None:
        """Injects depolarizing and phase damping noise channels across target qubits."""
        for w in wires:
            if self.depolarizing_prob > 0.0:
                qml.DepolarizingChannel(self.depolarizing_prob, wires=w)
            if self.phase_damping_prob > 0.0:
                qml.PhaseDamping(self.phase_damping_prob, wires=w)

    def apply_readout_noise(self, wires: List[int]) -> None:
        """Injects bit-flip measurement noise simulating imperfect detector readout."""
        for w in wires:
            if self.readout_error_prob > 0.0:
                qml.BitFlip(self.readout_error_prob, wires=w)

    def corrupt_expectation(self, true_val: float) -> float:
        """Classical readout bit-flip corruption model for expectation values: <Z>_noisy = (1 - 2*p_readout) * <Z>."""
        scale = 1.0 - 2.0 * self.readout_error_prob
        return float(scale * true_val)
