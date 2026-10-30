"""Zero-Noise Extrapolation (ZNE) via Unitary Folding and Richardson Extrapolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional, Tuple

import numpy as np
import pennylane as qml

from src.noise_mitigation.noise_models import NISQNoiseModel


@dataclass
class ZNEResult:
    """Telemetry and outcome of Zero-Noise Extrapolation."""

    mitigated_value: float
    unmitigated_value: float
    scale_factors: List[int]
    noisy_values: List[float]
    method: str
    polynomial_coefficients: Optional[List[float]] = None


class ZeroNoiseExtrapolator:
    r"""Implements unitary folding U -> U (U^\dagger U)^n and Richardson / polynomial extrapolation."""

    def __init__(
        self,
        scale_factors: Optional[List[int]] = None,
        extrapolation_method: Literal["richardson", "polynomial", "linear"] = "richardson",
        poly_degree: int = 2,
    ):
        self.scale_factors = scale_factors or [1, 3, 5]
        self.method = extrapolation_method
        self.poly_degree = poly_degree

    @staticmethod
    def fold_circuit_unitary(
        circuit_fn: Callable[[], None],
        scale_factor: int,
    ) -> None:
        r"""Folds the unitary block U -> U (U^\dagger U)^n such that total depth scales by lambda = 2n + 1."""
        if scale_factor < 1 or scale_factor % 2 == 0:
            raise ValueError(
                f"Unitary folding requires odd scale factors lambda >= 1. Got {scale_factor}"
            )

        n_folds = (scale_factor - 1) // 2

        # Initial forward pass U
        circuit_fn()

        # Followed by n_folds pairs of (U^\dagger U)
        for _ in range(n_folds):
            qml.adjoint(circuit_fn)()
            circuit_fn()

    @staticmethod
    def richardson_coefficients(scales: List[int] | np.ndarray) -> np.ndarray:
        """Calculates Richardson extrapolation weights gamma_j = prod_{l != j} lambda_l / (lambda_l - lambda_j)."""
        scales_arr = np.asarray(scales, dtype=np.float64)
        K = len(scales_arr)
        weights = np.zeros(K, dtype=np.float64)

        for j in range(K):
            prod = 1.0
            for idx in range(K):
                if idx != j:
                    prod *= scales_arr[idx] / (scales_arr[idx] - scales_arr[j])
            weights[j] = prod

        return weights

    def extrapolate(
        self,
        scales: List[int] | np.ndarray,
        noisy_values: List[float] | np.ndarray,
        method: Optional[Literal["richardson", "polynomial", "linear"]] = None,
    ) -> Tuple[float, Optional[List[float]]]:
        """Extrapolates noisy measurements to the zero-noise limit lambda -> 0."""
        chosen_method = method or self.method
        scales_arr = np.asarray(scales, dtype=np.float64)
        vals_arr = np.asarray(noisy_values, dtype=np.float64)

        if chosen_method == "richardson":
            weights = self.richardson_coefficients(scales_arr)
            mitigated = float(np.dot(weights, vals_arr))
            return mitigated, None

        elif chosen_method == "linear":
            coeffs = np.polyfit(scales_arr, vals_arr, deg=1)
            # coeffs = [slope, intercept]; at lambda=0, val = intercept
            mitigated = float(coeffs[1])
            return mitigated, coeffs.tolist()

        else:  # polynomial
            deg = min(self.poly_degree, len(scales_arr) - 1)
            coeffs = np.polyfit(scales_arr, vals_arr, deg=deg)
            # Evaluate at lambda = 0 (the constant term, which is the last coefficient in numpy polyfit)
            mitigated = float(coeffs[-1])
            return mitigated, coeffs.tolist()

    def run_mitigated_evaluation(
        self,
        circuit_fn: Callable[[], None],
        observable_fn: Callable[[], Any],
        noise_model: NISQNoiseModel,
        wires: List[int],
    ) -> ZNEResult:
        """Executes unitary folding across noise scales on a noisy simulator and extrapolates to zero noise."""
        dev = noise_model.dev
        noisy_values: List[float] = []

        for scale in self.scale_factors:

            @qml.qnode(dev)
            def noisy_folded_circuit():
                # Folded unitary
                self.fold_circuit_unitary(circuit_fn, scale_factor=scale)
                # Gate noise injected across wires
                noise_model.apply_gate_noise(wires)
                noise_model.apply_readout_noise(wires)
                return observable_fn()

            val = float(noisy_folded_circuit())
            noisy_values.append(val)

        unmitigated_val = noisy_values[0]
        mitigated_val, coeffs = self.extrapolate(self.scale_factors, noisy_values)

        return ZNEResult(
            mitigated_value=mitigated_val,
            unmitigated_value=unmitigated_val,
            scale_factors=self.scale_factors,
            noisy_values=noisy_values,
            method=self.method,
            polynomial_coefficients=coeffs,
        )
