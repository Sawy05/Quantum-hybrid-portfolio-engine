"""Pydantic request and response schemas for REST API."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RegimePredictionRequest(BaseModel):
    returns_series: Optional[List[List[float]]] = Field(
        default=None,
        description="Temporal sequence of returns (Seq_Len, N_Assets). If omitted, generated automatically.",
    )
    tickers: Optional[List[str]] = Field(
        default=None,
        description="Asset ticker symbols.",
    )
    sequence_length: int = Field(default=10, ge=3, le=50)


class RegimePredictionResponse(BaseModel):
    predicted_regime: str
    regime_id: int
    probabilities: Dict[str, float]
    quantum_rotation_angles: List[float]
    pauli_z_expectations: List[float]
    model_architecture: str = "LSTM-PennyLane-TorchLayer"


class PortfolioOptimizationRequest(BaseModel):
    expected_returns: Optional[List[float]] = Field(
        default=None,
        description="Expected annualized returns vector. If omitted, standard regime benchmark is used.",
    )
    covariance_matrix: Optional[List[List[float]]] = Field(
        default=None,
        description="Asset return covariance matrix.",
    )
    tickers: Optional[List[str]] = None
    risk_aversion: float = Field(default=0.5, ge=0.01, le=10.0)
    cardinality_k: int = Field(default=2, ge=1)
    penalty_coeff: float = Field(default=10.0, ge=0.1)
    p_steps: int = Field(default=2, ge=1, le=6)
    optimizer: Literal["COBYLA", "Adam", "SPSA"] = "COBYLA"
    max_iterations: int = Field(default=80, ge=10, le=500)


class PortfolioOptimizationResponse(BaseModel):
    best_bitstring: str
    selected_assets: List[str]
    asset_weights: Dict[str, float]
    approximation_ratio: float
    probability_of_success: float
    best_energy: float
    ground_state_energy: float
    ground_state_bitstring: str
    expected_annual_return: float
    expected_annual_volatility: float
    sharpe_ratio: float
    classical_simulated_annealing_energy: float
    classical_sa_bitstring: str
    qaoa_gamma: List[float]
    qaoa_beta: List[float]


class CircuitSpecsResponse(BaseModel):
    n_qubits: int
    circuit_depth: int
    encoding_type: str
    entanglement_topology: str
    trainable_parameters: int
    weight_shape: List[int]
    diff_method: str
    gate_types: Dict[str, int]


class NoiseMitigationRequest(BaseModel):
    unmitigated_value: Optional[float] = None
    scale_factors: List[int] = Field(default_factory=lambda: [1, 3, 5])
    method: Literal["richardson", "polynomial", "linear"] = "richardson"


class NoiseMitigationResponse(BaseModel):
    mitigated_value: float
    unmitigated_value: float
    scale_factors: List[int]
    noisy_values: List[float]
    method: str
    error_reduction_pct: float
