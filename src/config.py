"""Configuration models and loader for Hybrid Quantum-Classical Engine."""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, Field


class RegimeConfig(BaseModel):
    name: str
    drift: List[float]
    volatility: List[float]


class MarketConfig(BaseModel):
    n_assets: int = 4
    n_steps: int = 1200
    dt: float = 1.0 / 252.0
    random_seed: int = 42
    regimes: List[RegimeConfig] = Field(default_factory=list)
    transition_matrix: List[List[float]] = Field(default_factory=list)


class QuantumCircuitConfig(BaseModel):
    n_qubits: int = 4
    circuit_depth: int = 3
    encoding_type: Literal["angle", "iqp"] = "angle"
    entanglement_topology: Literal["ring", "all-to-all"] = "ring"
    diff_method: Literal["parameter-shift", "adjoint"] = "parameter-shift"


class HybridClassifierConfig(BaseModel):
    sequence_length: int = 10
    feature_dim: int = 4
    hidden_dim: int = 16
    num_classes: int = 3
    learning_rate: float = 0.01
    batch_size: int = 16
    epochs: int = 15


class KernelSVMConfig(BaseModel):
    n_qubits: int = 4
    circuit_layers: int = 2
    c_param: float = 1.0


class QAOAConfig(BaseModel):
    p_steps: int = 2
    optimizer: Literal["COBYLA", "Adam", "SPSA"] = "COBYLA"
    max_iterations: int = 80
    learning_rate: float = 0.05
    risk_aversion: float = 0.5
    penalty_coeff: float = 12.0
    cardinality_k: int = 2


class NoiseMitigationConfig(BaseModel):
    depolarizing_prob: float = 0.02
    phase_damping_prob: float = 0.01
    readout_error_prob: float = 0.02
    zne_scale_factors: List[int] = Field(default_factory=lambda: [1, 3, 5])
    extrapolation_method: Literal["richardson", "polynomial", "linear"] = "richardson"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class AppConfig(BaseModel):
    market: MarketConfig = Field(default_factory=MarketConfig)
    quantum_circuit: QuantumCircuitConfig = Field(default_factory=QuantumCircuitConfig)
    hybrid_classifier: HybridClassifierConfig = Field(default_factory=HybridClassifierConfig)
    kernel_svm: KernelSVMConfig = Field(default_factory=KernelSVMConfig)
    qaoa: QAOAConfig = Field(default_factory=QAOAConfig)
    noise_mitigation: NoiseMitigationConfig = Field(default_factory=NoiseMitigationConfig)
    api: APIConfig = Field(default_factory=APIConfig)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load application configuration from YAML file or default values."""
    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        config_path = base_dir / "configs" / "qml_config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        return AppConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return AppConfig(**data)
