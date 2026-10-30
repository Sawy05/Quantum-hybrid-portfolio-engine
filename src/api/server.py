"""FastAPI server for Quantum Hybrid Regime Detection and QAOA Portfolio Optimization."""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    CircuitSpecsResponse,
    NoiseMitigationRequest,
    NoiseMitigationResponse,
    PortfolioOptimizationRequest,
    PortfolioOptimizationResponse,
    RegimePredictionRequest,
    RegimePredictionResponse,
)
from src.config import load_config
from src.data.market_generator import MarketDataGenerator
from src.noise_mitigation.zne_mitigator import ZeroNoiseExtrapolator
from src.optimization.qaoa_solver import QAOAPortfolioSolver
from src.optimization.qubo_ising import PortfolioQUBO
from src.qml_models.circuits import QuantumCircuitFactory
from src.qml_models.hybrid_vqc import HybridQuantumClassifier

config = load_config()

app = FastAPI(
    title="Hybrid Quantum-Classical Financial Regime & QAOA Portfolio Engine",
    description="Production-grade QML Engine combining VQC regime classification and QAOA portfolio optimization.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
_model: HybridQuantumClassifier | None = None


def get_model() -> HybridQuantumClassifier:
    """Lazily initializes and returns the hybrid model instance."""
    global _model
    if _model is None:
        _model = HybridQuantumClassifier(
            n_qubits=config.quantum_circuit.n_qubits,
            circuit_depth=config.quantum_circuit.circuit_depth,
            feature_dim=config.hybrid_classifier.feature_dim,
            hidden_dim=config.hybrid_classifier.hidden_dim,
            num_classes=config.hybrid_classifier.num_classes,
            encoding_type=config.quantum_circuit.encoding_type,
            entanglement_topology=config.quantum_circuit.entanglement_topology,
            diff_method=config.quantum_circuit.diff_method,
        )
    return _model


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "quantum-hybrid-portfolio-engine",
        "quantum_backend": "PennyLane-TorchLayer",
        "n_qubits": config.quantum_circuit.n_qubits,
    }


@app.get("/v1/circuit-specs", response_model=CircuitSpecsResponse)
def circuit_specs():
    """Returns variational quantum circuit specifications, depth, and gate footprint."""
    factory = QuantumCircuitFactory(
        n_qubits=config.quantum_circuit.n_qubits,
        circuit_depth=config.quantum_circuit.circuit_depth,
        encoding_type=config.quantum_circuit.encoding_type,
        entanglement_topology=config.quantum_circuit.entanglement_topology,
        diff_method=config.quantum_circuit.diff_method,
    )
    specs = factory.get_circuit_specs()
    return CircuitSpecsResponse(**specs)


@app.post("/v1/predict-regime", response_model=RegimePredictionResponse)
def predict_regime(req: RegimePredictionRequest):
    """Predicts market regime using Classical LSTM -> PennyLane VQC -> Dense Head."""
    model = get_model()
    regime_names = ["BULL", "BEAR", "CRASH"]

    if req.returns_series is not None:
        raw_seq = np.array(req.returns_series, dtype=np.float32)
        if raw_seq.ndim == 2:
            raw_seq = np.expand_dims(raw_seq, axis=0)
    else:
        # Generate representative slice using synthetic market data
        gen = MarketDataGenerator(
            n_assets=config.quantum_circuit.n_qubits,
            n_steps=req.sequence_length + 20,
            random_seed=42,
        )
        data = gen.generate_synthetic()
        raw_seq = data.returns.iloc[-req.sequence_length :].to_numpy()
        raw_seq = np.expand_dims(raw_seq, axis=0).astype(np.float32)

    preds, probs, q_angles, q_expvals = model.predict_regime(raw_seq)

    reg_id = int(preds[0])
    reg_name = regime_names[reg_id]
    prob_dict = {regime_names[i]: float(probs[0][i]) for i in range(len(regime_names))}

    return RegimePredictionResponse(
        predicted_regime=reg_name,
        regime_id=reg_id,
        probabilities=prob_dict,
        quantum_rotation_angles=q_angles[0].tolist(),
        pauli_z_expectations=q_expvals[0].tolist(),
    )


@app.post("/v1/optimize-portfolio", response_model=PortfolioOptimizationResponse)
def optimize_portfolio(req: PortfolioOptimizationRequest):
    """Solves cardinality-constrained Markowitz portfolio problem via QAOA and classical eigensolvers."""
    n_assets = (
        len(req.expected_returns) if req.expected_returns else config.quantum_circuit.n_qubits
    )
    tickers = req.tickers or [f"ASSET_{i}" for i in range(n_assets)]

    if req.expected_returns is not None:
        mu = np.array(req.expected_returns, dtype=np.float64)
    else:
        # Default realistic returns
        mu = np.array([0.14, 0.11, 0.08, 0.16][:n_assets], dtype=np.float64)

    if req.covariance_matrix is not None:
        cov = np.array(req.covariance_matrix, dtype=np.float64)
    else:
        # Default covariance
        vols = np.array([0.15, 0.18, 0.12, 0.22][:n_assets])
        corr = 0.5 * np.ones((n_assets, n_assets)) + 0.5 * np.eye(n_assets)
        cov = np.outer(vols, vols) * corr

    # Formulate QUBO and Ising
    qubo = PortfolioQUBO(
        expected_returns=mu,
        covariance_matrix=cov,
        risk_aversion=req.risk_aversion,
        penalty_coeff=req.penalty_coeff,
        cardinality_k=req.cardinality_k,
    )
    ising = qubo.to_ising()

    # Solve with QAOA
    solver = QAOAPortfolioSolver(
        ising=ising,
        p_steps=req.p_steps,
        optimizer=req.optimizer,
        max_iterations=req.max_iterations,
    )
    qaoa_res = solver.solve(expected_returns=mu, covariance_matrix=cov)

    # Solve with Simulated Annealing for comparison
    sa_bitstring, sa_energy = solver.simulated_annealing_solver()

    selected = [tickers[i] for i, ch in enumerate(qaoa_res.best_bitstring) if ch == "1"]
    weights_dict = {tickers[i]: float(qaoa_res.weights[i]) for i in range(n_assets)}

    return PortfolioOptimizationResponse(
        best_bitstring=qaoa_res.best_bitstring,
        selected_assets=selected,
        asset_weights=weights_dict,
        approximation_ratio=qaoa_res.approximation_ratio,
        probability_of_success=qaoa_res.probability_of_success,
        best_energy=qaoa_res.best_energy,
        ground_state_energy=qaoa_res.ground_state_energy,
        ground_state_bitstring=qaoa_res.ground_state_bitstring,
        expected_annual_return=qaoa_res.expected_return,
        expected_annual_volatility=qaoa_res.expected_volatility,
        sharpe_ratio=qaoa_res.sharpe_ratio,
        classical_simulated_annealing_energy=sa_energy,
        classical_sa_bitstring=sa_bitstring,
        qaoa_gamma=qaoa_res.optimal_gamma,
        qaoa_beta=qaoa_res.optimal_beta,
    )


@app.post("/v1/mitigate-noise", response_model=NoiseMitigationResponse)
def mitigate_noise(req: NoiseMitigationRequest):
    """Runs Zero-Noise Extrapolation (ZNE) to recover noiseless expectation limit."""
    zne = ZeroNoiseExtrapolator(
        scale_factors=req.scale_factors,
        extrapolation_method=req.method,
    )

    true_expval = 0.85
    noise_rate = 0.05
    # Synthetic noisy observation curve: <O>(lambda) = true * exp(-noise * lambda)
    noisy_vals = [
        float(true_expval * np.exp(-noise_rate * s) + np.random.normal(0, 0.005))
        for s in req.scale_factors
    ]
    unmitigated = noisy_vals[0]
    mitigated, _ = zne.extrapolate(req.scale_factors, noisy_vals)

    err_unmitigated = abs(unmitigated - true_expval)
    err_mitigated = abs(mitigated - true_expval)
    reduction = max(
        0.0, float((err_unmitigated - err_mitigated) / max(1e-8, err_unmitigated)) * 100.0
    )

    return NoiseMitigationResponse(
        mitigated_value=mitigated,
        unmitigated_value=unmitigated,
        scale_factors=req.scale_factors,
        noisy_values=noisy_vals,
        method=req.method,
        error_reduction_pct=reduction,
    )
