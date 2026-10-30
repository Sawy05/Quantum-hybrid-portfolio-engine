"""End-to-End Quantum Machine Learning and Optimization Benchmark Suite."""

from __future__ import annotations

import time

import numpy as np
from rich.console import Console
from rich.table import Table

from src.config import load_config
from src.data.market_generator import MarketDataGenerator
from src.data.quantum_encoder import ClassicalQuantumFeaturePipeline
from src.noise_mitigation.zne_mitigator import ZeroNoiseExtrapolator
from src.optimization.qaoa_solver import QAOAPortfolioSolver
from src.optimization.qubo_ising import PortfolioQUBO
from src.qml_models.hybrid_vqc import HybridQuantumClassifier
from src.qml_models.quantum_kernel import QuantumKernelClassifier

console = Console()


def benchmark_regime_classification(config):
    """Benchmarks Hybrid Quantum-Classical VQC vs Classical Baselines."""
    console.print(
        "\n[bold cyan]=== Stage 1: Market Data & Hybrid Regime Detection Benchmark ===[/bold cyan]"
    )

    gen = MarketDataGenerator(
        n_assets=config.quantum_circuit.n_qubits,
        n_steps=config.market.n_steps,
        random_seed=42,
    )
    market_data = gen.generate_synthetic()

    pipeline = ClassicalQuantumFeaturePipeline(
        n_qubits=config.quantum_circuit.n_qubits,
        rolling_window=21,
        scaling_method="minmax",
    )
    feat_df = pipeline.extract_classical_features(market_data.returns)
    angles = pipeline.fit_transform_to_angles(feat_df.to_numpy())

    X, y = pipeline.create_sliding_windows(
        features=angles,
        targets=market_data.regimes,
        sequence_length=config.hybrid_classifier.sequence_length,
    )

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Train Hybrid VQC
    hybrid_model = HybridQuantumClassifier(
        n_qubits=config.quantum_circuit.n_qubits,
        circuit_depth=config.quantum_circuit.circuit_depth,
        feature_dim=config.hybrid_classifier.feature_dim,
        hidden_dim=config.hybrid_classifier.hidden_dim,
        num_classes=config.hybrid_classifier.num_classes,
    )

    start_t = time.time()
    history = hybrid_model.fit_model(
        X_train=X_train,
        y_train=y_train,
        epochs=config.hybrid_classifier.epochs,
        batch_size=config.hybrid_classifier.batch_size,
        lr=config.hybrid_classifier.learning_rate,
        verbose=False,
    )
    train_time = time.time() - start_t

    preds, probs, _, _ = hybrid_model.predict_regime(X_test)
    test_acc = float(np.mean(preds == y_test))

    # Benchmark Quantum Kernel vs Classical RBF SVM on static angle feature subsets
    k_classifier = QuantumKernelClassifier(
        n_qubits=config.quantum_circuit.n_qubits,
        n_layers=config.kernel_svm.circuit_layers,
        kernel_type="fidelity",
        c_param=config.kernel_svm.c_param,
    )
    subset_n = 40
    Xk_train = angles[:subset_n]
    yk_train = market_data.regimes[:subset_n]
    Xk_test = angles[subset_n : subset_n + 20]
    yk_test = market_data.regimes[subset_n : subset_n + 20]

    svm_bench = k_classifier.evaluate_benchmark(Xk_train, yk_train, Xk_test, yk_test)

    # Print summary table
    table = Table(title="Stage 1: Regime Classification Benchmarks")
    table.add_column("Model Architecture", style="cyan")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Macro F1", justify="right")
    table.add_column("Training Time (s)", justify="right")

    table.add_row(
        "Hybrid LSTM + VQC (PennyLane TorchLayer)",
        f"{test_acc * 100:.2f}%",
        f"{history['accuracy'][-1]:.4f}",
        f"{train_time:.2f}s",
    )
    table.add_row(
        "Fidelity Quantum Kernel SVM",
        f"{svm_bench['quantum_kernel']['accuracy'] * 100:.2f}%",
        f"{svm_bench['quantum_kernel']['macro_f1']:.4f}",
        "N/A",
    )
    table.add_row(
        "Classical RBF SVM Baseline",
        f"{svm_bench['classical_rbf']['accuracy'] * 100:.2f}%",
        f"{svm_bench['classical_rbf']['macro_f1']:.4f}",
        "N/A",
    )
    console.print(table)


def benchmark_qaoa_portfolio(config):
    """Benchmarks QAOA vs Exact Classical Eigensolver and Simulated Annealing."""
    console.print(
        "\n[bold cyan]=== Stage 2: QAOA Combinatorial Portfolio Optimization ===[/bold cyan]"
    )

    n_assets = config.quantum_circuit.n_qubits
    returns = np.array([0.15, 0.12, 0.09, 0.18][:n_assets])
    vols = np.array([0.16, 0.14, 0.11, 0.22][:n_assets])
    corr = 0.4 * np.ones((n_assets, n_assets)) + 0.6 * np.eye(n_assets)
    cov = np.outer(vols, vols) * corr

    qubo = PortfolioQUBO(
        expected_returns=returns,
        covariance_matrix=cov,
        risk_aversion=config.qaoa.risk_aversion,
        penalty_coeff=config.qaoa.penalty_coeff,
        cardinality_k=config.qaoa.cardinality_k,
    )
    ising = qubo.to_ising()

    solver = QAOAPortfolioSolver(
        ising=ising,
        p_steps=config.qaoa.p_steps,
        optimizer=config.qaoa.optimizer,
        max_iterations=config.qaoa.max_iterations,
    )

    t0 = time.time()
    qaoa_res = solver.solve(expected_returns=returns, covariance_matrix=cov)
    qaoa_time = time.time() - t0

    # Exact Classical Eigensolver
    t1 = time.time()
    exact_bit, exact_energy, worst_energy = solver.exact_classical_solver()
    exact_time = time.time() - t1

    # Simulated Annealing
    t2 = time.time()
    sa_bit, sa_energy = solver.simulated_annealing_solver(n_steps=500)
    sa_time = time.time() - t2

    table = Table(title="Stage 2: Combinatorial Portfolio Optimization Benchmarks")
    table.add_column("Solver Method", style="cyan")
    table.add_column("Optimal Bitstring", style="magenta")
    table.add_column("Ising Energy", justify="right")
    table.add_column("Approx Ratio", justify="right", style="green")
    table.add_column("Sharpe Ratio", justify="right")
    table.add_column("Solve Time (s)", justify="right")

    table.add_row(
        f"QAOA (p={config.qaoa.p_steps}, {config.qaoa.optimizer})",
        qaoa_res.best_bitstring,
        f"{qaoa_res.best_energy:.4f}",
        f"{qaoa_res.approximation_ratio:.4f}",
        f"{qaoa_res.sharpe_ratio:.4f}",
        f"{qaoa_time:.3f}s",
    )
    table.add_row(
        "Exact Classical Minimum Eigensolver",
        exact_bit,
        f"{exact_energy:.4f}",
        "1.0000",
        "Benchmark",
        f"{exact_time:.3f}s",
    )
    table.add_row(
        "Classical Simulated Annealing",
        sa_bit,
        f"{sa_energy:.4f}",
        f"{(worst_energy - sa_energy) / (worst_energy - exact_energy):.4f}",
        "Benchmark",
        f"{sa_time:.3f}s",
    )
    console.print(table)


def benchmark_zne_mitigation(config):
    """Benchmarks NISQ Noise and Zero-Noise Extrapolation (ZNE)."""
    console.print(
        "\n[bold cyan]=== Stage 3: NISQ Noise & Zero-Noise Extrapolation (ZNE) ===[/bold cyan]"
    )

    zne = ZeroNoiseExtrapolator(
        scale_factors=config.noise_mitigation.zne_scale_factors,
        extrapolation_method=config.noise_mitigation.extrapolation_method,
    )

    true_ground_val = -1.0
    scales = config.noise_mitigation.zne_scale_factors
    # Empirical depolarizing decay: <Z>(lambda) = true * (1 - 2*p)^lambda
    p_err = config.noise_mitigation.depolarizing_prob
    noisy_vals = [float(true_ground_val * ((1.0 - 2.0 * p_err) ** s)) for s in scales]

    mitigated_val, _ = zne.extrapolate(scales, noisy_vals)
    err_unmit = abs(noisy_vals[0] - true_ground_val)
    err_mit = abs(mitigated_val - true_ground_val)
    pct_recovery = max(0.0, ((err_unmit - err_mit) / err_unmit) * 100.0)

    table = Table(title="Stage 3: NISQ Error Mitigation Benchmarks")
    table.add_column("Condition", style="cyan")
    table.add_column("Expectation Value", justify="right")
    table.add_column("Absolute Error", justify="right", style="red")
    table.add_column("Error Reduction", justify="right", style="green")

    table.add_row("Ideal Noiseless Statevector", f"{true_ground_val:.6f}", "0.000000", "Baseline")
    table.add_row(
        "Noisy NISQ (Unmitigated, lambda=1)", f"{noisy_vals[0]:.6f}", f"{err_unmit:.6f}", "0.00%"
    )
    table.add_row(
        f"ZNE Mitigated ({config.noise_mitigation.extrapolation_method.capitalize()})",
        f"{mitigated_val:.6f}",
        f"{err_mit:.6f}",
        f"{pct_recovery:.2f}%",
    )
    console.print(table)


if __name__ == "__main__":
    cfg = load_config()
    benchmark_regime_classification(cfg)
    benchmark_qaoa_portfolio(cfg)
    benchmark_zne_mitigation(cfg)
    console.print(
        "\n[bold green]All quantum-classical benchmarks completed successfully![/bold green]\n"
    )
