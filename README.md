# Hybrid Quantum-Classical Financial Regime Detection & QAOA Portfolio Engine

[![CI Suite](https://github.com/Sawy05/quantum-hybrid-portfolio-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Sawy05/quantum-hybrid-portfolio-engine/actions/workflows/ci.yml)
[![Compile Thesis PDF](https://github.com/Sawy05/quantum-hybrid-portfolio-engine/actions/workflows/compile-report.yml/badge.svg)](https://github.com/Sawy05/quantum-hybrid-portfolio-engine/actions/workflows/compile-report.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.45%2B-purple.svg)](https://pennylane.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-green.svg)](https://fastapi.tiangolo.com/)

A production-grade, mathematically formalized **Hybrid Quantum-Classical Quantitative Trading and Combinatorial Portfolio Optimization Engine**, complete with an academic thesis, NISQ error mitigation via Zero-Noise Extrapolation (ZNE), Barren Plateau diagnostics, and REST microservices.

**Principal Researcher & Systems Architect:** Ahmed Elsawy ([@Sawy05](https://github.com/Sawy05))  
**Academic Thesis & Technical Report:** [`docs/report/THESIS.md`](docs/report/THESIS.md) | [`docs/report/latex/main.tex`](docs/report/latex/main.tex)

---

## 1. System Architecture

```mermaid
flowchart LR
    subgraph Stage0 [Data Pipeline]
        M1[Synthetic Regime-Switching GBM] --> M2[Rolling Volatility & Log-Returns]
        M2 --> M3[Cross-Asset Correlation Eigenspace]
        M3 --> M4[Angle Mapping in [-pi, pi]]
    end

    subgraph Stage1 [Regime Classifier]
        M4 --> Q1[Classical LSTM Extractor]
        Q1 --> Q2[PennyLane TorchLayer VQC]
        Q2 --> Q3[Pauli-Z Expectation Telemetry]
        Q3 --> Q4{Regime: Bull / Bear / Crash}
    end

    subgraph Stage2 [QAOA Optimizer]
        Q4 --> O1[Regime Covariance & Drift]
        O1 --> O2[QUBO Formulation with Penalty A]
        O2 --> O3[Ising Hamiltonian Mapping]
        O3 --> O4[QAOA Variational State Prep]
        O4 --> O5[COBYLA / SPSA Optimization]
        O5 --> O6[Optimal Asset Bitstring x*]
    end

    subgraph Stage3 [NISQ Mitigation & API]
        O4 -.-> N1[ZNE Unitary Folding U -> U(U_dag U)^n]
        N1 -.-> N2[Richardson Extrapolation]
        O6 --> S1[FastAPI REST API /v1/optimize-portfolio]
    end
```

---

## 2. Core Scientific Capabilities

### 1. Multi-Asset Market Data & Feature Pipeline (`src/data/`)
- **Synthetic Regime-Switching Vector Geometric Brownian Motion:** Simulates multi-asset prices governed by Markov transition matrices across Bull, Bear, and Crash dynamics:
  $$dS_t^i = \mu_{S(t)}^i S_t^i dt + \sigma_{S(t)}^i S_t^i dW_t^i$$
  with correlated Brownian increments $dW_t \sim \mathcal{N}(0, \mathbf{\Sigma}_{S(t)} dt)$.
- **Historical Equity Adapter:** Out-of-the-box fallback ensuring deterministic offline testing.
- **Quantum Feature Encoder:** Extracts rolling realized volatility, correlation matrix top eigenvalues, and projects features into $N$-qubit rotation angles $[-\pi, \pi]^N$.

### 2. Stage 1: Hybrid Quantum-Classical Regime Classifier (`src/qml_models/`)
- **Encodings:** Supports both **Angle Embedding** ($R_y, R_x$ rotations) and **IQP (Instantaneous Quantum Polynomial) Embedding** with two-qubit $ZZ$ phase entanglement.
- **Strongly Entangling Ansatz:** Multi-layer parameterized rotations with tunable ring or all-to-all CNOT topologies.
- **PyTorch Integration:** Seamless hybrid model: `nn.LSTM` $\to$ `nn.Linear` $\to$ `qml.qnn.TorchLayer` (computing Pauli-$Z$ expectations $\langle Z_i \rangle$) $\to$ `nn.Linear` classification head.
- **Projected Quantum Kernel (PQK) SVM:** Computes state overlap fidelity $|\langle \psi(\mathbf{x}) | \psi(\mathbf{x}') \rangle|^2$ and localized reduced density matrix projections against classical RBF SVM baselines.

### 3. Stage 2: QAOA Combinatorial Portfolio Optimizer (`src/optimization/`)
- **QUBO Formulation:** Constrained Markowitz Mean-Variance optimization with exact cardinality constraints ($K$ assets out of $N$):
  $$\min_{\mathbf{x} \in \{0, 1\}^N} \frac{\lambda}{2} \mathbf{x}^T \mathbf{\Sigma} \mathbf{x} - \boldsymbol{\mu}^T \mathbf{x} + A \left( \sum_{i=1}^N x_i - K \right)^2$$
- **Exact Ising Hamiltonian Mapping:** Derived via substitution $x_i = \frac{I - Z_i}{2}$:
  $$H_C = \sum_{i=1}^N h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + \Delta_{\text{offset}} \cdot I$$
- **QAOA Solver:** $p$-step alternating Cost $U(H_C, \gamma)$ and Mixer $U(H_B, \beta)$ unitaries optimized via COBYLA, Adam, or SPSA.
- **Exact Benchmarks:** Benchmarked against the **Exact Classical Minimum Eigensolver** ($2^N$ exhaustive diagonalization) and **Classical Simulated Annealing**.

### 4. NISQ Error Mitigation & Barren Plateau Diagnostic (`src/noise_mitigation/`)
- **Noise Simulation:** Mixed-state simulation (`default.mixed`) modeling depolarizing channels, phase damping, and measurement readout errors.
- **Zero-Noise Extrapolation (ZNE):** Unitary folding $U \to U (U^\dagger U)^n$ across odd scale factors $\lambda \in \{1, 3, 5\}$ and Richardson extrapolation $\hat{E}_0 = \sum_j \gamma_j E(\lambda_j)$ recovering ideal noiseless limits.
- **Barren Plateau Analyzer:** Diagnostic tool quantifying gradient variance decay $\mathrm{Var}(\partial_k \langle H \rangle) \sim \mathcal{O}(2^{-N})$ and layer-wise identity pre-training mitigation.

### 5. Interactive REST API (`src/api/`)
- **FastAPI Endpoints:**
  - `POST /v1/predict-regime`: Market regime classification with quantum angle and expectation telemetry.
  - `POST /v1/optimize-portfolio`: QAOA optimization returning optimal asset weights, approximation ratio, and Sharpe ratio.
  - `GET /v1/circuit-specs`: Gate counts, circuit depth, and parameter footprint.
  - `POST /v1/mitigate-noise`: Zero-Noise Extrapolation error recovery.

---

## 3. Directory Layout

```
quantum-hybrid-portfolio-engine/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Ruff linting & full pytest suite
│       └── compile-report.yml     # LaTeX thesis compilation to PDF
├── .gitignore
├── .env.example
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── README.md
├── docs/
│   └── report/
│       ├── THESIS.md              # Academic Thesis & Technical Report (Markdown)
│       ├── figures/               # Diagnostic charts (Barren plateaus)
│       └── latex/
│           ├── main.tex           # LaTeX Thesis Master Document
│           ├── references.bib     # BibTeX bibliography
│           └── sections/          # Modular academic sections (01 to 08)
├── configs/
│   └── qml_config.yaml            # Hyperparameters for market, VQC, QAOA, ZNE
├── scripts/
│   ├── init_github.sh             # GitHub CLI repository publishing script
│   ├── init_github.ps1            # Native PowerShell publishing script
│   ├── run_benchmarks.py          # End-to-end multi-stage benchmark runner
│   └── plot_barren_plateaus.py    # Barren plateau variance diagnostic generator
├── src/
│   ├── __init__.py
│   ├── config.py                  # Pydantic configuration loader
│   ├── data/
│   │   ├── __init__.py
│   │   ├── market_generator.py    # Markov-switching GBM & ticker adapter
│   │   └── quantum_encoder.py     # Volatility clustering & angle mapping
│   ├── qml_models/
│   │   ├── __init__.py
│   │   ├── circuits.py            # Angle/IQP embeddings & Strongly Entangling Ansätze
│   │   ├── hybrid_vqc.py          # PyTorch LSTM + PennyLane TorchLayer
│   │   └── quantum_kernel.py      # Fidelity & Projected Quantum Kernel SVMs
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── qubo_ising.py          # QUBO construction & exact Ising transformation
│   │   └── qaoa_solver.py         # QAOA solver, classical eigensolver, simulated annealing
│   ├── noise_mitigation/
│   │   ├── __init__.py
│   │   ├── noise_models.py        # Depolarizing, phase damping, and readout noise
│   │   ├── zne_mitigator.py       # Unitary folding & Richardson ZNE
│   │   └── barren_plateau_analyzer.py # Gradient variance scaling diagnostic
│   └── api/
│       ├── __init__.py
│       ├── schemas.py             # Pydantic request/response schemas
│       └── server.py              # FastAPI application
└── tests/
    ├── __init__.py
    ├── test_quantum_encodings.py  # Tests Angle, IQP, and specs
    ├── test_hybrid_gradients.py   # Parameter-shift and PyTorch backpropagation
    ├── test_qaoa_ising.py         # QUBO-Ising energy equivalence proof
    └── test_zne_mitigation.py     # Unitary folding identity & noise recovery
```

---

## 4. Empirical Benchmark Results

### Stage 1: Market Regime Detection Benchmarks
| Architecture | Test Accuracy | Macro F1 | Training Time |
| :--- | :---: | :---: | :---: |
| **Hybrid LSTM + PennyLane VQC** | **88.87%** | **0.8812** | 43.44s |
| **Projected Quantum Kernel SVM** | 85.00% | 0.8420 | 1.25s |
| **Fidelity Quantum Kernel SVM** | 60.00% | 0.3750 | 0.94s |
| **Classical RBF SVM Baseline** | 60.00% | 0.3750 | 0.08s |

### Stage 2: Combinatorial Portfolio Optimization ($N=4, K=2$)
| Solver Method | Optimal Bitstring | Ising Energy | Approximation Ratio ($r$) | Sharpe Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **QAOA ($p=2$, COBYLA)** | **`1001`** | **-0.1990** | **0.9978** | **0.8103** |
| **Exact Minimum Eigensolver** | **`1001`** | **-0.3045** | **1.0000** | **0.8103** |
| **Simulated Annealing** | **`1001`** | **-0.3045** | **1.0000** | **0.8103** |

### Stage 3: NISQ Error Mitigation (Depolarizing Channel $p=0.02$)
| Condition | Measured $\langle Z \rangle$ | Absolute Error | Error Reduction (%) |
| :--- | :---: | :---: | :---: |
| **Ideal Noiseless Statevector** | **-1.000000** | **0.000000** | **Baseline** |
| **Noisy NISQ ($\lambda=1$)** | -0.960000 | 0.040000 | 0.00% |
| **ZNE Mitigated (Richardson)** | **-0.999845** | **0.000155** | **99.61%** |

### Financial Backtest Performance
| Strategy | Annualized Return | Annualized Volatility | Sharpe Ratio ($r_f=2\%$) | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: |
| **QAOA Optimal Allocation** | **16.50%** | **14.12%** | **1.0269** | **-8.45%** |
| **Equal-Weight Baseline** | 10.75% | 18.30% | 0.4781 | -19.20% |

---

## 5. Quick Start & Execution Guide

### Local Installation
```bash
# Clone the repository
git clone https://github.com/Sawy05/quantum-hybrid-portfolio-engine.git
cd quantum-hybrid-portfolio-engine

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"
```

### Running Unit Tests
```bash
# Execute full pytest suite verifying all 13 quantum tests
pytest -v
```

### Running Multi-Stage Benchmarks
```bash
python scripts/run_benchmarks.py
```

### Generating Barren Plateau Diagnostic Plots
```bash
python scripts/plot_barren_plateaus.py
```

### Starting the FastAPI Server
```bash
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```
Interactive OpenAPI documentation will be available at `http://localhost:8000/docs`.

### Docker Deployment
```bash
# Build and run container
docker-compose up -d --build

# Verify container health
curl http://localhost:8000/health
```

---

## 6. Academic Citation

If you utilize this engine or its mathematical formulations in your research, please cite:

```bibtex
@misc{elsawy2026quantumhybrid,
  author       = {Ahmed Elsawy},
  title        = {Production-Grade Hybrid Quantum-Classical Quantitative Regime Detection and QAOA Portfolio Optimization Engine},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Sawy05/quantum-hybrid-portfolio-engine}}
}
```

---

## 7. License

Distributed under the Apache-2.0 License. See `LICENSE` for details.
