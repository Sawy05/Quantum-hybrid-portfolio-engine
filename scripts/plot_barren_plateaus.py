"""Analyzes and plots Barren Plateau gradient variance decay across qubit scales."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.noise_mitigation.barren_plateau_analyzer import BarrenPlateauAnalyzer


def main():
    print("Initializing Barren Plateau Analyzer...")
    analyzer = BarrenPlateauAnalyzer(seed=42)

    qubits = [4, 6, 8]
    print(f"Scanning qubit counts: {qubits} with depth=3 and 25 random samples per point...")
    results = analyzer.scan_qubit_scaling(qubit_range=qubits, circuit_depth=3, n_samples=25)

    print("\n--- Empirical Gradient Variance Results ---")
    for q, v_rand, v_ident in zip(
        results["qubit_counts"],
        results["random_init_variances"],
        results["identity_init_variances"],
    ):
        print(
            f"Qubits: {q:2d} | Random Init Var: {v_rand:.6e} | Identity/Layerwise Var: {v_ident:.6e}"
        )

    # Generate and save diagnostic plot
    out_dir = Path(__file__).resolve().parent.parent / "docs" / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "barren_plateaus.png"

    plt.figure(figsize=(8, 5))
    plt.semilogy(
        qubits, results["random_init_variances"], "ro-", label="Random Haar Init (Barren Plateau)"
    )
    plt.semilogy(
        qubits,
        results["identity_init_variances"],
        "bs--",
        label="Layer-wise Identity Init (Mitigated)",
    )
    plt.title(
        r"Barren Plateau Scaling: $\mathrm{Var}\left(\partial_\theta \langle H \rangle\right)$ vs Qubit Count $N$"
    )
    plt.xlabel(r"Qubit Count ($N$)")
    plt.ylabel(r"Gradient Variance $\mathrm{Var}(\partial_k \langle H \rangle)$ (log scale)")
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"\nPlot saved successfully to: {out_path}")


if __name__ == "__main__":
    main()
