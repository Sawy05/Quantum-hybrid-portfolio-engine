"""Classical financial feature engineering and quantum state angle embedding transformations."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class ClassicalQuantumFeaturePipeline:
    """Extracts classical quantitative features and maps them into N-qubit rotation angles [-pi, pi]."""

    def __init__(
        self,
        n_qubits: int = 4,
        rolling_window: int = 21,
        scaling_method: Literal["pca", "minmax", "arctan"] = "minmax",
    ):
        self.n_qubits = n_qubits
        self.rolling_window = rolling_window
        self.scaling_method = scaling_method
        self.pca: Optional[PCA] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.standard_scaler: Optional[StandardScaler] = None

    def extract_classical_features(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Computes rolling annualized returns, realized volatility, and correlation eigenvalues."""
        features = pd.DataFrame(index=returns_df.index)

        # 1. Rolling Mean Log-Returns (annualized)
        rolling_mean = returns_df.rolling(window=self.rolling_window).mean() * 252.0
        for col in returns_df.columns:
            features[f"{col}_rolling_mean"] = rolling_mean[col]

        # 2. Rolling Realized Volatility (annualized, volatility clustering)
        rolling_vol = returns_df.rolling(window=self.rolling_window).std() * np.sqrt(252.0)
        for col in returns_df.columns:
            features[f"{col}_rolling_vol"] = rolling_vol[col]

        # 3. Cross-asset dispersion / market-wide volatility proxy
        features["market_avg_vol"] = rolling_vol.mean(axis=1)
        features["cross_asset_dispersion"] = returns_df.std(axis=1)

        # 4. Rolling Correlation Matrix Top Eigenvalues
        n_assets = returns_df.shape[1]
        top_eigenvalues = []
        for i in range(len(returns_df)):
            if i < self.rolling_window:
                top_eigenvalues.append([1.0] * min(2, n_assets))
                continue
            window_slice = returns_df.iloc[i - self.rolling_window : i]
            corr = window_slice.corr().fillna(0.0).to_numpy()
            evals = np.linalg.eigvalsh(corr)
            evals = np.sort(evals)[::-1]  # Descending
            top_eigenvalues.append(evals[: min(2, n_assets)].tolist())

        eig_df = pd.DataFrame(
            top_eigenvalues,
            index=returns_df.index,
            columns=[f"corr_eig_{k}" for k in range(min(2, n_assets))],
        )
        features = pd.concat([features, eig_df], axis=1)

        # Drop initial NaN rows created by rolling windows
        features = features.bfill().ffill()
        return features

    def fit_transform_to_angles(
        self,
        features: np.ndarray,
        method: Optional[Literal["pca", "minmax", "arctan"]] = None,
    ) -> np.ndarray:
        """Fits scalers and transforms multidimensional features into N-qubit rotation angles [-pi, pi]."""
        chosen_method = method or self.scaling_method

        if chosen_method == "pca":
            self.standard_scaler = StandardScaler()
            scaled_feat = self.standard_scaler.fit_transform(features)
            self.pca = PCA(n_components=self.n_qubits)
            reduced = self.pca.fit_transform(scaled_feat)

            self.scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
            angles = self.scaler.fit_transform(reduced)
            return angles

        elif chosen_method == "arctan":
            # Map through standard scale then 2 * arctan(z) which strictly lies in (-pi, pi)
            self.standard_scaler = StandardScaler()
            scaled_feat = self.standard_scaler.fit_transform(features)
            # If feature dimension > n_qubits, reduce via PCA first
            if scaled_feat.shape[1] != self.n_qubits:
                self.pca = PCA(n_components=self.n_qubits)
                scaled_feat = self.pca.fit_transform(scaled_feat)
            angles = 2.0 * np.arctan(scaled_feat)
            return np.clip(angles, -np.pi, np.pi)

        else:  # MinMax scaling directly to [-pi, pi]
            if features.shape[1] != self.n_qubits:
                self.pca = PCA(n_components=self.n_qubits)
                reduced = self.pca.fit_transform(features)
            else:
                reduced = features
            self.scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
            angles = self.scaler.fit_transform(reduced)
            return angles

    def transform_to_angles(
        self,
        features: np.ndarray,
        method: Optional[Literal["pca", "minmax", "arctan"]] = None,
    ) -> np.ndarray:
        """Transforms features using previously fitted scalers into angles [-pi, pi]."""
        chosen_method = method or self.scaling_method

        if chosen_method == "pca":
            if self.standard_scaler is None or self.pca is None or self.scaler is None:
                return self.fit_transform_to_angles(features, method="pca")
            scaled_feat = self.standard_scaler.transform(features)
            reduced = self.pca.transform(scaled_feat)
            return self.scaler.transform(reduced)

        elif chosen_method == "arctan":
            if self.standard_scaler is None:
                return self.fit_transform_to_angles(features, method="arctan")
            scaled_feat = self.standard_scaler.transform(features)
            if self.pca is not None:
                scaled_feat = self.pca.transform(scaled_feat)
            angles = 2.0 * np.arctan(scaled_feat)
            return np.clip(angles, -np.pi, np.pi)

        else:
            if self.scaler is None:
                return self.fit_transform_to_angles(features, method="minmax")
            if self.pca is not None:
                features = self.pca.transform(features)
            return self.scaler.transform(features)

    @staticmethod
    def create_sliding_windows(
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates (samples, sequence_length, feature_dim) tensors for sequential architectures."""
        X, y = [], []
        n_samples = len(features)
        for i in range(n_samples - sequence_length):
            X.append(features[i : i + sequence_length])
            y.append(targets[i + sequence_length])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
