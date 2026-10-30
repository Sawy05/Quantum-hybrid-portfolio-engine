"""Synthetic and Historical Multi-Asset Market Data Generator with Regime Switching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MarketRegimeData:
    """Container for simulated or fetched multi-asset market data."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    regimes: np.ndarray  # Discrete labels: 0=BULL, 1=BEAR, 2=CRASH
    regime_names: List[str]
    tickers: List[str]
    dt: float = 1.0 / 252.0
    correlation_matrix: Optional[np.ndarray] = None


class MarketDataGenerator:
    """Generates synthetic multi-asset price dynamics under regime-switching GBM."""

    def __init__(
        self,
        n_assets: int = 4,
        n_steps: int = 1200,
        dt: float = 1.0 / 252.0,
        random_seed: int = 42,
        tickers: Optional[List[str]] = None,
    ):
        self.n_assets = n_assets
        self.n_steps = n_steps
        self.dt = dt
        self.random_seed = random_seed
        self.tickers = tickers or [f"ASSET_{i}" for i in range(n_assets)]
        self.regime_names = ["BULL", "BEAR", "CRASH"]

    def _default_regime_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns standard regime drifts, volatilities, asset correlations, and transition matrix."""
        # 3 regimes x N assets
        # BULL: positive drift, low vol
        # BEAR: negative drift, moderate vol
        # CRASH: sharp negative drift, high vol
        drifts = np.array(
            [
                [0.14, 0.16, 0.11, 0.09][: self.n_assets],  # Bull
                [-0.12, -0.15, -0.09, -0.10][: self.n_assets],  # Bear
                [-0.45, -0.52, -0.38, -0.42][: self.n_assets],  # Crash
            ]
        )
        if drifts.shape[1] < self.n_assets:
            extra = self.n_assets - drifts.shape[1]
            pad_bull = np.full((1, extra), 0.12)
            pad_bear = np.full((1, extra), -0.10)
            pad_crash = np.full((1, extra), -0.40)
            drifts = np.hstack([drifts, np.vstack([pad_bull, pad_bear, pad_crash])])

        volatilities = np.array(
            [
                [0.12, 0.14, 0.11, 0.10][: self.n_assets],
                [0.24, 0.28, 0.21, 0.23][: self.n_assets],
                [0.55, 0.65, 0.48, 0.52][: self.n_assets],
            ]
        )
        if volatilities.shape[1] < self.n_assets:
            extra = self.n_assets - volatilities.shape[1]
            pad_bull = np.full((1, extra), 0.12)
            pad_bear = np.full((1, extra), 0.25)
            pad_crash = np.full((1, extra), 0.55)
            volatilities = np.hstack([volatilities, np.vstack([pad_bull, pad_bear, pad_crash])])

        # Base cross-asset correlation matrix
        rng = np.random.default_rng(self.random_seed)
        A = rng.normal(size=(self.n_assets, self.n_assets))
        base_corr = np.dot(A, A.T)
        d = np.sqrt(np.diag(base_corr))
        base_corr = base_corr / np.outer(d, d)
        # Ensure positive-definiteness
        base_corr = 0.6 * base_corr + 0.4 * np.eye(self.n_assets)

        # Transition matrix (3x3): Bull, Bear, Crash
        transition_matrix = np.array(
            [
                [0.96, 0.03, 0.01],  # Bull persistence
                [0.05, 0.92, 0.03],  # Bear persistence
                [0.10, 0.20, 0.70],  # Crash exit
            ]
        )

        return drifts, volatilities, base_corr, transition_matrix

    def generate_synthetic(
        self,
        initial_prices: Optional[np.ndarray] = None,
        drifts: Optional[np.ndarray] = None,
        volatilities: Optional[np.ndarray] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        transition_matrix: Optional[np.ndarray] = None,
    ) -> MarketRegimeData:
        """Simulates multi-asset regime-switching geometric Brownian motion."""
        rng = np.random.default_rng(self.random_seed)

        def_drifts, def_vols, def_corr, def_trans = self._default_regime_parameters()
        mu = drifts if drifts is not None else def_drifts
        sigma = volatilities if volatilities is not None else def_vols
        corr = correlation_matrix if correlation_matrix is not None else def_corr
        P = transition_matrix if transition_matrix is not None else def_trans

        # Cholesky decomposition of cross-asset correlation
        L = np.linalg.cholesky(corr)

        # 1. Simulate Markov Chain of regimes
        regime_states = np.zeros(self.n_steps, dtype=int)
        current_state = 0  # Start in BULL
        for t in range(self.n_steps):
            regime_states[t] = current_state
            current_state = rng.choice([0, 1, 2], p=P[current_state])

        # 2. Simulate Correlated Wiener Processes and asset price paths
        S0 = initial_prices if initial_prices is not None else np.full(self.n_assets, 100.0)
        prices = np.zeros((self.n_steps, self.n_assets))
        log_returns = np.zeros((self.n_steps, self.n_assets))
        prices[0] = S0

        sqrt_dt = np.sqrt(self.dt)
        for t in range(1, self.n_steps):
            reg = regime_states[t]
            current_mu = mu[reg]
            current_sigma = sigma[reg]

            # Correlated random normal increments
            z = rng.standard_normal(self.n_assets)
            correlated_z = L @ z

            # Geometric Brownian Motion log-return: (mu - 0.5 * sigma^2) * dt + sigma * dW
            step_return = (current_mu - 0.5 * (current_sigma**2)) * self.dt + (
                current_sigma * sqrt_dt * correlated_z
            )
            log_returns[t] = step_return
            prices[t] = prices[t - 1] * np.exp(step_return)

        # Set first day returns to zero
        log_returns[0] = 0.0

        dates = pd.date_range(start="2022-01-01", periods=self.n_steps, freq="B")
        prices_df = pd.DataFrame(prices, index=dates, columns=self.tickers)
        returns_df = pd.DataFrame(log_returns, index=dates, columns=self.tickers)

        return MarketRegimeData(
            prices=prices_df,
            returns=returns_df,
            regimes=regime_states,
            regime_names=self.regime_names,
            tickers=self.tickers,
            dt=self.dt,
            correlation_matrix=corr,
        )

    def fetch_historical_or_synthetic(
        self,
        tickers: Optional[List[str]] = None,
        period: str = "3y",
    ) -> MarketRegimeData:
        """Fetches historical ticker data via yfinance with automatic synthetic fallback."""
        target_tickers = tickers or ["AAPL", "MSFT", "GOOGL", "NVDA"][: self.n_assets]
        try:
            import yfinance as yf

            data = yf.download(target_tickers, period=period, progress=False, auto_adjust=True)
            if not data.empty and "Close" in data:
                close = data["Close"].dropna()
                if len(close) >= 200:
                    returns = np.log(close / close.shift(1)).fillna(0.0)
                    # Heuristic pseudo-regimes based on rolling volatility & drawdown
                    rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)
                    mean_vol = rolling_vol.mean(axis=1)
                    q33 = mean_vol.quantile(0.33)
                    q66 = mean_vol.quantile(0.66)

                    regimes = np.zeros(len(returns), dtype=int)
                    regimes[mean_vol >= q66] = 2  # High volatility crash
                    regimes[(mean_vol >= q33) & (mean_vol < q66)] = 1  # Bear / Choppy
                    # Bull is 0 (low vol)

                    return MarketRegimeData(
                        prices=close,
                        returns=returns,
                        regimes=regimes,
                        regime_names=self.regime_names,
                        tickers=list(close.columns),
                        dt=self.dt,
                        correlation_matrix=returns.corr().to_numpy(),
                    )
        except Exception:
            pass  # Fall back to deterministic synthetic data

        # Deterministic fallback
        self.tickers = target_tickers
        return self.generate_synthetic()
