"""End-to-end pipeline: price + features -> router -> structure -> signals."""

from __future__ import annotations

import pandas as pd

from .config import Config
from .data import load_price, load_regime_features
from .regime import build_router, features_from_chain
from .strategy import build_signals
from .structure import add_structure


def build_dataset(cfg: Config, price: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Compose the full signal frame from aligned price bars and regime features."""
    df = build_router(price, features, cfg.regime)
    df = add_structure(df, cfg.structure)
    df = build_signals(df, cfg.backtest)
    return df


def load_dataset(cfg: Config) -> pd.DataFrame:
    """Load everything the config points at and build the signal frame."""
    if not cfg.data.price_csv:
        raise ValueError("config.data.price_csv is required")
    price = load_price(cfg.data.price_csv, cfg.data)

    if cfg.data.regime_csv:
        features = load_regime_features(cfg.data.regime_csv, cfg.data)
    elif cfg.data.chain_csv:
        chain = pd.read_csv(cfg.data.chain_csv)
        features = features_from_chain(chain, cfg.data)
        if features.index.tz is None:
            features.index = features.index.tz_localize(cfg.data.timezone)
    else:
        raise ValueError("provide either data.regime_csv or data.chain_csv")

    return build_dataset(cfg, price, features)
