"""
Simplified Model Loader
Compatible with SignalPredictor and LiveMonitor
"""

import joblib
from pathlib import Path


class ModelLoader:
    def __init__(self):
        self.model_dir = self._detect_model_root()
        self.xg_models_dir = self.model_dir / "xg_models"
        self.features_dir = self.model_dir / "features"
        self.scalers_dir = self.model_dir / "scalars"

    # =========================
    # Utilities
    # =========================

    @staticmethod
    def interval_to_minutes(interval: str) -> int:
        return {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
        }.get(interval, 60)

    @staticmethod
    def calculate_horizon_shift(horizon_minutes: int, interval: str) -> int:
        return int(horizon_minutes / ModelLoader.interval_to_minutes(interval))

    def _detect_model_root(self):
        base_path = Path("../Crypto-Agent")
        legacy_path = base_path / "models"

        candidates = [p for p in base_path.glob("models_*") if p.is_dir()]

        if legacy_path.is_dir():
            candidates.append(legacy_path)

        if not candidates:
            raise FileNotFoundError(
                "No model directories found (models_* or legacy models/)"
            )

        # Pick most recently modified directory
        return max(candidates, key=lambda p: p.stat().st_mtime)

    # =========================
    # Model loading
    # =========================

    def load_model(self, symbol, interval, horizon_minutes, silent=True):
        """
        Loads the most recent model for symbol / interval / horizon
        """
        pattern = f"{symbol}_{interval}_{horizon_minutes}min_*.pkl"
        model_files = list(self.xg_models_dir.glob(pattern))

        if not model_files:
            return {
                "success": False,
                "model": None,
                "scaler": None,
                "feature_names": None,
            }

        model_file = max(model_files, key=lambda p: p.stat().st_mtime)
        model = joblib.load(model_file)

        timestamp = "_".join(model_file.stem.split("_")[-2:])

        feature_file = (
            self.features_dir
            / f"{symbol}_{interval}_{horizon_minutes}min_{timestamp}.txt"
        )
        scaler_file = (
            self.scalers_dir
            / f"{symbol}_{interval}_{horizon_minutes}min_{timestamp}.pkl"
        )

        if not feature_file.exists():
            return {
                "success": False,
                "model": None,
                "scaler": None,
                "feature_names": None,
            }

        with open(feature_file) as f:
            feature_names = [line.strip() for line in f.readlines()]

        scaler = joblib.load(scaler_file) if scaler_file.exists() else None

        return {
            "success": True,
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
        }

    # =========================
    # Discovery (used by LiveMonitor)
    # =========================

    def list_available_models(self):
        """
        Returns list of available models with symbol / interval / horizon
        """
        models = []

        for file in self.xg_models_dir.glob("*.pkl"):
            name = file.stem

            # Skip SMC models entirely (simplified system)
            if name.startswith("smc_"):
                continue

            parts = name.split("_")
            if len(parts) < 4:
                continue

            try:
                symbol = parts[0]
                interval = parts[1]
                horizon_minutes = int(parts[2].replace("min", ""))

                models.append(
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "horizon_minutes": horizon_minutes,
                    }
                )
            except:
                continue

        return models
