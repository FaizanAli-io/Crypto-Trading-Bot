"""
Centralized Model Loader
All model loading logic consolidated in one place for DRY principle
"""

import joblib
from pathlib import Path
from loguru import logger


class ModelLoader:
    """Centralized model loading for all prediction classes"""

    def __init__(self, model_root: str | Path | None = None):
        # Discover the latest model root (models_YYYY-MM-DD) unless explicitly provided
        if model_root:
            chosen_root = Path(model_root)
        else:
            candidates = [p for p in Path(".").glob("models_*") if p.is_dir()]
            # include legacy "models" folder as a fallback candidate
            legacy = Path("models")
            if legacy.is_dir():
                candidates.append(legacy)

            if candidates:
                chosen_root = max(candidates, key=lambda p: p.stat().st_mtime)
            else:
                chosen_root = legacy  # still default to models even if missing; callers can create later

        self.model_dir = chosen_root
        self.xg_models_dir = self.model_dir / "xg_models"
        self.features_dir = self.model_dir / "features"
        self.scalars_dir = self.model_dir / "scalars"
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_smc_model = False

        logger.info(f"ModelLoader using model root: {self.model_dir}")

    @staticmethod
    def interval_to_minutes(interval: str) -> int:
        """Convert interval string to minutes"""
        interval_map = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "8h": 480,
            "12h": 720,
            "1d": 1440,
            "3d": 4320,
            "1w": 10080,
        }
        return interval_map.get(interval, 60)

    @staticmethod
    def calculate_horizon_shift(horizon_minutes: int, interval: str) -> int:
        """
        Calculate number of candles to shift based on prediction horizon

        Args:
            horizon_minutes: How many minutes ahead to predict
            interval: Candle interval (e.g., "15m", "1h")

        Returns:
            Number of candles to shift
        """
        interval_minutes = ModelLoader.interval_to_minutes(interval)
        shift = horizon_minutes / interval_minutes

        if shift != int(shift):
            logger.warning(
                f"Horizon {horizon_minutes}min is not evenly divisible by interval {interval} "
                f"({interval_minutes}min). Using shift={int(shift)} candles."
            )

        return int(shift)

    def load_model(
        self,
        symbol="BTCUSDT",
        interval="1h",
        horizon_minutes=60,
        use_smc=None,
        silent=False,
    ):
        """
        Load trained XGBoost model for specific symbol, interval, and horizon

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "15m", "1h")
            horizon_minutes: Prediction horizon in minutes (e.g., 60, 360)
            use_smc: Whether to load SMC model (None=auto-detect, True=prefer SMC, False=prefer non-SMC)
            silent: Suppress logging output

        Returns:
            dict: {'success': bool, 'model': model_obj, 'scaler': scaler_obj, 'feature_names': list, 'is_smc': bool}
        """
        try:
            # Determine which patterns to try (in preference order)
            patterns_to_try = []

            if use_smc is True:
                # Prefer SMC, fallback to non-SMC
                patterns_to_try = [
                    f"smc_{symbol}_{interval}_{horizon_minutes}min_*.pkl",
                    f"{symbol}_{interval}_{horizon_minutes}min_*.pkl",
                ]
            elif use_smc is False:
                # Prefer non-SMC, fallback to SMC
                patterns_to_try = [
                    f"{symbol}_{interval}_{horizon_minutes}min_*.pkl",
                    f"smc_{symbol}_{interval}_{horizon_minutes}min_*.pkl",
                ]
            else:
                # Auto-detect: prefer SMC if available, otherwise use non-SMC
                patterns_to_try = [
                    f"smc_{symbol}_{interval}_{horizon_minutes}min_*.pkl",
                    f"{symbol}_{interval}_{horizon_minutes}min_*.pkl",
                ]

            # Try each pattern in order
            model_files = []
            selected_pattern = None
            for pattern in patterns_to_try:
                model_files = list(self.xg_models_dir.glob(pattern))
                if model_files:
                    selected_pattern = pattern
                    break

            if not model_files:
                if not silent:
                    logger.error(
                        f"No model found for {symbol} with interval={interval}, horizon={horizon_minutes}min"
                    )

                return {
                    "success": False,
                    "model": None,
                    "scaler": None,
                    "feature_names": None,
                    "is_smc": False,
                }

            # Get most recent model
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            if not silent:
                logger.info(f"Loading model: {latest_model.name}")

            # Detect if this is an SMC model from filename
            is_smc_model = latest_model.name.startswith("smc_")

            # Load model
            model = joblib.load(latest_model)

            # Extract timestamp from filename
            # File format: {symbol}_{interval}_{horizon_minutes}min_{timestamp}.pkl
            # or: smc_{symbol}_{interval}_{horizon_minutes}min_{timestamp}.pkl
            parts = latest_model.stem.split("_")
            timestamp = f"{parts[-2]}_{parts[-1]}"

            # Build feature file name
            smc_prefix = "smc_" if is_smc_model else ""
            features_file = (
                self.features_dir
                / f"{smc_prefix}{symbol}_{interval}_{horizon_minutes}min_{timestamp}.txt"
            )

            feature_names = None
            if features_file.exists():
                with open(features_file, "r") as f:
                    feature_names = [line.strip() for line in f.readlines()]
                if not silent:
                    logger.info(f"Loaded {len(feature_names)} feature names")
            else:
                if not silent:
                    logger.warning(f"Feature names file not found: {features_file}")
                return {
                    "success": False,
                    "model": None,
                    "scaler": None,
                    "feature_names": None,
                    "is_smc": False,
                }

            # Try to load scaler
            scaler_file = (
                self.scalars_dir
                / f"{smc_prefix}{symbol}_{interval}_{horizon_minutes}min_{timestamp}.pkl"
            )
            scaler = None
            if scaler_file.exists():
                scaler = joblib.load(scaler_file)
                if not silent:
                    logger.info("Loaded scaler")
            elif not silent:
                logger.warning("No scaler found - predictions may be inaccurate!")

            if not silent:
                logger.info("✅ Model loaded successfully!")
                logger.info(f"   Symbol: {symbol}")
                logger.info(f"   Interval: {interval}")
                logger.info(f"   Horizon: {horizon_minutes} minutes")
                logger.info(
                    f"   Features: {len(feature_names) if feature_names else 0}"
                )
                logger.info(f"   SMC Model: {'YES' if is_smc_model else 'NO'}")

            return {
                "success": True,
                "model": model,
                "scaler": scaler,
                "feature_names": feature_names,
                "is_smc": is_smc_model,
            }

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return {
                "success": False,
                "model": None,
                "scaler": None,
                "feature_names": None,
                "is_smc": False,
            }

    def load_latest_model_for_symbol(self, symbol="BTCUSDT"):
        """
        Load the most recent model for a symbol (regardless of interval/horizon)

        Args:
            symbol: Trading pair

        Returns:
            dict: Model loading result with interval and horizon_minutes info
        """
        try:
            pattern = f"{symbol}_*.pkl"
            model_files = list(self.xg_models_dir.glob(pattern))

            if not model_files:
                logger.error(f"No models found for {symbol}")
                return {
                    "success": False,
                    "model": None,
                    "scaler": None,
                    "feature_names": None,
                    "is_smc": False,
                }

            # Get most recent
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

            # Parse filename to extract interval and horizon
            parts = latest_model.stem.split("_")

            if len(parts) < 5:
                logger.error(f"Unexpected filename format: {latest_model.name}")
                return {
                    "success": False,
                    "model": None,
                    "scaler": None,
                    "feature_names": None,
                    "is_smc": False,
                }

            interval = parts[1]  # "15m"
            horizon_str = parts[2]  # "60min"
            horizon_minutes = int(horizon_str.replace("min", ""))

            logger.info(
                f"Auto-detected: interval={interval}, horizon={horizon_minutes}min"
            )

            result = self.load_model(
                symbol=symbol, interval=interval, horizon_minutes=horizon_minutes
            )
            result["interval"] = interval
            result["horizon_minutes"] = horizon_minutes

            return result

        except Exception as e:
            logger.error(f"Error loading latest model: {e}")
            return {
                "success": False,
                "model": None,
                "scaler": None,
                "feature_names": None,
                "is_smc": False,
            }

    def get_all_available_models(self):
        """
        Parse all model files and extract symbol, interval, and horizon_minutes

        Returns:
            list: List of dicts with model info
        """
        model_files = list(self.xg_models_dir.glob("*.pkl"))

        models_info = []
        for model_file in model_files:
            try:
                # Parse filename: {symbol}_{interval}_{horizon_minutes}min_{timestamp}.pkl
                # or: smc_{symbol}_{interval}_{horizon_minutes}min_{timestamp}.pkl
                stem = model_file.stem

                # Handle SMC prefix
                if stem.startswith("smc_"):
                    stem = stem[4:]  # Remove "smc_" prefix

                parts = stem.split("_")

                if len(parts) < 4:
                    logger.warning(
                        f"Skipping unexpected filename format: {model_file.name}"
                    )
                    continue

                symbol = parts[0]  # "BTCUSDT"
                interval = parts[1]  # "15m"
                horizon_str = parts[2]  # "15min"
                horizon_minutes = int(horizon_str.replace("min", ""))

                models_info.append(
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "horizon_minutes": horizon_minutes,
                        "model_file": model_file.name,
                    }
                )

            except Exception as e:
                logger.warning(f"Error parsing model file {model_file.name}: {e}")
                continue

        logger.info(f"Found {len(models_info)} models")
        return models_info
