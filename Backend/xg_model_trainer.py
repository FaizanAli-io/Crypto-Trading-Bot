"""
Train XGBoost Model with All Technical Indicators + OHLC
File: train_feature_model.py
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import joblib
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from data_collector import DataCollector
from feature_engineering import FeatureEngineer
import config

from smc_features import SMCFeatureEngineer, integrate_smc_into_feature_engineer

# Initialize
smc = SMCFeatureEngineer(
    swing_lookback=5,      # Lookback for swing points
    fvg_threshold=0.001    # Minimum gap size (0.1%)
)


class FeatureModelTrainer:
    """Train XGBoost model with proper interval handling"""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.feature_names = None
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
    
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
            
        Example:
            interval="15m", horizon_minutes=60 -> shift=4 candles
            interval="1h", horizon_minutes=60 -> shift=1 candle
            interval="15m", horizon_minutes=30 -> shift=2 candles
        """
        interval_minutes = FeatureModelTrainer.interval_to_minutes(interval)
        shift = horizon_minutes / interval_minutes
        
        if shift != int(shift):
            logger.warning(
                f"Horizon {horizon_minutes}min is not evenly divisible by interval {interval} "
                f"({interval_minutes}min). Using shift={int(shift)} candles."
            )
        
        return int(shift)
    
    def prepare_training_data(
        self, 
        symbol="BTCUSDT", 
        days=365, 
        horizon_minutes=60,  # Changed from horizon to horizon_minutes
        interval='1h',
        use_smc=False  # NEW: Control SMC feature inclusion
    ):
        """
        Prepare training dataset with proper horizon handling
        
        Args:
            symbol: Trading pair to train on
            days: Historical data to use (e.g., 365 for 1 year)
            horizon_minutes: How many MINUTES ahead to predict (e.g., 60 = 1 hour)
            interval: Candle interval (e.g., "15m", "1h")
            use_smc: Whether to include Smart Money Concepts features (default: False)
        
        Returns:
            X_train, X_test, y_train, y_test, df
            
        Examples:
            # Predict 1 hour ahead using 15-min candles
            prepare_training_data(symbol="BTCUSDT", interval="15m", horizon_minutes=60)
            -> Will shift by 4 candles (4 × 15min = 60min)
            
            # Predict 30 minutes ahead using 15-min candles with SMC
            prepare_training_data(symbol="BTCUSDT", interval="15m", horizon_minutes=30, use_smc=True)
            -> Will shift by 2 candles (2 × 15min = 30min) + SMC features
            
            # Predict 6 hours ahead using 1-hour candles
            prepare_training_data(symbol="BTCUSDT", interval="1h", horizon_minutes=360)
            -> Will shift by 6 candles (6 × 1h = 6h)
        """
        logger.info(f"Preparing training data for {symbol}...")
        logger.info(f"Using {days} days of historical data")
        logger.info(f"Interval: {interval}")
        logger.info(f"Prediction horizon: {horizon_minutes} minutes")
        logger.info(f"SMC Features: {'ENABLED' if use_smc else 'DISABLED'}")
        
        # Calculate shift amount
        shift_candles = self.calculate_horizon_shift(horizon_minutes, interval)
        logger.info(f"Will shift by {shift_candles} candles for prediction")
        
        # Store for later use
        self.interval = interval
        self.horizon_minutes = horizon_minutes
        self.shift_candles = shift_candles
        self.use_smc = use_smc  # Store SMC flag
        
        # Fetch data
        df = self.data_collector.get_historical_data_by_date_range(
            symbol=symbol,
            start_date="2025-08-01",
            days=days,
            interval=interval
        )
        
        if df is None or len(df) < 200:
            raise ValueError(f"Insufficient data for {symbol}")
        
        logger.info(f"Fetched {len(df)} candles")
        
        # Add all technical indicators
        logger.info("Calculating technical indicators...")

        # Conditionally integrate SMC features
        if use_smc:
            logger.info("Integrating Smart Money Concepts features...")
            integrate_smc_into_feature_engineer(self.feature_engineer)

        df = self.feature_engineer.add_all_features(df)
        
        logger.info(f"Dataset shape after features: {df.shape}")
        
        # Create target variable with CORRECT shift
        df['future_price'] = df['close'].shift(-shift_candles)
        df['future_return'] = (df['future_price'] - df['close']) / df['close']
        
        # Binary classification: 1 if price goes up, 0 if down
        df['target'] = (df['future_price'] > df['close']).astype(int)
        
        # Remove rows with NaN target (last shift_candles rows)
        df = df[:-shift_candles].copy()
        
        logger.info(f"Removed last {shift_candles} rows (no future data)")
        
        # Handle missing values in features
        logger.info("Handling missing values...")
        initial_rows = len(df)
        
        # Forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # If still have NaNs, drop those rows
        df = df.dropna()
        
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            logger.warning(f"Dropped {rows_dropped} rows with missing values")
        
        logger.info(f"Final dataset shape: {df.shape}")
        
        # Select features for training
        exclude_cols = [
            'target', 'future_price', 'future_return',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore',
            'returns', 'log_returns',
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Ensure OHLCV is included
        ohlc_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlc_cols:
            if col not in feature_cols:
                feature_cols.append(col)
        
        # Remove highly correlated features
        logger.info("Checking for highly correlated features...")
        X_temp = df[feature_cols].copy()
        corr_matrix = X_temp.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        
        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features")
            feature_cols = [col for col in feature_cols if col not in to_drop]
        
        self.feature_names = feature_cols
        
        logger.info(f"Training with {len(feature_cols)} features")
        
        # Prepare X and y
        X = df[feature_cols].values
        y = df['target'].values
        
        # Handle infinite/NaN values
        logger.info("Handling infinite values...")
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Time series split
        split_idx = int(len(X) * 0.8)
        X_train_raw = X[:split_idx]
        X_test_raw = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        logger.info(f"Training samples: {len(X_train_raw)}")
        logger.info(f"Test samples: {len(X_test_raw)}")
        
        # Normalize features
        logger.info("Normalizing features with RobustScaler...")
        from sklearn.preprocessing import RobustScaler
        
        self.scaler = RobustScaler()
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)
        
        # Check class distribution
        logger.info(f"Class distribution (train):")
        logger.info(f"  Up (1): {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
        logger.info(f"  Down (0): {len(y_train) - y_train.sum()} ({(1-y_train.sum()/len(y_train))*100:.1f}%)")
        
        # Final validation
        assert not np.any(np.isnan(X_train)), "Training data contains NaN"
        assert not np.any(np.isinf(X_train)), "Training data contains Inf"
        
        logger.info("Data preparation complete ✓")
        
        return X_train, X_test, y_train, y_test, df
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train XGBoost classifier"""
        logger.info("Training XGBoost model...")
        
        # Calculate scale_pos_weight
        neg_samples = len(y_train) - y_train.sum()
        pos_samples = y_train.sum()
        scale_pos_weight = neg_samples / pos_samples
        
        logger.info(f"Class balance adjustment: scale_pos_weight={scale_pos_weight:.2f}")
        
        self.model = XGBClassifier(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric=['auc', 'logloss'],
            random_state=42,
            early_stopping_rounds=50,
            n_jobs=-1
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100
        )
        
        logger.info("Training complete!")
        return self.model
    
    def save_model(self, symbol="BTCUSDT"):
        """Save trained model with interval and SMC indicator in filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create descriptive filename with interval and SMC indicator
        # Example with SMC: xgb_model_smc_BTCUSDT_15m_60min_20250108_143022.pkl
        # Example without SMC: xgb_model_BTCUSDT_15m_60min_20250108_143022.pkl
        smc_prefix = "smc_" if self.use_smc else ""
        model_path = self.model_dir / f"xgb_model_{smc_prefix}{symbol}_{self.interval}_{self.horizon_minutes}min_{timestamp}.pkl"
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save feature names
        features_path = self.model_dir / f"features_{smc_prefix}{symbol}_{self.interval}_{self.horizon_minutes}min_{timestamp}.txt"
        with open(features_path, 'w') as f:
            f.write('\n'.join(self.feature_names))
        logger.info(f"Feature names saved to {features_path}")
        
        # Save scaler
        scaler_path = self.model_dir / f"scaler_{smc_prefix}{symbol}_{self.interval}_{self.horizon_minutes}min_{timestamp}.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'interval': self.interval,
            'horizon_minutes': self.horizon_minutes,
            'shift_candles': self.shift_candles,
            'num_features': len(self.feature_names),
            'use_smc': self.use_smc,  # Track SMC usage
            'trained_at': timestamp,
            'model_path': str(model_path),
            'features_path': str(features_path),
            'scaler_path': str(scaler_path)
        }
        
        import json
        metadata_path = self.model_dir / f"metadata_{smc_prefix}{symbol}_{self.interval}_{self.horizon_minutes}min_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
        
        return model_path, features_path, metadata_path
    
    # ... rest of the methods (evaluate, _plot_feature_importance, cross_validate) remain the same
    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Comprehensive model evaluation
        """
        logger.info("\n" + "="*70)
        logger.info("MODEL EVALUATION")
        logger.info("="*70)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Training metrics
        logger.info("\nTRAINING SET:")
        train_acc = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_auc = roc_auc_score(y_train, y_train_proba)
        
        logger.info(f"  Accuracy:  {train_acc:.4f}")
        logger.info(f"  Precision: {train_precision:.4f}")
        logger.info(f"  Recall:    {train_recall:.4f}")
        logger.info(f"  F1 Score:  {train_f1:.4f}")
        logger.info(f"  ROC AUC:   {train_auc:.4f}")
        
        # Test metrics
        logger.info("\nTEST SET:")
        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        logger.info(f"  Accuracy:  {test_acc:.4f}")
        logger.info(f"  Precision: {test_precision:.4f}")
        logger.info(f"  Recall:    {test_recall:.4f}")
        logger.info(f"  F1 Score:  {test_f1:.4f}")
        logger.info(f"  ROC AUC:   {test_auc:.4f}")
        
        # Confusion Matrix
        logger.info("\nCONFUSION MATRIX (Test Set):")
        cm = confusion_matrix(y_test, y_test_pred)
        logger.info(f"  True Negatives:  {cm[0,0]}")
        logger.info(f"  False Positives: {cm[0,1]}")
        logger.info(f"  False Negatives: {cm[1,0]}")
        logger.info(f"  True Positives:  {cm[1,1]}")
        
        # Classification report
        logger.info("\nCLASSIFICATION REPORT (Test Set):")
        logger.info("\n" + classification_report(y_test, y_test_pred, 
                                                  target_names=['Down', 'Up']))
        
        # Feature importance
        logger.info("\nTOP 20 MOST IMPORTANT FEATURES:")
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in importance_df.head(20).iterrows():
            logger.info(f"  {row['feature']:30s}: {row['importance']:.6f}")
        
        # Save importance plot
        self._plot_feature_importance(importance_df.head(20))
        
        logger.info("="*70 + "\n")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'confusion_matrix': cm.tolist(),
            'feature_importance': importance_df.to_dict('records')
        }
    
    def _plot_feature_importance(self, importance_df):
        """Plot and save feature importance"""
        try:
            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title('Top 20 Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            plot_path = self.model_dir / "feature_importance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {plot_path}")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not save plot: {e}")
    
    def cross_validate(self, X, y, n_splits=5):
        """
        Time series cross-validation
        """
        logger.info(f"\nPerforming {n_splits}-fold time series cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            model_cv = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1
            )
            
            model_cv.fit(X_train_cv, y_train_cv, verbose=False)
            y_pred_cv = model_cv.predict(X_val_cv)
            
            acc = accuracy_score(y_val_cv, y_pred_cv)
            scores.append(acc)
            
            logger.info(f"  Fold {fold}: Accuracy = {acc:.4f}")
        
        logger.info(f"\nCross-validation results:")
        logger.info(f"  Mean Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        
        return scores
# "BTC": "BTCUSDT",
#     "ETH": "ETHUSDT",
#     "BNB": "BNBUSDT",
#     "XRP": "XRPUSDT",
#     "ADA": "ADAUSDT",
#     "DOGE": "DOGEUSDT",
#     "SOL": "SOLUSDT",
#     "DOT": "DOTUSDT",
#     "LINK": "LINKUSDT",
#     "LTC": "LTCUSDT",

def train_currency(symbols, interval, horizon_minutes, use_smc=False):
    """
    Train models for multiple symbols
    
    Args:
        symbols: List of trading pairs to train
        interval: Candle interval (e.g., "15m", "1h")
        horizon_minutes: Prediction horizon in minutes
        use_smc: Whether to include Smart Money Concepts features (default: False)
    """
    trainer = FeatureModelTrainer()
    print(symbols)
    for symbol in symbols:

        DAYS = 365  
        logger.info(f"\nConfiguration:")
        logger.info(f"  Symbol: {symbol}")
        logger.info(f"  Interval: {interval}")
        logger.info(f"  Horizon: {horizon_minutes} minutes")
        logger.info(f"  Data period: {DAYS} days")
        logger.info(f"  SMC Features: {'ENABLED' if use_smc else 'DISABLED'}")
        

        # Step 1: Prepare data
        logger.info("\nSTEP 1: Data Preparation")
        X_train, X_test, y_train, y_test, df = trainer.prepare_training_data(
                symbol=symbol,
                days=DAYS,
                horizon_minutes=horizon_minutes,
                interval=interval,
                use_smc=use_smc
            )
        # Train model
        model = trainer.train(X_train, y_train, X_test, y_test)
        
        # Evaluate
        metrics = trainer.evaluate(X_train, y_train, X_test, y_test)
        
        # Save model
        model_path, features_path, metadata_path = trainer.save_model(symbol=symbol)
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"Model: {model_path}")
        logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        logger.info(f"Test AUC: {metrics['test_auc']:.4f}")
        logger.info(f"SMC Features: {'YES' if use_smc else 'NO'}")
        logger.info("="*70)
        
        

def main():
    """Main training pipeline with examples"""
    
    logger.info("="*70)
    logger.info("TRAINING XGBOOST MODEL WITH PROPER INTERVAL HANDLING")
    logger.info("="*70)
    
    # trainer = FeatureModelTrainer()
    
    # ==============================================================
    # EXAMPLE 1: Train on 15-minute candles, predict 1 hour ahead
    # ==============================================================
    
    # symbols = ["BTCUSDT","ETHUSDT"
    # "BNBUSDT",
    # "XRPUSDT",
    #  "ADAUSDT",
    # "DOGEUSDT",
    # "SOLUSDT",
    # "DOTUSDT",
    # "LINKUSDT",
    # "LTCUSDT" ]

    symbols = ["ETHUSDT"]
    interval = "15m"
    horizon_minutes = 15
    use_smc = True  # Set to True to train with Smart Money Concepts features
    
    train_currency(symbols, interval, horizon_minutes, use_smc=use_smc)
    # Configuration
    # for symbol in symbols:
    #     DAYS = 365  # 6 months of data
    #     HORIZON_MINUTES = 15  # Predict time ahead
    #     interval = "15m"
    #     # Step 1: Prepare data
    #     logger.info("\nSTEP 1: Data Preparation")
    #     X_train, X_test, y_train, y_test, df = trainer.prepare_training_data(
    #             symbol=symbol,
    #             days=DAYS,
    #             horizon_minutes=HORIZON_MINUTES,
    #             interval=INTERVAL
    #         )
    # # Prepare data
    
    
    # # Train model
    # model = trainer.train(X_train, y_train, X_test, y_test)
    
    # # Evaluate
    # metrics = trainer.evaluate(X_train, y_train, X_test, y_test)
    
    # # Save model
    # model_path, features_path, metadata_path = trainer.save_model(symbol=SYMBOL)
    
    # logger.info("\n" + "="*70)
    # logger.info("TRAINING COMPLETE!")
    # logger.info("="*70)
    # logger.info(f"Model: {model_path}")
    # logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    # logger.info(f"Test AUC: {metrics['test_auc']:.4f}")
    # logger.info("="*70)
    
    # # ==============================================================
    # # EXAMPLE 2: Train on 5-minute candles, predict 30 minutes ahead
    # # ==============================================================
    # # trainer2 = FeatureModelTrainer()
    # # X_train, X_test, y_train, y_test, df = trainer2.prepare_training_data(
    # #     symbol="BTCUSDT",
    # #     days=180,
    # #     horizon_minutes=30,  # 30 minutes = 6 candles of 5min
    # #     interval="5m"
    # # )
    
    # # ==============================================================
    # # EXAMPLE 3: Train on 1-hour candles, predict 6 hours ahead
    # # ==============================================================
    # # trainer3 = FeatureModelTrainer()
    # # X_train, X_test, y_train, y_test, df = trainer3.prepare_training_data(
    # #     symbol="BTCUSDT",
    # #     days=365,
    # #     horizon_minutes=360,  # 6 hours = 6 candles of 1h
    # #     interval="1h"
    # # )
    
    # return model_path, metrics


if __name__ == "__main__":
    main()