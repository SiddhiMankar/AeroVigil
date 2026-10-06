"""
AeroVigil - C-MAPSS FD001 RUL Prediction Model
ingestion/cmapss_rul_model.py

Trains a Gradient Boosting Regressor on FD001 windowed sensor features
and evaluates predictions against RUL_FD001.txt ground truth.

ML Design Principles (Anti-Leakage):
    1. Train/validation split: by ENGINE ID (80/20 engines), never by rows.
    2. Scaler fitted on training engines only; applied to val/test.
    3. RUL ground truth from RUL_FD001.txt is NEVER seen during training.
    4. Test engine predictions use last-window features only (no future info).
    5. Evaluation uses standard MAE, RMSE, R² plus NASA PHM asymmetric score.

NASA Asymmetric Score:
    Penalizes early predictions (predicting failure too late) more than
    late predictions (predicting failure too early), reflecting operational
    asymmetry: missing an imminent failure is worse than a false alarm.
    Score = sum( exp(-e/13) - 1 for e < 0 ) + sum( exp(e/10) - 1 for e >= 0 )
    where e = predicted_RUL - true_RUL.
    Lower score = better. This score supplements standard metrics; it does NOT
    replace MAE/RMSE.

Domain Disclaimer:
    This model is trained on C-MAPSS FD001 turbofan data. Results validate
    AeroVigil's generic PHM/prognostics capability. They do NOT directly
    apply to aero-piston engine RUL under real UAV operational conditions.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional sklearn import — provide clear error if missing
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn is not installed. CMAPSSRULModel will not be functional. "
        "Install it with: pip install scikit-learn"
    )

# ---------------------------------------------------------------------------
# Default model hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_HYPERPARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_samples_leaf": 5,
    "random_state": 42,
}

# Fraction of training engines used for training (remainder = validation)
TRAIN_ENGINE_FRACTION: float = 0.80
WINDOW_SIZE: int = 15


class CMAPSSRULModel:
    """
    Gradient Boosting RUL Predictor for NASA C-MAPSS FD001.

    Trains on windowed sensor statistics from training engine trajectories
    and predicts cycle-level RUL for test engines.

    Usage
    -----
    >>> from ingestion.cmapss_loader import load_fd001
    >>> from ingestion.cmapss_preprocessor import CMAPSSPreprocessor
    >>> from ingestion.cmapss_rul_model import CMAPSSRULModel
    >>>
    >>> train_df, test_df, rul_gt = load_fd001()
    >>> preprocessor = CMAPSSPreprocessor()
    >>> train_tf = preprocessor.fit_transform(train_df)
    >>> test_tf  = preprocessor.transform(test_df)
    >>>
    >>> model = CMAPSSRULModel()
    >>> results = model.train_and_evaluate(train_tf, test_tf, rul_gt, preprocessor)
    >>> print(results["metrics"])
    """

    def __init__(self, hyperparams: Optional[Dict] = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for CMAPSSRULModel. "
                "Install with: pip install scikit-learn"
            )
        params = hyperparams or DEFAULT_HYPERPARAMS
        self.model = GradientBoostingRegressor(**params)
        self.is_trained: bool = False
        self.train_engine_ids: Optional[np.ndarray] = None
        self.val_engine_ids: Optional[np.ndarray] = None
        self.feature_names: Optional[list] = None

    def _split_engines(
        self, all_engine_ids: np.ndarray, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits engine IDs into training and validation sets.
        Split is by ENGINE ID to prevent data leakage between engines.
        """
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(all_engine_ids)
        n_train = int(len(shuffled) * TRAIN_ENGINE_FRACTION)
        return shuffled[:n_train], shuffled[n_train:]

    def train(
        self,
        train_transformed_df: pd.DataFrame,
        preprocessor,
        window_size: int = WINDOW_SIZE,
        seed: int = 42,
    ) -> Dict:
        """
        Trains the GBR model on windowed features from training engine data.
        Holds out 20% of training engines for validation.

        Parameters
        ----------
        train_transformed_df : pd.DataFrame
            Output of CMAPSSPreprocessor.transform() on training data.
            Must contain 'rul' column (from CMAPSSLoader).
        preprocessor : CMAPSSPreprocessor
            Fitted preprocessor (needed for feature building).
        window_size : int
            Sliding window size in cycles.
        seed : int
            Random seed for engine split.

        Returns
        -------
        dict with keys: train_mae, val_mae, train_rmse, val_rmse,
                        n_train_engines, n_val_engines, n_train_samples
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available.")

        all_engine_ids = train_transformed_df["unit_number"].unique()
        self.train_engine_ids, self.val_engine_ids = self._split_engines(
            all_engine_ids, seed=seed
        )

        logger.info(
            "Engine split: %d training engines, %d validation engines.",
            len(self.train_engine_ids), len(self.val_engine_ids),
        )

        # Build features from training engines only
        train_eng_df = train_transformed_df[
            train_transformed_df["unit_number"].isin(self.train_engine_ids)
        ]
        val_eng_df = train_transformed_df[
            train_transformed_df["unit_number"].isin(self.val_engine_ids)
        ]

        X_train, y_train, _ = preprocessor.build_windowed_features(
            train_eng_df, window_size=window_size
        )
        X_val, y_val, _ = preprocessor.build_windowed_features(
            val_eng_df, window_size=window_size
        )

        # Build feature names for interpretability
        n_sensors = len(preprocessor.selected_sensors)
        self.feature_names = (
            [f"{s}_mean" for s in preprocessor.selected_sensors]
            + [f"{s}_std" for s in preprocessor.selected_sensors]
        )

        logger.info(
            "Training GBR on %d samples (%d features). Val set: %d samples.",
            len(X_train), X_train.shape[1], len(X_val),
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Training metrics
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)

        train_mae = float(mean_absolute_error(y_train, y_train_pred))
        val_mae = float(mean_absolute_error(y_val, y_val_pred))
        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))

        logger.info(
            "Training: MAE=%.2f, RMSE=%.2f | Validation: MAE=%.2f, RMSE=%.2f",
            train_mae, train_rmse, val_mae, val_rmse,
        )

        return {
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "n_train_engines": len(self.train_engine_ids),
            "n_val_engines": len(self.val_engine_ids),
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
        }

    def predict_test(
        self,
        test_transformed_df: pd.DataFrame,
        preprocessor,
        window_size: int = WINDOW_SIZE,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts RUL for all test engines using last-window features.

        Parameters
        ----------
        test_transformed_df : pd.DataFrame
            Transformed test data.
        preprocessor : CMAPSSPreprocessor
            Must be the SAME fitted preprocessor used during training.
        window_size : int
            Must match training window size.

        Returns
        -------
        predictions : np.ndarray, shape (n_test_engines,)
        unit_ids : np.ndarray, shape (n_test_engines,)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X_test, unit_ids = preprocessor.build_last_window_features(
            test_transformed_df, window_size=window_size
        )

        predictions = self.model.predict(X_test)
        # Clip to non-negative (RUL cannot be negative)
        predictions = np.maximum(predictions, 0.0)

        logger.info(
            "Predicted RUL for %d test engines. "
            "Pred min=%.1f, max=%.1f, mean=%.1f",
            len(predictions), predictions.min(), predictions.max(), predictions.mean()
        )
        return predictions, unit_ids

    def evaluate(
        self,
        predictions: np.ndarray,
        unit_ids: np.ndarray,
        rul_gt: np.ndarray,
    ) -> Dict:
        """
        Evaluates test-engine predictions against RUL_FD001.txt ground truth.

        Parameters
        ----------
        predictions : np.ndarray, shape (n_engines,)
            Predicted RUL values (in cycles) per test engine.
        unit_ids : np.ndarray, shape (n_engines,)
            Engine IDs corresponding to predictions (1-indexed).
        rul_gt : np.ndarray, shape (100,)
            Ground-truth RUL values loaded from RUL_FD001.txt.
            Index 0 = engine 1, index 99 = engine 100.

        Returns
        -------
        dict with: mae, rmse, r2, nasa_score,
                   per_engine (DataFrame: unit, pred_rul, true_rul, error)
        """
        # Align ground truth to prediction engine order
        true_rul = rul_gt[unit_ids - 1]  # unit_ids are 1-indexed

        errors = predictions - true_rul  # e > 0 = predicted too high (late alarm)

        mae = float(mean_absolute_error(true_rul, predictions))
        rmse = float(np.sqrt(mean_squared_error(true_rul, predictions)))
        r2 = float(r2_score(true_rul, predictions))
        nasa_score = float(self._nasa_score(errors))

        per_engine = pd.DataFrame({
            "unit_number": unit_ids,
            "pred_rul": predictions,
            "true_rul": true_rul,
            "error": errors,
            "abs_error": np.abs(errors),
        }).sort_values("unit_number").reset_index(drop=True)

        logger.info(
            "Test Evaluation Results — MAE: %.2f | RMSE: %.2f | R²: %.4f | "
            "NASA Score: %.2f",
            mae, rmse, r2, nasa_score,
        )

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "nasa_score": nasa_score,
            "per_engine": per_engine,
        }

    def train_and_evaluate(
        self,
        train_transformed_df: pd.DataFrame,
        test_transformed_df: pd.DataFrame,
        rul_gt: np.ndarray,
        preprocessor,
        window_size: int = WINDOW_SIZE,
        seed: int = 42,
    ) -> Dict:
        """
        Convenience method: trains, predicts on test, and evaluates in one call.

        Returns
        -------
        dict with keys: training_info, metrics, per_engine_results, predictions, unit_ids
        """
        training_info = self.train(
            train_transformed_df, preprocessor, window_size=window_size, seed=seed
        )
        predictions, unit_ids = self.predict_test(
            test_transformed_df, preprocessor, window_size=window_size
        )
        eval_results = self.evaluate(predictions, unit_ids, rul_gt)

        return {
            "training_info": training_info,
            "metrics": {
                "test_mae": eval_results["mae"],
                "test_rmse": eval_results["rmse"],
                "test_r2": eval_results["r2"],
                "test_nasa_score": eval_results["nasa_score"],
                "val_mae": training_info["val_mae"],
                "val_rmse": training_info["val_rmse"],
            },
            "per_engine_results": eval_results["per_engine"],
            "predictions": predictions,
            "unit_ids": unit_ids,
        }

    @staticmethod
    def _nasa_score(errors: np.ndarray) -> float:
        """
        Computes the NASA/PHM asymmetric RUL score.

        Penalizes late predictions (e >= 0, predicting failure later than actual)
        more than early predictions (e < 0, over-conservative).

        Score = sum( exp(-e/13) - 1 for e < 0 )
              + sum( exp( e/10) - 1 for e >= 0 )

        Lower is better. Supplementary to standard metrics; not a replacement.

        Reference: Saxena et al., PHM08.
        """
        score = 0.0
        for e in errors:
            if e < 0:
                score += np.exp(-e / 13.0) - 1.0
            else:
                score += np.exp(e / 10.0) - 1.0
        return score

    def get_feature_importances(self) -> Optional[pd.DataFrame]:
        """
        Returns feature importance from the trained GBR model.
        Only available after training.
        """
        if not self.is_trained or self.feature_names is None:
            return None
        importances = self.model.feature_importances_
        return (
            pd.DataFrame({
                "feature": self.feature_names,
                "importance": importances,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
