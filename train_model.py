#!/usr/bin/env python3
"""
FOOT DROP ORTHOSIS - MODEL TRAINING SCRIPT
==========================================

This script trains the ML model used by the orthosis controller.
It's designed to work with data collected from BOTH:
  - Myomatrix intramuscular EMG (primary training data)
  - HD-sEMG (concurrent validation data, if available)

The trained model uses features that transfer between modalities,
allowing training on high-fidelity iEMG data while deploying with
non-invasive surface electrodes.

USAGE:
  python train_model.py --data training_data.csv --output trained_model.pkl

DATA FORMAT:
  CSV file with columns:
    timestamp, ta_rms, ta_mav, ta_wl, ta_zc, ta_ssc, mg_rms, target_activation
  
  Where target_activation is the ground truth (0.0 to 1.0) from:
    - Force sensor measurements during calibration trials
    - Manual labeling of intended movement
    - Kinematic data from motion capture

Author: Chioke Swann
Date: 3 Feb 2026
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline


# =============================================================================
# CONFIGURATION
# =============================================================================

# Features used for training (must match controller extraction)
FEATURE_COLUMNS = [
    'ta_rms',      # Root mean square of TA EMG
    'ta_mav',      # Mean absolute value
    'ta_wl',       # Waveform length  
    'ta_zc',       # Zero crossings
    'ta_ssc',      # Slope sign changes
    'mg_rms',      # Gastrocnemius RMS
    'ta_mg_ratio'  # TA/MG activation ratio (key transfer feature)
]

TARGET_COLUMN = 'target_activation'

# Model options
MODEL_TYPES = {
    'random_forest': RandomForestRegressor,
    'gradient_boosting': GradientBoostingRegressor,
    'ridge': Ridge,
    'elastic_net': ElasticNet
}

DEFAULT_MODEL = 'random_forest'


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_training_data(data_path: str) -> pd.DataFrame:
    """
    Load and validate training data from CSV.
    
    Args:
        data_path: Path to CSV file
    
    Returns:
        DataFrame with features and target
    """
    logger = logging.getLogger("DataLoader")
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Validate columns
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        # Try to compute ta_mg_ratio if missing
        if 'ta_mg_ratio' in missing_features and 'ta_rms' in df.columns and 'mg_rms' in df.columns:
            df['ta_mg_ratio'] = df['ta_rms'] / (df['mg_rms'] + 1e-6)
            missing_features.remove('ta_mg_ratio')
    
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    
    # Remove rows with NaN
    initial_len = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with NaN values")
    
    # Validate target range
    if df[TARGET_COLUMN].min() < 0 or df[TARGET_COLUMN].max() > 1:
        logger.warning("Target values outside [0, 1] range - clipping")
        df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(0, 1)
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Target distribution: min={df[TARGET_COLUMN].min():.3f}, "
                f"max={df[TARGET_COLUMN].max():.3f}, "
                f"mean={df[TARGET_COLUMN].mean():.3f}")
    
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Extract feature matrix and target vector.
    
    Returns:
        (X, y) tuple of numpy arrays
    """
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    return X, y


# =============================================================================
# MODEL TRAINING
# =============================================================================

def get_model_params(model_type: str) -> dict:
    """
    Get hyperparameter grid for model type.
    """
    if model_type == 'random_forest':
        return {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [5, 10, 15, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'gradient_boosting':
        return {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'ridge':
        return {
            'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    elif model_type == 'elastic_net':
        return {
            'model__alpha': [0.01, 0.1, 1.0],
            'model__l1_ratio': [0.1, 0.5, 0.9]
        }
    else:
        return {}


def train_model(X: np.ndarray, y: np.ndarray, 
                model_type: str = DEFAULT_MODEL,
                do_grid_search: bool = True) -> tuple:
    """
    Train activation prediction model.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        model_type: Type of model to train
        do_grid_search: Whether to perform hyperparameter search
    
    Returns:
        (model, scaler, metrics) tuple
    """
    logger = logging.getLogger("Training")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Create pipeline with scaler
    scaler = StandardScaler()
    model_class = MODEL_TYPES[model_type]
    
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model_class(random_state=42))
    ])
    
    # Hyperparameter search
    if do_grid_search:
        logger.info("Performing hyperparameter search...")
        param_grid = get_model_params(model_type)
        
        if param_grid:
            grid_search = GridSearchCV(
                pipeline, param_grid, 
                cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            pipeline = grid_search.best_estimator_
        else:
            pipeline.fit(X_train, y_train)
    else:
        pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    metrics = {
        'train_mse': mean_squared_error(y_train, y_pred_train),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_r2': r2_score(y_test, y_pred_test)
    }
    
    logger.info("=== Training Results ===")
    logger.info(f"Train MSE: {metrics['train_mse']:.6f}")
    logger.info(f"Train MAE: {metrics['train_mae']:.6f}")
    logger.info(f"Train R²:  {metrics['train_r2']:.4f}")
    logger.info(f"Test MSE:  {metrics['test_mse']:.6f}")
    logger.info(f"Test MAE:  {metrics['test_mae']:.6f}")
    logger.info(f"Test R²:   {metrics['test_r2']:.4f}")
    
    # Cross-validation for robustness estimate
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    metrics['cv_mse_mean'] = -cv_scores.mean()
    metrics['cv_mse_std'] = cv_scores.std()
    logger.info(f"CV MSE: {metrics['cv_mse_mean']:.6f} (+/- {metrics['cv_mse_std']:.6f})")
    
    # Extract trained components
    trained_scaler = pipeline.named_steps['scaler']
    trained_model = pipeline.named_steps['model']
    
    return trained_model, trained_scaler, metrics


def analyze_feature_importance(model, feature_names: list) -> dict:
    """
    Analyze feature importance if model supports it.
    """
    logger = logging.getLogger("Analysis")
    importance = {}
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for name, imp in zip(feature_names, importances):
            importance[name] = imp
        
        logger.info("=== Feature Importance ===")
        for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
            logger.info(f"  {name}: {imp:.4f}")
    
    elif hasattr(model, 'coef_'):
        coefs = model.coef_
        for name, coef in zip(feature_names, coefs):
            importance[name] = abs(coef)
        
        logger.info("=== Feature Coefficients (absolute) ===")
        for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
            logger.info(f"  {name}: {imp:.4f}")
    
    return importance


# =============================================================================
# MODEL SAVING
# =============================================================================

def save_model(model, scaler, metrics: dict, feature_names: list,
               output_path: str, model_type: str):
    """
    Save trained model to pickle file.
    
    Format compatible with orthosis controller's ActivationModel.load()
    """
    logger = logging.getLogger("Save")
    
    save_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metadata': {
            'model_type': model_type,
            'metrics': metrics,
            'training_date': pd.Timestamp.now().isoformat()
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    logger.info(f"Model saved to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train activation prediction model for foot drop orthosis'
    )
    parser.add_argument(
        '--data', '-d', required=True,
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--output', '-o', default='trained_model.pkl',
        help='Output path for trained model'
    )
    parser.add_argument(
        '--model-type', '-m', default=DEFAULT_MODEL,
        choices=list(MODEL_TYPES.keys()),
        help='Type of model to train'
    )
    parser.add_argument(
        '--no-grid-search', action='store_true',
        help='Skip hyperparameter search (faster but may be suboptimal)'
    )
    parser.add_argument(
        '--log-level', '-l', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR']
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    logger = logging.getLogger("Main")
    
    # Load data
    df = load_training_data(args.data)
    X, y = prepare_features(df)
    
    # Train
    model, scaler, metrics = train_model(
        X, y, 
        model_type=args.model_type,
        do_grid_search=not args.no_grid_search
    )
    
    # Analyze
    importance = analyze_feature_importance(model, FEATURE_COLUMNS)
    metrics['feature_importance'] = importance
    
    # Save
    save_model(model, scaler, metrics, FEATURE_COLUMNS, 
               args.output, args.model_type)
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
