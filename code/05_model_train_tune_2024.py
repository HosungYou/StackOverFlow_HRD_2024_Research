#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stack Overflow Developer Survey 2024 - Model Training and Tuning
Human Resource Development (HRD) Focus

This script trains and tunes models to predict developer market value:
- Classification models for high wage prediction (is_high_wage)
- Regression models for salary prediction (salary_log)

Models used:
- Random Forest
- XGBoost
- ElasticNet (baseline for regression)
- Logistic Regression (baseline for classification)

Hyperparameter optimization using Optuna.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import time
import joblib
import json
import optuna
from optuna.integration import XGBoostPruningCallback

# ML libraries
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb

# Add the project root to the path so we can import the config
project_root = Path("/Volumes/External SSD/Pycharm/StackOverFlow_HRD_2024")
sys.path.append(str(project_root))

# Import config using importlib to handle module name with digit prefix
import importlib.util
config_path = project_root / "src" / "00_config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Access config variables
PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
MODELS_DIR = config.MODELS_DIR
logger = config.logger

# Make sure the models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_training_data(year=2024):
    """
    Load the training data created in the feature engineering step.
    
    Parameters:
    -----------
    year : int
        Survey year
        
    Returns:
    --------
    tuple
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test
    """
    # Load the training and test data
    train_path = PROCESSED_DATA_DIR / f"features_{year}_train.parquet"
    test_path = PROCESSED_DATA_DIR / f"features_{year}_test.parquet"
    
    if not train_path.exists() or not test_path.exists():
        logger.error(f"Training/test data not found. Please run feature engineering first.")
        return None, None, None, None, None, None
    
    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_parquet(train_path)
    
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_parquet(test_path)
    
    logger.info(f"Loaded {train_df.shape[0]} training samples and {test_df.shape[0]} test samples")
    
    # Get the outcome variables
    class_target = 'is_high_wage'
    reg_target = 'salary_log'
    
    # Check if both targets exist in the data
    if class_target not in train_df.columns:
        logger.error(f"Classification target '{class_target}' not found in the data")
        return None, None, None, None, None, None
    
    if reg_target not in train_df.columns:
        logger.error(f"Regression target '{reg_target}' not found in the data")
        return None, None, None, None, None, None
    
    # Get the feature columns
    id_cols = ['ResponseId', config.COLUMN_NAMES['salary']]
    outcome_cols = [class_target, reg_target]
    feature_cols = [col for col in train_df.columns if col not in id_cols and col not in outcome_cols]
    
    logger.info(f"Using {len(feature_cols)} features for modeling")
    
    # Create the train/test feature matrices and target vectors
    X_train = train_df[feature_cols]
    y_class_train = train_df[class_target]
    y_reg_train = train_df[reg_target]
    
    X_test = test_df[feature_cols]
    y_class_test = test_df[class_target]
    y_reg_test = test_df[reg_target]
    
    return X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test

def preprocess_features(X_train, X_test):
    """
    Preprocess the features for modeling:
    - Handle categorical variables
    - Scale numerical features
    - Handle missing values
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
        
    Returns:
    --------
    tuple
        X_train_proc, X_test_proc, feature_names
    """
    logger.info("Preprocessing features")
    
    # Make copies to avoid modifying the originals
    X_train_proc = X_train.copy()
    X_test_proc = X_test.copy()
    
    # Find and drop non-numeric columns
    non_numeric_cols = X_train_proc.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        logger.warning(f"Dropping {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols}")
        X_train_proc = X_train_proc.drop(columns=non_numeric_cols)
        X_test_proc = X_test_proc.drop(columns=non_numeric_cols)
    
    # Find and drop columns with missing values
    missing_cols = X_train_proc.columns[X_train_proc.isna().any()].tolist()
    if missing_cols:
        logger.warning(f"Dropping {len(missing_cols)} columns with missing values: {missing_cols}")
        X_train_proc = X_train_proc.drop(columns=missing_cols)
        X_test_proc = X_test_proc.drop(columns=missing_cols)
    
    # Scale numeric features
    scaler = StandardScaler()
    X_train_proc = pd.DataFrame(
        scaler.fit_transform(X_train_proc),
        columns=X_train_proc.columns,
        index=X_train_proc.index
    )
    
    X_test_proc = pd.DataFrame(
        scaler.transform(X_test_proc),
        columns=X_test_proc.columns,
        index=X_test_proc.index
    )
    
    # Save the scaler for later use
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.joblib")
    
    logger.info(f"Preprocessing complete. Final feature set has {X_train_proc.shape[1]} columns")
    
    return X_train_proc, X_test_proc, X_train_proc.columns.tolist()

def train_baseline_models(X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test):
    """
    Train baseline models for classification and regression.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Feature matrices
    y_class_train, y_class_test : pd.Series
        Binary classification targets
    y_reg_train, y_reg_test : pd.Series
        Regression targets
        
    Returns:
    --------
    dict
        Dictionary of baseline model results
    """
    logger.info("Training baseline models")
    
    baseline_results = {}
    
    # Baseline Classification Model: Logistic Regression
    logger.info("Training Logistic Regression baseline model")
    
    lr_model = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=config.RANDOM_STATE
    )
    
    lr_model.fit(X_train, y_class_train)
    
    # Evaluate classification model
    lr_train_pred = lr_model.predict(X_train)
    lr_test_pred = lr_model.predict(X_test)
    
    lr_train_prob = lr_model.predict_proba(X_train)[:, 1]
    lr_test_prob = lr_model.predict_proba(X_test)[:, 1]
    
    baseline_results['logistic_regression'] = {
        'train_accuracy': accuracy_score(y_class_train, lr_train_pred),
        'test_accuracy': accuracy_score(y_class_test, lr_test_pred),
        'train_precision': precision_score(y_class_train, lr_train_pred),
        'test_precision': precision_score(y_class_test, lr_test_pred),
        'train_recall': recall_score(y_class_train, lr_train_pred),
        'test_recall': recall_score(y_class_test, lr_test_pred),
        'train_f1': f1_score(y_class_train, lr_train_pred),
        'test_f1': f1_score(y_class_test, lr_test_pred),
        'train_auc': roc_auc_score(y_class_train, lr_train_prob),
        'test_auc': roc_auc_score(y_class_test, lr_test_prob)
    }
    
    logger.info(f"Logistic Regression baseline results: "
                f"AUC = {baseline_results['logistic_regression']['test_auc']:.4f}, "
                f"F1 = {baseline_results['logistic_regression']['test_f1']:.4f}")
    
    # Save the model
    joblib.dump(lr_model, MODELS_DIR / "baseline_classifier.joblib")
    
    # Baseline Regression Model: ElasticNet
    logger.info("Training ElasticNet baseline model")
    
    en_model = ElasticNet(
        alpha=1.0,
        l1_ratio=0.5,
        max_iter=1000,
        random_state=config.RANDOM_STATE
    )
    
    en_model.fit(X_train, y_reg_train)
    
    # Evaluate regression model
    en_train_pred = en_model.predict(X_train)
    en_test_pred = en_model.predict(X_test)
    
    baseline_results['elasticnet'] = {
        'train_mae': mean_absolute_error(y_reg_train, en_train_pred),
        'test_mae': mean_absolute_error(y_reg_test, en_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_reg_train, en_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_reg_test, en_test_pred)),
        'train_r2': r2_score(y_reg_train, en_train_pred),
        'test_r2': r2_score(y_reg_test, en_test_pred)
    }
    
    logger.info(f"ElasticNet baseline results: "
                f"RMSE = {baseline_results['elasticnet']['test_rmse']:.4f}, "
                f"R² = {baseline_results['elasticnet']['test_r2']:.4f}")
    
    # Save the model
    joblib.dump(en_model, MODELS_DIR / "baseline_regressor.joblib")
    
    return baseline_results

def optimize_xgb_classifier(X_train, y_train):
    """
    Optimize XGBoost classifier using Optuna.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Binary classification target
        
    Returns:
    --------
    dict
        Best hyperparameters
    """
    logger.info("Optimizing XGBoost classifier")
    
    def objective(trial):
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'booster': trial.suggest_categorical('booster', ['gbtree']),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': config.RANDOM_STATE
        }
        
        cv = KFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        
        pruning_callback = XGBoostPruningCallback(trial, "test-auc")
        
        cv_scores = []
        for train_idx, valid_idx in cv.split(X_train, y_train):
            X_cv_train, X_cv_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_cv_train, y_cv_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]
            
            dtrain = xgb.DMatrix(X_cv_train, label=y_cv_train)
            dvalid = xgb.DMatrix(X_cv_valid, label=y_cv_valid)
            
            # evals_result is used to store evaluation results
            evals_result = {}
            
            bst = xgb.train(
                param,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, 'train'), (dvalid, 'test')],
                evals_result=evals_result,
                early_stopping_rounds=10,
                verbose_eval=False,
                callbacks=[pruning_callback]
            )
            
            # Get the best iteration
            best_score = evals_result['test']['auc'][bst.best_iteration - 1]
            cv_scores.append(best_score)
        
        # Return the mean AUC score
        return np.mean(cv_scores)
    
    # Create the study
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    
    # Run the optimization
    study.optimize(objective, n_trials=config.N_TRIALS, timeout=config.TIMEOUT)
    
    # Get the best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    logger.info(f"Best XGBoost classifier parameters: {best_params}")
    logger.info(f"Best XGBoost classifier AUC: {best_score:.4f}")
    
    # Return the best parameters
    return best_params

def optimize_xgb_regressor(X_train, y_train):
    """
    Optimize XGBoost regressor using Optuna.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Regression target
        
    Returns:
    --------
    dict
        Best hyperparameters
    """
    logger.info("Optimizing XGBoost regressor")
    
    def objective(trial):
        param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': trial.suggest_categorical('booster', ['gbtree']),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': config.RANDOM_STATE
        }
        
        cv = KFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        
        cv_scores = []
        for train_idx, valid_idx in cv.split(X_train, y_train):
            X_cv_train, X_cv_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_cv_train, y_cv_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]
            
            dtrain = xgb.DMatrix(X_cv_train, label=y_cv_train)
            dvalid = xgb.DMatrix(X_cv_valid, label=y_cv_valid)
            
            # evals_result is used to store evaluation results
            evals_result = {}
            
            bst = xgb.train(
                param,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, 'train'), (dvalid, 'test')],
                evals_result=evals_result,
                early_stopping_rounds=10,
                verbose_eval=False
            )
            
            # Predict on validation set
            preds = bst.predict(dvalid)
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_cv_valid, preds))
            cv_scores.append(rmse)
        
        # Return the mean RMSE score (lower is better)
        return np.mean(cv_scores)
    
    # Create the study
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    
    # Run the optimization
    study.optimize(objective, n_trials=config.N_TRIALS, timeout=config.TIMEOUT)
    
    # Get the best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    logger.info(f"Best XGBoost regressor parameters: {best_params}")
    logger.info(f"Best XGBoost regressor RMSE: {best_score:.4f}")
    
    # Return the best parameters
    return best_params

def optimize_rf_classifier(X_train, y_train):
    """
    Optimize Random Forest classifier using Optuna.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Binary classification target
        
    Returns:
    --------
    dict
        Best hyperparameters
    """
    logger.info("Optimizing Random Forest classifier")
    
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': config.RANDOM_STATE
        }
        
        # Create model
        model = RandomForestClassifier(**param)
        
        # Cross-validation
        cv = KFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        # Return the mean AUC score
        return np.mean(cv_scores)
    
    # Create the study
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    
    # Run the optimization
    study.optimize(objective, n_trials=config.N_TRIALS, timeout=config.TIMEOUT)
    
    # Get the best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    logger.info(f"Best Random Forest classifier parameters: {best_params}")
    logger.info(f"Best Random Forest classifier AUC: {best_score:.4f}")
    
    # Return the best parameters
    return best_params

def optimize_rf_regressor(X_train, y_train):
    """
    Optimize Random Forest regressor using Optuna.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Regression target
        
    Returns:
    --------
    dict
        Best hyperparameters
    """
    logger.info("Optimizing Random Forest regressor")
    
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': config.RANDOM_STATE
        }
        
        # Create model
        model = RandomForestRegressor(**param)
        
        # Cross-validation
        cv = KFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        cv_scores = -cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
        
        # Return the mean RMSE score (lower is better)
        return np.mean(cv_scores)
    
    # Create the study
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    
    # Run the optimization
    study.optimize(objective, n_trials=config.N_TRIALS, timeout=config.TIMEOUT)
    
    # Get the best parameters
    best_params = study.best_params
    best_score = study.best_value
    
    logger.info(f"Best Random Forest regressor parameters: {best_params}")
    logger.info(f"Best Random Forest regressor RMSE: {best_score:.4f}")
    
    # Return the best parameters
    return best_params

def train_xgb_classifier(X_train, X_test, y_train, y_test, params=None):
    """
    Train XGBoost classifier with given parameters or default parameters.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series
        Training and test targets
    params : dict
        Model parameters
        
    Returns:
    --------
    tuple
        (model, results_dict)
    """
    logger.info("Training XGBoost classifier")
    
    if params is None:
        logger.info("Using default XGBoost parameters")
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': config.RANDOM_STATE
        }
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Train the model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Get predictions
    train_pred_prob = model.predict(dtrain)
    test_pred_prob = model.predict(dtest)
    
    train_pred = (train_pred_prob > 0.5).astype(int)
    test_pred = (test_pred_prob > 0.5).astype(int)
    
    # Compute metrics
    results = {
        'train_accuracy': accuracy_score(y_train, train_pred),
        'test_accuracy': accuracy_score(y_test, test_pred),
        'train_precision': precision_score(y_train, train_pred),
        'test_precision': precision_score(y_test, test_pred),
        'train_recall': recall_score(y_train, train_pred),
        'test_recall': recall_score(y_test, test_pred),
        'train_f1': f1_score(y_train, train_pred),
        'test_f1': f1_score(y_test, test_pred),
        'train_auc': roc_auc_score(y_train, train_pred_prob),
        'test_auc': roc_auc_score(y_test, test_pred_prob)
    }
    
    logger.info(f"XGBoost classifier results: "
                f"AUC = {results['test_auc']:.4f}, "
                f"F1 = {results['test_f1']:.4f}")
    
    # Save the model
    model.save_model(str(MODELS_DIR / "xgb_classifier_model.json"))
    
    return model, results

def train_xgb_regressor(X_train, X_test, y_train, y_test, params=None):
    """
    Train XGBoost regressor with given parameters or default parameters.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series
        Training and test targets
    params : dict
        Model parameters
        
    Returns:
    --------
    tuple
        (model, results_dict)
    """
    logger.info("Training XGBoost regressor")
    
    if params is None:
        logger.info("Using default XGBoost parameters")
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': config.RANDOM_STATE
        }
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Train the model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Get predictions
    train_pred = model.predict(dtrain)
    test_pred = model.predict(dtest)
    
    # Compute metrics
    results = {
        'train_mae': mean_absolute_error(y_train, train_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred)
    }
    
    logger.info(f"XGBoost regressor results: "
                f"RMSE = {results['test_rmse']:.4f}, "
                f"R² = {results['test_r2']:.4f}")
    
    # Save the model
    model.save_model(str(MODELS_DIR / "xgb_regressor_model.json"))
    
    return model, results

def train_rf_classifier(X_train, X_test, y_train, y_test, params=None):
    """
    Train Random Forest classifier with given parameters or default parameters.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series
        Training and test targets
    params : dict
        Model parameters
        
    Returns:
    --------
    tuple
        (model, results_dict)
    """
    logger.info("Training Random Forest classifier")
    
    if params is None:
        logger.info("Using default Random Forest parameters")
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': config.RANDOM_STATE
        }
    
    # Create and train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Get predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_pred_prob = model.predict_proba(X_train)[:, 1]
    test_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    results = {
        'train_accuracy': accuracy_score(y_train, train_pred),
        'test_accuracy': accuracy_score(y_test, test_pred),
        'train_precision': precision_score(y_train, train_pred),
        'test_precision': precision_score(y_test, test_pred),
        'train_recall': recall_score(y_train, train_pred),
        'test_recall': recall_score(y_test, test_pred),
        'train_f1': f1_score(y_train, train_pred),
        'test_f1': f1_score(y_test, test_pred),
        'train_auc': roc_auc_score(y_train, train_pred_prob),
        'test_auc': roc_auc_score(y_test, test_pred_prob)
    }
    
    logger.info(f"Random Forest classifier results: "
                f"AUC = {results['test_auc']:.4f}, "
                f"F1 = {results['test_f1']:.4f}")
    
    # Save the model
    joblib.dump(model, MODELS_DIR / "rf_classifier_model.joblib")
    
    return model, results

def train_rf_regressor(X_train, X_test, y_train, y_test, params=None):
    """
    Train Random Forest regressor with given parameters or default parameters.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series
        Training and test targets
    params : dict
        Model parameters
        
    Returns:
    --------
    tuple
        (model, results_dict)
    """
    logger.info("Training Random Forest regressor")
    
    if params is None:
        logger.info("Using default Random Forest parameters")
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': config.RANDOM_STATE
        }
    
    # Create and train model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    # Get predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Compute metrics
    results = {
        'train_mae': mean_absolute_error(y_train, train_pred),
        'test_mae': mean_absolute_error(y_test, test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'train_r2': r2_score(y_train, train_pred),
        'test_r2': r2_score(y_test, test_pred)
    }
    
    logger.info(f"Random Forest regressor results: "
                f"RMSE = {results['test_rmse']:.4f}, "
                f"R² = {results['test_r2']:.4f}")
    
    # Save the model
    joblib.dump(model, MODELS_DIR / "rf_regressor_model.joblib")
    
    return model, results

def save_feature_importance(model, feature_names, model_name, task_type):
    """
    Save feature importance information for a given model.
    
    Parameters:
    -----------
    model : trained model
        The trained model to extract feature importance from
    feature_names : list
        Names of features
    model_name : str
        Name of the model
    task_type : str
        Type of task (classification or regression)
    """
    logger.info(f"Saving feature importance for {model_name}")
    
    # Different models have different ways of accessing feature importance
    if model_name.startswith('xgb'):
        # For XGBoost
        importance_type = 'weight'  # alternatives: 'gain', 'cover', 'total_gain', 'total_cover'
        importance_dict = model.get_score(importance_type=importance_type)
        
        # Convert feature indices to names if needed
        if all(key.isdigit() for key in importance_dict.keys()):
            importance = [(feature_names[int(idx)], score) for idx, score in importance_dict.items()]
        else:
            importance = [(feature, score) for feature, score in importance_dict.items()]
    
    else:
        # For scikit-learn models like Random Forest
        importance = [(name, score) for name, score in 
                      zip(feature_names, model.feature_importances_)]
    
    # Sort by importance
    importance.sort(key=lambda x: x[1], reverse=True)
    
    # Convert to DataFrame
    importance_df = pd.DataFrame(importance, columns=['feature', 'importance'])
    
    # Save to parquet
    file_path = MODELS_DIR / f"{model_name}_{task_type}_feature_importance.parquet"
    importance_df.to_parquet(file_path)
    
    # Also save top N features to CSV for easy viewing
    top_n = min(30, len(importance_df))
    top_features_path = MODELS_DIR / f"{model_name}_{task_type}_top_features.csv"
    importance_df.head(top_n).to_csv(top_features_path, index=False)
    
    logger.info(f"Saved feature importance to {file_path}")
    logger.info(f"Top {top_n} features saved to {top_features_path}")
    
    # Print top 10 features
    top_10 = importance_df.head(10)
    logger.info(f"Top 10 features for {model_name} {task_type}:")
    for idx, row in top_10.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return importance_df

def calculate_skill_category_importance(importance_df, skill_categories):
    """
    Calculate the importance of different skill categories based on feature importance.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        Feature importance dataframe
    skill_categories : dict
        Dictionary mapping skill categories to skills
        
    Returns:
    --------
    pd.DataFrame
        Category importance dataframe
    """
    logger.info("Calculating skill category importance")
    
    # Create a reverse mapping from skill to category
    skill_to_category = {}
    for category, skills in skill_categories.items():
        for skill in skills:
            # Convert skill to lowercase for case-insensitive matching
            skill_to_category[skill.lower()] = category
    
    # Initialize category importance dictionary
    category_importance = {category: 0.0 for category in skill_categories.keys()}
    category_feature_count = {category: 0 for category in skill_categories.keys()}
    
    # Calculate total importance for each category
    for _, row in importance_df.iterrows():
        feature = row['feature'].lower()
        importance = row['importance']
        
        # Check if the feature contains any of the skills
        for skill, category in skill_to_category.items():
            if skill in feature:
                category_importance[category] += importance
                category_feature_count[category] += 1
                break
    
    # Create category importance dataframe
    category_df = pd.DataFrame({
        'category': list(category_importance.keys()),
        'total_importance': list(category_importance.values()),
        'feature_count': list(category_feature_count.values())
    })
    
    # Calculate average importance per feature in each category
    category_df['avg_importance_per_feature'] = (
        category_df['total_importance'] / category_df['feature_count'].clip(lower=1)
    )
    
    # Sort by total importance
    category_df = category_df.sort_values('total_importance', ascending=False)
    
    # Save category importance
    file_path = MODELS_DIR / "skill_category_importance.csv"
    category_df.to_csv(file_path, index=False)
    
    logger.info(f"Saved skill category importance to {file_path}")
    
    return category_df

def create_career_stage_specific_importances(importance_df, X_train, y_train, feature_names, model_type='rf', task_type='classification'):
    """
    Create career stage-specific feature importances.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        Overall feature importance dataframe
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    feature_names : list
        Feature names
    model_type : str
        Type of model to use ('rf' or 'xgb')
    task_type : str
        Type of task ('classification' or 'regression')
        
    Returns:
    --------
    dict
        Dictionary mapping career stages to feature importance dataframes
    """
    logger.info("Creating career stage-specific feature importances")
    
    # Get the career stage column
    career_stage_col = 'career_stage'
    
    # Check if the career stage column exists in the training data
    if career_stage_col not in X_train.columns:
        logger.warning(f"Career stage column '{career_stage_col}' not found in training data")
        return None
    
    # Define career stages
    career_stages = {
        'junior': [0, 3],  # 0-3 years
        'mid_level': [4, 7],  # 4-7 years
        'senior': [8, 15],  # 8-15 years
        'principal': [16, 100]  # 16+ years
    }
    
    # Create a dictionary to store importance by career stage
    importance_by_stage = {}
    
    # For each career stage, train a model and get feature importance
    for stage, year_range in career_stages.items():
        logger.info(f"Training model for career stage: {stage} ({year_range[0]}-{year_range[1]} years)")
        
        # Filter data for this career stage
        stage_mask = (X_train[career_stage_col] >= year_range[0]) & (X_train[career_stage_col] <= year_range[1])
        X_stage = X_train[stage_mask].drop(columns=[career_stage_col])
        y_stage = y_train[stage_mask]
        
        # Skip if not enough data
        if len(X_stage) < 50:
            logger.warning(f"Not enough data for career stage {stage} (only {len(X_stage)} samples)")
            continue
        
        # Train a new model for this career stage
        if model_type == 'rf':
            if task_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=config.RANDOM_STATE
                )
            else:  # regression
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=config.RANDOM_STATE
                )
        else:  # xgboost
            params = {
                'max_depth': 6,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': config.RANDOM_STATE
            }
            
            if task_type == 'classification':
                params['objective'] = 'binary:logistic'
                params['eval_metric'] = 'auc'
            else:  # regression
                params['objective'] = 'reg:squarederror'
                params['eval_metric'] = 'rmse'
            
            dtrain = xgb.DMatrix(X_stage, label=y_stage)
            model = xgb.train(params, dtrain, num_boost_round=100)
        
        # Get feature importance
        if model_type == 'rf':
            importance = [(name, score) for name, score in 
                          zip(feature_names[:-1], model.feature_importances_)]  # exclude career_stage
        else:  # xgboost
            importance_type = 'weight'
            importance_dict = model.get_score(importance_type=importance_type)
            
            # Convert feature indices to names if needed
            if all(key.isdigit() for key in importance_dict.keys()):
                stage_feature_names = [name for name in feature_names if name != career_stage_col]
                importance = [(stage_feature_names[int(idx)], score) for idx, score in importance_dict.items()]
            else:
                importance = [(feature, score) for feature, score in importance_dict.items()]
        
        # Sort by importance
        importance.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to DataFrame
        importance_df_stage = pd.DataFrame(importance, columns=['feature', 'importance'])
        
        # Save to file
        file_path = MODELS_DIR / f"{model_type}_{task_type}_{stage}_feature_importance.parquet"
        importance_df_stage.to_parquet(file_path)
        
        # Also save top N features to CSV for easy viewing
        top_n = min(30, len(importance_df_stage))
        top_features_path = MODELS_DIR / f"{model_type}_{task_type}_{stage}_top_features.csv"
        importance_df_stage.head(top_n).to_csv(top_features_path, index=False)
        
        logger.info(f"Saved {stage} feature importance to {file_path}")
        
        # Store in dictionary
        importance_by_stage[stage] = importance_df_stage
    
    return importance_by_stage

def save_model_metrics(all_results):
    """
    Save model metrics to file.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary of model results
    """
    logger.info("Saving model metrics")
    
    # Save as pickle for easy loading in future analysis
    with open(MODELS_DIR / "model_metrics.pkl", 'wb') as f:
        pickle.dump(all_results, f)
    
    # Create a summary DataFrame for easy viewing
    summary_rows = []
    
    for model_name, metrics in all_results.items():
        if 'test_auc' in metrics:  # Classification model
            row = {
                'model': model_name,
                'type': 'classification',
                'test_auc': metrics['test_auc'],
                'test_f1': metrics['test_f1'],
                'test_accuracy': metrics['test_accuracy'],
                'test_precision': metrics['test_precision'],
                'test_recall': metrics['test_recall']
            }
        else:  # Regression model
            row = {
                'model': model_name,
                'type': 'regression',
                'test_rmse': metrics['test_rmse'],
                'test_r2': metrics['test_r2'],
                'test_mae': metrics['test_mae']
            }
        
        summary_rows.append(row)
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary to CSV
    summary_path = MODELS_DIR / "model_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"Saved model metrics summary to {summary_path}")
    
    return summary_df

def main(year=2024, optimize=True):
    """
    Main function to train and evaluate models.
    
    Parameters:
    -----------
    year : int
        Survey year
    optimize : bool
        Whether to optimize hyperparameters
    """
    logger.info(f"Starting model training for year {year}")
    
    # Load the data
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = load_training_data(year)
    
    if X_train is None:
        logger.error("Failed to load training data")
        return
    
    # Preprocess features
    X_train_proc, X_test_proc, feature_names = preprocess_features(X_train, X_test)
    
    # Dictionary to store all results
    all_results = {}
    
    # Train baseline models
    baseline_results = train_baseline_models(
        X_train_proc, X_test_proc, y_class_train, y_class_test, y_reg_train, y_reg_test
    )
    
    all_results.update(baseline_results)
    
    # Train Random Forest models
    if optimize:
        # Optimize Random Forest classifier
        rf_class_params = optimize_rf_classifier(X_train_proc, y_class_train)
        
        # Optimize Random Forest regressor
        rf_reg_params = optimize_rf_regressor(X_train_proc, y_reg_train)
    else:
        rf_class_params = None
        rf_reg_params = None
    
    # Train Random Forest classifier
    rf_class_model, rf_class_results = train_rf_classifier(
        X_train_proc, X_test_proc, y_class_train, y_class_test, rf_class_params
    )
    
    all_results['rf_classifier'] = rf_class_results
    
    # Train Random Forest regressor
    rf_reg_model, rf_reg_results = train_rf_regressor(
        X_train_proc, X_test_proc, y_reg_train, y_reg_test, rf_reg_params
    )
    
    all_results['rf_regressor'] = rf_reg_results
    
    # Train XGBoost models
    if optimize:
        # Optimize XGBoost classifier
        xgb_class_params = optimize_xgb_classifier(X_train_proc, y_class_train)
        
        # Optimize XGBoost regressor
        xgb_reg_params = optimize_xgb_regressor(X_train_proc, y_reg_train)
    else:
        xgb_class_params = None
        xgb_reg_params = None
    
    # Train XGBoost classifier
    xgb_class_model, xgb_class_results = train_xgb_classifier(
        X_train_proc, X_test_proc, y_class_train, y_class_test, xgb_class_params
    )
    
    all_results['xgb_classifier'] = xgb_class_results
    
    # Train XGBoost regressor
    xgb_reg_model, xgb_reg_results = train_xgb_regressor(
        X_train_proc, X_test_proc, y_reg_train, y_reg_test, xgb_reg_params
    )
    
    all_results['xgb_regressor'] = xgb_reg_results
    
    # Save model metrics
    summary_df = save_model_metrics(all_results)
    
    # Print the best models
    if 'rf_classifier' in all_results and 'xgb_classifier' in all_results:
        rf_auc = all_results['rf_classifier']['test_auc']
        xgb_auc = all_results['xgb_classifier']['test_auc']
        
        if rf_auc > xgb_auc:
            logger.info(f"Best classifier: Random Forest (AUC = {rf_auc:.4f})")
            best_class_model = rf_class_model
            best_class_type = 'rf'
        else:
            logger.info(f"Best classifier: XGBoost (AUC = {xgb_auc:.4f})")
            best_class_model = xgb_class_model
            best_class_type = 'xgb'
    else:
        logger.warning("Could not determine best classifier")
        best_class_model = None
        best_class_type = None
    
    if 'rf_regressor' in all_results and 'xgb_regressor' in all_results:
        rf_rmse = all_results['rf_regressor']['test_rmse']
        xgb_rmse = all_results['xgb_regressor']['test_rmse']
        
        if rf_rmse < xgb_rmse:
            logger.info(f"Best regressor: Random Forest (RMSE = {rf_rmse:.4f})")
            best_reg_model = rf_reg_model
            best_reg_type = 'rf'
        else:
            logger.info(f"Best regressor: XGBoost (RMSE = {xgb_rmse:.4f})")
            best_reg_model = xgb_reg_model
            best_reg_type = 'xgb'
    else:
        logger.warning("Could not determine best regressor")
        best_reg_model = None
        best_reg_type = None
    
    # Save feature importance for the best models
    if best_class_model is not None:
        logger.info("Saving feature importance for best classifier")
        class_importance = save_feature_importance(
            best_class_model, feature_names, best_class_type, 'classification'
        )
        
        # Calculate skill category importance
        if hasattr(config, 'SKILL_CATEGORIES'):
            calculate_skill_category_importance(class_importance, config.SKILL_CATEGORIES)
        
        # Create career stage-specific importances
        create_career_stage_specific_importances(
            class_importance, X_train, y_class_train, feature_names, 
            model_type=best_class_type, task_type='classification'
        )
    
    if best_reg_model is not None:
        logger.info("Saving feature importance for best regressor")
        reg_importance = save_feature_importance(
            best_reg_model, feature_names, best_reg_type, 'regression'
        )
        
        # Calculate skill category importance
        if hasattr(config, 'SKILL_CATEGORIES'):
            calculate_skill_category_importance(reg_importance, config.SKILL_CATEGORIES)
        
        # Create career stage-specific importances
        create_career_stage_specific_importances(
            reg_importance, X_train, y_reg_train, feature_names, 
            model_type=best_reg_type, task_type='regression'
        )
    
    logger.info("Model training complete")
    return all_results

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train models on Stack Overflow Developer Survey data')
    parser.add_argument('--year', type=int, default=2024, help='Survey year')
    parser.add_argument('--no-optimize', action='store_true', help='Skip hyperparameter optimization')
    
    args = parser.parse_args()
    
    # Run main function
    main(year=args.year, optimize=not args.no_optimize)
