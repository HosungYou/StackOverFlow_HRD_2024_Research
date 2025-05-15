#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stack Overflow Developer Survey 2024 - Model Interpretation
Human Resource Development (HRD) Focus

This script uses SHAP (SHapley Additive exPlanations) to interpret the trained models
and quantify the value of different developer skills from an HRD perspective.

Key analyses:
1. Global feature importance analysis
2. Individual prediction explanations
3. Career stage-specific skill value analysis
4. Interaction effects between skills and experience
5. Skill domain importance across different regions and roles
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
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Add the project root to the path so we can import the config
project_root = Path("/Volumes/External SSD/Pycharm/StackOverFlow_HRD_2024")
sys.path.append(str(project_root))

# Import config module
import importlib.util
config_path = project_root / "src" / "00_config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Set paths
PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
MODELS_DIR = config.MODELS_DIR
REPORTS_DIR = config.REPORTS_DIR
REPORTS_FIGURES_DIR = config.REPORTS_DIR / "figures"
logger = config.logger

# Make sure the report directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_data_and_models(year=2024):
    """
    Load the processed data and trained models.
    
    Parameters:
    -----------
    year : int
        Survey year
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test, 
         best_classifier, best_regressor, feature_names)
    """
    logger.info("Loading data and models")
    
    # Load the training and test data
    train_path = PROCESSED_DATA_DIR / f"features_{year}_train.parquet"
    test_path = PROCESSED_DATA_DIR / f"features_{year}_test.parquet"
    
    if not train_path.exists() or not test_path.exists():
        logger.error(f"Training/test data not found. Please run feature engineering first.")
        return None
    
    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_parquet(train_path)
    
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_parquet(test_path)
    
    # Get the outcome variables
    class_target = 'is_high_wage'
    reg_target = 'salary_log'
    
    # Get the feature columns
    id_cols = ['ResponseId', config.COLUMN_NAMES['salary']]
    outcome_cols = [class_target, reg_target]
    feature_cols = [col for col in train_df.columns if col not in id_cols and col not in outcome_cols]
    
    # Create the train/test feature matrices and target vectors
    X_train = train_df[feature_cols]
    y_class_train = train_df[class_target]
    y_reg_train = train_df[reg_target]
    
    X_test = test_df[feature_cols]
    y_class_test = test_df[class_target]
    y_reg_test = test_df[reg_target]
    
    # Apply the same preprocessing as in the modeling script
    # Find and drop non-numeric columns
    non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        logger.info(f"Dropping {len(non_numeric_cols)} non-numeric columns")
        X_train = X_train.drop(columns=non_numeric_cols)
        X_test = X_test.drop(columns=non_numeric_cols)
    
    # Find and drop columns with missing values
    missing_cols = X_train.columns[X_train.isna().any()].tolist()
    if missing_cols:
        logger.info(f"Dropping {len(missing_cols)} columns with missing values")
        X_train = X_train.drop(columns=missing_cols)
        X_test = X_test.drop(columns=missing_cols)
    
    # Get feature names after preprocessing
    feature_names = X_train.columns.tolist()
    
    # Load the scaler
    scaler_path = MODELS_DIR / "feature_scaler.joblib"
    if scaler_path.exists():
        logger.info(f"Loading feature scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        
        # Apply scaling
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
    else:
        logger.warning("Feature scaler not found. Using unscaled features.")
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Try to load model metrics to determine best models
    try:
        with open(MODELS_DIR / "model_metrics.pkl", 'rb') as f:
            all_metrics = pickle.load(f)
        
        # Determine best classifier (highest AUC)
        if 'xgb_classifier' in all_metrics and 'rf_classifier' in all_metrics:
            if all_metrics['xgb_classifier']['test_auc'] > all_metrics['rf_classifier']['test_auc']:
                best_classifier_name = 'xgb_classifier'
                best_classifier_path = MODELS_DIR / "xgb_classifier_model.json"
                is_xgb_classifier = True
            else:
                best_classifier_name = 'rf_classifier'
                best_classifier_path = MODELS_DIR / "rf_classifier_model.joblib"
                is_xgb_classifier = False
            
            logger.info(f"Best classifier: {best_classifier_name}")
        else:
            logger.warning("Could not determine best classifier from metrics")
            best_classifier_name = None
            best_classifier_path = None
            is_xgb_classifier = None
        
        # Determine best regressor (lowest RMSE)
        if 'xgb_regressor' in all_metrics and 'rf_regressor' in all_metrics:
            if all_metrics['xgb_regressor']['test_rmse'] < all_metrics['rf_regressor']['test_rmse']:
                best_regressor_name = 'xgb_regressor'
                best_regressor_path = MODELS_DIR / "xgb_regressor_model.json"
                is_xgb_regressor = True
            else:
                best_regressor_name = 'rf_regressor'
                best_regressor_path = MODELS_DIR / "rf_regressor_model.joblib"
                is_xgb_regressor = False
            
            logger.info(f"Best regressor: {best_regressor_name}")
        else:
            logger.warning("Could not determine best regressor from metrics")
            best_regressor_name = None
            best_regressor_path = None
            is_xgb_regressor = None
    
    except Exception as e:
        logger.warning(f"Failed to load model metrics: {e}")
        logger.warning("Trying to find best models based on file existence")
        
        # Try to find best classifier
        if (MODELS_DIR / "xgb_classifier_model.json").exists():
            best_classifier_name = 'xgb_classifier'
            best_classifier_path = MODELS_DIR / "xgb_classifier_model.json"
            is_xgb_classifier = True
        elif (MODELS_DIR / "rf_classifier_model.joblib").exists():
            best_classifier_name = 'rf_classifier'
            best_classifier_path = MODELS_DIR / "rf_classifier_model.joblib"
            is_xgb_classifier = False
        else:
            logger.error("No classifier model found")
            best_classifier_name = None
            best_classifier_path = None
            is_xgb_classifier = None
        
        # Try to find best regressor
        if (MODELS_DIR / "xgb_regressor_model.json").exists():
            best_regressor_name = 'xgb_regressor'
            best_regressor_path = MODELS_DIR / "xgb_regressor_model.json"
            is_xgb_regressor = True
        elif (MODELS_DIR / "rf_regressor_model.joblib").exists():
            best_regressor_name = 'rf_regressor'
            best_regressor_path = MODELS_DIR / "rf_regressor_model.joblib"
            is_xgb_regressor = False
        else:
            logger.error("No regressor model found")
            best_regressor_name = None
            best_regressor_path = None
            is_xgb_regressor = None
    
    # Load the best classifier
    best_classifier = None
    if best_classifier_path and best_classifier_path.exists():
        logger.info(f"Loading best classifier from {best_classifier_path}")
        try:
            if is_xgb_classifier:
                import xgboost as xgb
                best_classifier = xgb.Booster()
                best_classifier.load_model(str(best_classifier_path))
            else:
                best_classifier = joblib.load(best_classifier_path)
        except Exception as e:
            logger.error(f"Failed to load best classifier: {e}")
    else:
        logger.warning("Best classifier not found")
    
    # Load the best regressor
    best_regressor = None
    if best_regressor_path and best_regressor_path.exists():
        logger.info(f"Loading best regressor from {best_regressor_path}")
        try:
            if is_xgb_regressor:
                import xgboost as xgb
                best_regressor = xgb.Booster()
                best_regressor.load_model(str(best_regressor_path))
            else:
                best_regressor = joblib.load(best_regressor_path)
        except Exception as e:
            logger.error(f"Failed to load best regressor: {e}")
    else:
        logger.warning("Best regressor not found")
    
    model_info = {
        'classifier': {
            'name': best_classifier_name,
            'model': best_classifier,
            'is_xgb': is_xgb_classifier
        },
        'regressor': {
            'name': best_regressor_name,
            'model': best_regressor,
            'is_xgb': is_xgb_regressor
        }
    }
    
    # Return the data and models
    return X_train_scaled, X_test_scaled, y_class_train, y_class_test, y_reg_train, y_reg_test, model_info, feature_names

def compute_shap_values(model_info, X_train, X_test, model_type='classifier'):
    """
    Compute SHAP values for model explanation.
    
    Parameters:
    -----------
    model_info : dict
        Dictionary with model information
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    model_type : str
        Type of model ('classifier' or 'regressor')
        
    Returns:
    --------
    tuple
        (explainer, shap_values, shap_values_test)
    """
    logger.info(f"Computing SHAP values for {model_type}")
    
    # Get the model information
    model = model_info[model_type]['model']
    is_xgb = model_info[model_type]['is_xgb']
    model_name = model_info[model_type]['name']
    
    if model is None:
        logger.error(f"No {model_type} model available")
        return None, None, None
    
    # Create the explainer based on the model type
    if is_xgb:
        # For XGBoost models
        import xgboost as xgb
        
        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train)
        dtest = xgb.DMatrix(X_test)
        
        # Use TreeExplainer for XGBoost
        explainer = shap.TreeExplainer(model)
    else:
        # For scikit-learn models
        # Use TreeExplainer for Random Forest
        explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    logger.info(f"Calculating SHAP values for {model_type} on training data")
    if is_xgb:
        # For XGBoost, we need to use DMatrix
        shap_values = explainer.shap_values(dtrain)
    else:
        # For scikit-learn, we can use the dataframe directly
        shap_values = explainer.shap_values(X_train)
    
    logger.info(f"Calculating SHAP values for {model_type} on test data")
    if is_xgb:
        # For XGBoost, we need to use DMatrix
        shap_values_test = explainer.shap_values(dtest)
    else:
        # For scikit-learn, we can use the dataframe directly
        shap_values_test = explainer.shap_values(X_test)
    
    logger.info(f"SHAP values computed for {model_type}")
    
    return explainer, shap_values, shap_values_test

def plot_shap_summary(shap_values, X, feature_names, model_type, output_path=None):
    """
    Create SHAP summary plot and save to file.
    
    Parameters:
    -----------
    shap_values : numpy array
        SHAP values
    X : pd.DataFrame
        Feature values
    feature_names : list
        Names of features
    model_type : str
        Type of model ('classifier' or 'regressor')
    output_path : Path
        Path to save the plot
    """
    logger.info(f"Creating SHAP summary plot for {model_type}")
    
    # Set figure size
    plt.figure(figsize=(12, max(8, min(25, len(feature_names) * 0.3))))
    
    # Create summary plot
    # For classifier, we often get a list of SHAP values (one array per class)
    # For binary classification, use the second array (positive class)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_plot = shap_values[1]
    else:
        shap_values_plot = shap_values
    
    shap.summary_plot(
        shap_values_plot, 
        X, 
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    
    # Add title and adjust layout
    plt.title(f"Feature Importance ({model_type.capitalize()})", fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved SHAP summary plot to {output_path}")
    
    plt.close()
    
    # Also create a dot plot
    plt.figure(figsize=(12, max(8, min(25, len(feature_names) * 0.3))))
    
    shap.summary_plot(
        shap_values_plot, 
        X, 
        feature_names=feature_names,
        plot_type="dot",
        show=False
    )
    
    # Add title and adjust layout
    plt.title(f"Feature Value Impact ({model_type.capitalize()})", fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    if output_path:
        dot_plot_path = str(output_path).replace('.png', '_dot.png')
        plt.savefig(dot_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved SHAP dot plot to {dot_plot_path}")
    
    plt.close()

def plot_shap_skill_category_importance(shap_values, X, feature_names, skill_categories, model_type, output_path=None):
    """
    Create and save a plot of SHAP importance by skill category.
    
    Parameters:
    -----------
    shap_values : numpy array
        SHAP values
    X : pd.DataFrame
        Feature values
    feature_names : list
        Names of features
    skill_categories : dict
        Dictionary mapping skill categories to skills
    model_type : str
        Type of model ('classifier' or 'regressor')
    output_path : Path
        Path to save the plot
    """
    logger.info(f"Analyzing SHAP values by skill category for {model_type}")
    
    # Handle multi-class SHAP values (for classifiers)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_use = shap_values[1]  # Use positive class for binary classification
    else:
        shap_values_use = shap_values
    
    # Create a reverse mapping from skill to category
    skill_to_category = {}
    for category, skills in skill_categories.items():
        for skill in skills:
            # Convert skill to lowercase for case-insensitive matching
            skill_to_category[skill.lower()] = category
    
    # Initialize SHAP importance by category
    category_importance = {category: [] for category in skill_categories.keys()}
    
    # Assign features to categories and collect SHAP values
    for i, feature in enumerate(feature_names):
        # Check if feature matches any skill
        feature_lower = feature.lower()
        matched = False
        
        for skill, category in skill_to_category.items():
            if skill in feature_lower:
                # If feature matches a skill, add its SHAP values to the category
                category_importance[category].append(np.abs(shap_values_use[:, i]))
                matched = True
                break
    
    # Calculate mean absolute SHAP value for each category
    category_mean_shap = {}
    for category, shap_arrays in category_importance.items():
        if shap_arrays:  # Skip empty categories
            # Concatenate all SHAP values for this category
            all_shap = np.concatenate([s.reshape(-1, 1) for s in shap_arrays], axis=1)
            # Calculate mean across all features in this category
            category_mean_shap[category] = np.mean(all_shap)
    
    # Create a DataFrame for plotting
    category_df = pd.DataFrame({
        'Category': list(category_mean_shap.keys()),
        'Mean SHAP Impact': list(category_mean_shap.values())
    })
    
    # Sort by importance
    category_df = category_df.sort_values('Mean SHAP Impact', ascending=False)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Mean SHAP Impact', y='Category', data=category_df, palette='viridis')
    
    plt.title(f'Skill Category Importance ({model_type.capitalize()})', fontsize=16)
    plt.xlabel('Mean |SHAP Value|', fontsize=14)
    plt.ylabel('Skill Category', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved skill category importance plot to {output_path}")
    
    plt.close()
    
    # Also save the data
    if output_path:
        data_path = str(output_path).replace('.png', '.csv')
        category_df.to_csv(data_path, index=False)
        logger.info(f"Saved skill category importance data to {data_path}")
    
    return category_df

def analyze_career_stage_interactions(shap_values, X, feature_names, model_type, output_path=None):
    """
    Analyze interactions between career stage and skills.
    
    Parameters:
    -----------
    shap_values : numpy array
        SHAP values
    X : pd.DataFrame
        Feature values
    feature_names : list
        Names of features
    model_type : str
        Type of model ('classifier' or 'regressor')
    output_path : Path
        Path to save the plot
    """
    logger.info(f"Analyzing career stage interactions for {model_type}")
    
    # Handle multi-class SHAP values (for classifiers)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_use = shap_values[1]  # Use positive class for binary classification
    else:
        shap_values_use = shap_values
    
    # Look for career stage column
    career_stage_col = 'career_stage'
    if career_stage_col not in X.columns:
        logger.warning(f"Career stage column '{career_stage_col}' not found in features")
        return None
    
    # Get index of career stage column
    career_stage_idx = list(X.columns).index(career_stage_col)
    
    # Define career stage bins
    career_bins = [0, 3, 7, 15, 100]
    career_labels = ['Junior (0-3)', 'Mid (4-7)', 'Senior (8-15)', 'Principal (16+)']
    
    # Create a career stage category for each sample
    career_stage_values = X[career_stage_col].values
    career_stage_bins = np.digitize(career_stage_values, career_bins)
    career_stage_cats = np.array([career_labels[min(b - 1, len(career_labels) - 1)] for b in career_stage_bins])
    
    # Find top skills (excluding career stage)
    feature_importances = np.mean(np.abs(shap_values_use), axis=0)
    # Exclude career stage from top features
    feature_importances[career_stage_idx] = 0
    
    # Get top N skills
    top_n = 10
    top_indices = np.argsort(feature_importances)[-top_n:]
    top_features = [feature_names[i] for i in top_indices]
    
    # Create a plot for each top feature showing interaction with career stage
    plt.figure(figsize=(15, 10))
    
    for i, feat_idx in enumerate(top_indices):
        # Skip if the feature is career stage
        if feat_idx == career_stage_idx:
            continue
        
        # Create a subplot
        plt.subplot(2, 5, i + 1)
        
        # Get feature values and SHAP values
        feature_vals = X.iloc[:, feat_idx]
        feature_shap = shap_values_use[:, feat_idx]
        
        # Create scatter plot colored by career stage
        unique_stages = np.unique(career_stage_cats)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_stages)))
        
        for j, stage in enumerate(unique_stages):
            mask = career_stage_cats == stage
            plt.scatter(
                feature_vals[mask],
                feature_shap[mask],
                label=stage,
                alpha=0.6,
                s=20,
                color=colors[j]
            )
        
        plt.title(feature_names[feat_idx], fontsize=12)
        plt.xlabel('Feature Value')
        plt.ylabel('SHAP Value')
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add legend
    plt.figlegend(loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4)
    
    plt.suptitle(f'Skill Value by Career Stage ({model_type.capitalize()})', fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for the title and legend
    
    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved career stage interaction plot to {output_path}")
    
    plt.close()
    
    # Also create boxplots showing distribution of SHAP values by career stage for top features
    plt.figure(figsize=(15, 10))
    
    for i, feat_idx in enumerate(top_indices):
        # Skip if the feature is career stage
        if feat_idx == career_stage_idx:
            continue
        
        # Create a subplot
        plt.subplot(2, 5, i + 1)
        
        # Get SHAP values
        feature_shap = shap_values_use[:, feat_idx]
        
        # Create a DataFrame for easier plotting
        plot_df = pd.DataFrame({
            'SHAP Value': feature_shap,
            'Career Stage': career_stage_cats
        })
        
        # Create boxplot
        sns.boxplot(x='Career Stage', y='SHAP Value', data=plot_df, palette='viridis')
        
        plt.title(feature_names[feat_idx], fontsize=12)
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.suptitle(f'Skill Value Distribution by Career Stage ({model_type.capitalize()})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the title
    
    # Save the plot
    if output_path:
        boxplot_path = str(output_path).replace('.png', '_boxplot.png')
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved career stage boxplot to {boxplot_path}")
    
    plt.close()

def analyze_hrd_roi(shap_values, X, feature_names, model_type, output_path=None):
    """
    Analyze potential Return on Investment (ROI) for skill development.
    
    Parameters:
    -----------
    shap_values : numpy array
        SHAP values
    X : pd.DataFrame
        Feature values
    feature_names : list
        Names of features
    model_type : str
        Type of model ('classifier' or 'regressor')
    output_path : Path
        Path to save the plot
    """
    logger.info(f"Analyzing HRD ROI for {model_type}")
    
    # Handle multi-class SHAP values (for classifiers)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_use = shap_values[1]  # Use positive class for binary classification
    else:
        shap_values_use = shap_values
    
    # Calculate mean absolute SHAP value for each feature
    mean_shap = np.mean(np.abs(shap_values_use), axis=0)
    
    # Calculate median feature value
    median_values = np.median(X, axis=0)
    
    # Calculate room for growth (1 - median_value) for each feature
    # For binary features, this is the percentage of developers who don't have the skill
    # For continuous features, this is the normalized room for improvement
    room_for_growth = 1 - median_values
    
    # Calculate ROI as (mean_shap * room_for_growth)
    # This prioritizes high-impact skills that are currently uncommon
    roi_scores = mean_shap * room_for_growth
    
    # Create DataFrame for plotting
    roi_df = pd.DataFrame({
        'Feature': feature_names,
        'Impact (Mean |SHAP|)': mean_shap,
        'Room for Growth': room_for_growth,
        'ROI Score': roi_scores
    })
    
    # Sort by ROI score
    roi_df = roi_df.sort_values('ROI Score', ascending=False)
    
    # Get top N features
    top_n = min(20, len(roi_df))
    top_roi = roi_df.head(top_n)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Create a color map based on Impact (Mean |SHAP|)
    norm = plt.Normalize(top_roi['Impact (Mean |SHAP|)'].min(), top_roi['Impact (Mean |SHAP|)'].max())
    colors = plt.cm.viridis(norm(top_roi['Impact (Mean |SHAP|)'].values))
    
    # Plot bars
    bars = plt.barh(top_roi['Feature'], top_roi['ROI Score'], color=colors)
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Impact (Mean |SHAP|)', rotation=270, labelpad=20)
    
    plt.title(f'Top Skills by ROI for Development ({model_type.capitalize()})', fontsize=16)
    plt.xlabel('ROI Score (Impact * Room for Growth)', fontsize=14)
    plt.ylabel('Skill/Feature', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved HRD ROI plot to {output_path}")
    
    plt.close()
    
    # Also save the data
    if output_path:
        data_path = str(output_path).replace('.png', '.csv')
        roi_df.to_csv(data_path, index=False)
        logger.info(f"Saved HRD ROI data to {data_path}")
    
    return roi_df

def analyze_skill_combinations(shap_values, X, feature_names, model_type, output_path=None):
    """
    Analyze synergies between skill combinations.
    
    Parameters:
    -----------
    shap_values : numpy array
        SHAP values
    X : pd.DataFrame
        Feature values
    feature_names : list
        Names of features
    model_type : str
        Type of model ('classifier' or 'regressor')
    output_path : Path
        Path to save the plot
    """
    logger.info(f"Analyzing skill combinations for {model_type}")
    
    # Handle multi-class SHAP values (for classifiers)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_use = shap_values[1]  # Use positive class for binary classification
    else:
        shap_values_use = shap_values
    
    # Calculate mean absolute SHAP value for each feature
    mean_shap = np.mean(np.abs(shap_values_use), axis=0)
    
    # Get top N features
    top_n = min(10, len(feature_names))
    top_indices = np.argsort(mean_shap)[-top_n:]
    top_features = [feature_names[i] for i in top_indices]
    
    # Create a correlation matrix for SHAP values of top features
    top_shap_values = shap_values_use[:, top_indices]
    shap_corr = np.corrcoef(top_shap_values.T)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Create a heatmap
    sns.heatmap(
        shap_corr,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        xticklabels=top_features,
        yticklabels=top_features
    )
    
    plt.title(f'SHAP Value Correlations Between Top Skills ({model_type.capitalize()})', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved skill combination plot to {output_path}")
    
    plt.close()
    
    # Also analyze joint impact of pairs of features
    # Look at the top pairs with highest combined impact
    pairs = []
    
    for i in range(len(top_indices)):
        for j in range(i+1, len(top_indices)):
            idx1 = top_indices[i]
            idx2 = top_indices[j]
            feat1 = feature_names[idx1]
            feat2 = feature_names[idx2]
            
            # Calculate combined absolute SHAP values
            combined_shap = np.abs(shap_values_use[:, idx1]) + np.abs(shap_values_use[:, idx2])
            mean_combined = np.mean(combined_shap)
            
            # Calculate individual impacts
            mean_shap1 = mean_shap[idx1]
            mean_shap2 = mean_shap[idx2]
            
            # Calculate synergy as the difference between combined impact and sum of individual impacts
            # If positive, the combination has synergy
            synergy = mean_combined - (mean_shap1 + mean_shap2)
            
            pairs.append({
                'Feature1': feat1,
                'Feature2': feat2,
                'Combined Impact': mean_combined,
                'Synergy': synergy
            })
    
    # Convert to DataFrame
    pairs_df = pd.DataFrame(pairs)
    
    # Sort by synergy
    pairs_df = pairs_df.sort_values('Synergy', ascending=False)
    
    # Save the data
    if output_path:
        pairs_path = str(output_path).replace('.png', '_pairs.csv')
        pairs_df.to_csv(pairs_path, index=False)
        logger.info(f"Saved skill pair analysis to {pairs_path}")
    
    return pairs_df

def map_shap_values_to_hct_framework(shap_values, X, feature_names, model_type, output_path=None):
    """
    Map SHAP values to Human Capital Theory framework, distinguishing between:
    - Basic vs. Applied skills (foundational vs. task-specific)
    - Technical vs. Non-technical attributes
    - Individual vs. Interactive effects
    
    This aligns with HCT concepts discussed in the academic paper section 2.1.
    
    Parameters:
    -----------
    shap_values : numpy array
        SHAP values
    X : pd.DataFrame
        Feature values
    feature_names : list
        Names of features
    model_type : str
        Type of model ('classifier' or 'regressor')
    output_path : Path
        Path to save the plot
        
    Returns:
    --------
    dict
        Dictionary with HCT-aligned analysis results
    """
    logger.info("Mapping SHAP values to Human Capital Theory framework")
    
    # Create HCT framework categories
    hct_categories = {
        "Basic_Skills": [col for col in feature_names if any(x in col.lower() for x in 
                        ["traditional", "education", "yearscode", "edlevel"])],
        "Applied_Skills": [col for col in feature_names if any(x in col.lower() for x in 
                          ["ai_ml", "cloud", "devops", "frontend", "backend", "mobile", "security"])],
        "Technical_Attributes": [col for col in feature_names if any(x in col.lower() for x in 
                                ["uses_", "has_", "skill", "tech", "lang", "framework"])],
        "Non_Technical_Attributes": [col for col in feature_names if any(x in col.lower() for x in 
                                    ["years", "age", "org", "employment", "remote", "region"])]
    }
    
    # Calculate average absolute SHAP values for each category
    category_impacts = {}
    for category, cols in hct_categories.items():
        valid_cols = [col for col in cols if col in feature_names]
        if valid_cols:
            # Get indices of these features
            indices = [feature_names.index(col) for col in valid_cols if feature_names.index(col) < shap_values.shape[1]]
            
            if indices:
                # Take absolute SHAP values and average across samples and features
                avg_impact = np.mean(np.abs(shap_values[:, indices]))
                category_impacts[category] = avg_impact
    
    # Calculate the ratio of applied to basic skills impact (measuring skill-specific returns)
    if "Applied_Skills" in category_impacts and "Basic_Skills" in category_impacts:
        applied_to_basic_ratio = category_impacts["Applied_Skills"] / category_impacts["Basic_Skills"]
    else:
        applied_to_basic_ratio = None
    
    # Calculate the ratio of technical to non-technical attributes
    if "Technical_Attributes" in category_impacts and "Non_Technical_Attributes" in category_impacts:
        technical_to_nontechnical_ratio = category_impacts["Technical_Attributes"] / category_impacts["Non_Technical_Attributes"]
    else:
        technical_to_nontechnical_ratio = None
    
    # Create visualization
    if category_impacts:
        plt.figure(figsize=(12, 10))
        
        # Bar chart of category impacts
        plt.subplot(2, 1, 1)
        categories = list(category_impacts.keys())
        impacts = [category_impacts[cat] for cat in categories]
        colors = sns.color_palette("viridis", len(categories))
        
        bars = plt.bar(categories, impacts, color=colors)
        plt.title("Human Capital Theory Framework: Impact of Skill Categories", fontsize=14)
        plt.ylabel("Average |SHAP Value|", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Create a ratio comparison
        plt.subplot(2, 1, 2)
        ratio_labels = []
        ratio_values = []
        
        if applied_to_basic_ratio is not None:
            ratio_labels.append("Applied/Basic Skills")
            ratio_values.append(applied_to_basic_ratio)
        
        if technical_to_nontechnical_ratio is not None:
            ratio_labels.append("Technical/Non-Technical")
            ratio_values.append(technical_to_nontechnical_ratio)
        
        if ratio_labels:
            bars = plt.bar(ratio_labels, ratio_values, color=sns.color_palette("viridis", len(ratio_labels)))
            plt.title("Relative Impact Ratios in Human Capital Framework", fontsize=14)
            plt.ylabel("Impact Ratio", fontsize=12)
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label="Equal Impact")
            plt.legend()
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved HCT framework analysis to {output_path}")
        
        plt.close()
    
    # Return results
    results = {
        "category_impacts": category_impacts,
        "applied_to_basic_ratio": applied_to_basic_ratio,
        "technical_to_nontechnical_ratio": technical_to_nontechnical_ratio
    }
    
    return results

def visualize_skill_development_roi(shap_values, X, feature_names, model_type, output_path=None):
    """
    Create visualizations specifically for ROI analysis, highlighting:
    - Estimated salary premium by skill domain
    - Comparative analysis of skill investment returns
    - Development pathway visualization
    
    This function supports RQ2 about ROI quantification.
    
    Parameters:
    -----------
    shap_values : numpy array
        SHAP values
    X : pd.DataFrame
        Feature values
    feature_names : list
        Names of features
    model_type : str
        Type of model ('classifier' or 'regressor')
    output_path : Path
        Path to save the plot
        
    Returns:
    --------
    dict
        Dictionary with ROI visualization metrics
    """
    logger.info("Creating skill development ROI visualizations")
    
    # Only proceed for regression model (salary prediction)
    if model_type != 'regressor':
        logger.info("Skipping ROI visualization for non-regression model")
        return None
    
    # Define skill domains of interest
    skill_domains = {
        "AI_ML": [col for col in feature_names if "ai" in col.lower() or "ml" in col.lower()],
        "Cloud": [col for col in feature_names if "cloud" in col.lower() or "aws" in col.lower() 
                 or "azure" in col.lower() or "gcp" in col.lower()],
        "DevOps": [col for col in feature_names if "devops" in col.lower() or "docker" in col.lower() 
                 or "kubernetes" in col.lower()],
        "Database": [col for col in feature_names if "sql" in col.lower() or "database" in col.lower()
                   or "postgres" in col.lower() or "mongodb" in col.lower()],
        "Frontend": [col for col in feature_names if "frontend" in col.lower() or "react" in col.lower()
                   or "angular" in col.lower() or "vue" in col.lower()],
        "Backend": [col for col in feature_names if "backend" in col.lower() or "java" in col.lower()
                  or "node" in col.lower() or "python" in col.lower()],
        "Security": [col for col in feature_names if "security" in col.lower() or "cyber" in col.lower()]
    }
    
    # Calculate average SHAP value for each domain
    domain_shap_values = {}
    for domain, cols in skill_domains.items():
        valid_cols = [col for col in cols if col in feature_names]
        if valid_cols:
            # Get indices of these features
            indices = [feature_names.index(col) for col in valid_cols if feature_names.index(col) < shap_values.shape[1]]
            
            if indices:
                # Take SHAP values and average across features (not absolute value)
                avg_shap = np.mean([np.mean(shap_values[:, idx]) for idx in indices])
                
                # Convert log-salary SHAP value to percentage salary increase
                # Using approximation: exp(shap_value) - 1 ≈ percentage change
                percent_increase = (np.exp(avg_shap) - 1) * 100
                
                domain_shap_values[domain] = {
                    "avg_shap": avg_shap,
                    "percent_increase": percent_increase
                }
    
    # Create a DataFrame for easier visualization
    roi_data = []
    for domain, values in domain_shap_values.items():
        roi_data.append({
            "Domain": domain,
            "Avg_SHAP": values["avg_shap"],
            "Percent_Increase": values["percent_increase"]
        })
    
    roi_df = pd.DataFrame(roi_data)
    
    # Sort by percentage increase
    roi_df = roi_df.sort_values("Percent_Increase", ascending=False)
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Salary premium by skill domain
    plt.subplot(2, 1, 1)
    bars = plt.bar(roi_df["Domain"], roi_df["Percent_Increase"], 
                  color=sns.color_palette("viridis", len(roi_df)))
    plt.title("Estimated Salary Premium by Skill Domain", fontsize=14)
    plt.ylabel("Estimated Salary Increase (%)", fontsize=12)
    plt.xlabel("Skill Domain", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Add a horizontal line for average premium across all domains
    avg_premium = roi_df["Percent_Increase"].mean()
    plt.axhline(y=avg_premium, color='r', linestyle='--', alpha=0.7, 
               label=f"Average Premium: {avg_premium:.1f}%")
    plt.legend()
    
    # Return on Investment Comparison
    # Simplified ROI calculation using dummy cost values
    # In practice, these would come from the hrd_roi_analysis module
    plt.subplot(2, 1, 2)
    
    # Dummy training costs for illustration
    # In production, these would be imported from the hrd_roi_analysis module
    training_costs = {
        "AI_ML": 10000,
        "Cloud": 8000,
        "DevOps": 9000,
        "Database": 7000,
        "Frontend": 7500,
        "Backend": 8500,
        "Security": 9500
    }
    
    # Calculate simple ROI: benefit/cost
    roi_df["Training_Cost"] = roi_df["Domain"].map(lambda x: training_costs.get(x, 8000))
    
    # Assume base salary of $100,000 for benefit calculation
    base_salary = 100000
    roi_df["Annual_Benefit"] = roi_df["Percent_Increase"] / 100 * base_salary
    roi_df["ROI_Ratio"] = roi_df["Annual_Benefit"] / roi_df["Training_Cost"]
    
    # Sort by ROI ratio
    roi_df = roi_df.sort_values("ROI_Ratio", ascending=False)
    
    # Plot ROI ratio
    bars = plt.bar(roi_df["Domain"], roi_df["ROI_Ratio"], 
                  color=sns.color_palette("viridis", len(roi_df)))
    plt.title("Return on Investment Ratio by Skill Domain (Annual Benefit / Training Cost)", fontsize=14)
    plt.ylabel("ROI Ratio", fontsize=12)
    plt.xlabel("Skill Domain", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Add a horizontal line for breakeven ROI
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, 
               label="Breakeven ROI (1.0)")
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved skill development ROI visualization to {output_path}")
    
    plt.close()
    
    # Return results
    results = {
        "domain_data": roi_df.to_dict(orient="records"),
        "avg_premium": avg_premium
    }
    
    return results

def identify_optimal_skill_development_paths(shap_values, X, feature_names, model_type, output_path=None):
    """
    Identify optimal skill development pathways, including:
    - Personalized pathways based on existing skills
    - Career stage-specific recommendations
    - Skill gap identification
    
    This function supports RQ4 about skill gap analysis and development pathways.
    
    Parameters:
    -----------
    shap_values : numpy array
        SHAP values
    X : pd.DataFrame
        Feature values
    feature_names : list
        Names of features
    model_type : str
        Type of model ('classifier' or 'regressor')
    output_path : Path
        Path to save the visualization
        
    Returns:
    --------
    dict
        Dictionary with optimal skill development paths
    """
    logger.info("Identifying optimal skill development paths")
    
    # Only proceed for regression model (salary prediction)
    if model_type != 'regressor':
        logger.info("Skipping pathway analysis for non-regression model")
        return None
    
    # Define skill categories and their features
    skill_categories = {
        "AI_ML": [col for col in feature_names if "ai" in col.lower() or "ml" in col.lower()],
        "Cloud": [col for col in feature_names if "cloud" in col.lower() or "aws" in col.lower() 
                 or "azure" in col.lower() or "gcp" in col.lower()],
        "DevOps": [col for col in feature_names if "devops" in col.lower() or "docker" in col.lower() 
                  or "kubernetes" in col.lower()],
        "WebDev": [col for col in feature_names if "web" in col.lower() or "frontend" in col.lower()
                  or "backend" in col.lower() or "fullstack" in col.lower()],
        "DataScience": [col for col in feature_names if "data" in col.lower() or "science" in col.lower()
                       or "analytics" in col.lower() or "statistics" in col.lower()],
        "Security": [col for col in feature_names if "security" in col.lower() or "cyber" in col.lower()]
    }
    
    # Define career stages and their features
    career_stages = {
        "Junior": [col for col in feature_names if "junior" in col.lower() or "0-3" in col],
        "Mid-level": [col for col in feature_names if "mid" in col.lower() or "4-7" in col],
        "Senior": [col for col in feature_names if "senior" in col.lower() or "8-15" in col],
        "Principal": [col for col in feature_names if "principal" in col.lower() or "16+" in col]
    }
    
    # Calculate the value of skill additions for each career stage
    # This represents the average SHAP value when someone has the skill
    development_paths = {}
    
    for stage, stage_cols in career_stages.items():
        if not stage_cols:
            continue
            
        # Find indices for this career stage
        stage_indices = []
        for col in stage_cols:
            if col in feature_names:
                idx = feature_names.index(col)
                if idx < X.shape[1]:
                    stage_indices.append(idx)
        
        if not stage_indices:
            continue
            
        # Find samples in this career stage
        stage_samples = np.where(X.iloc[:, stage_indices].any(axis=1))[0]
        
        if len(stage_samples) == 0:
            continue
            
        # Calculate skill value for each category in this career stage
        stage_values = {}
        
        for skill, skill_cols in skill_categories.items():
            if not skill_cols:
                continue
                
            # Find indices for this skill category
            skill_indices = []
            for col in skill_cols:
                if col in feature_names:
                    idx = feature_names.index(col)
                    if idx < shap_values.shape[1]:
                        skill_indices.append(idx)
            
            if not skill_indices:
                continue
                
            # Calculate average SHAP value for this skill in this career stage
            skill_shap = np.mean([np.mean(shap_values[stage_samples, idx]) for idx in skill_indices])
            
            # Convert to percentage
            skill_percent = (np.exp(skill_shap) - 1) * 100
            
            stage_values[skill] = {
                "shap_value": skill_shap,
                "percent_increase": skill_percent
            }
        
        # Sort skills by value for this career stage
        sorted_skills = sorted(stage_values.items(), key=lambda x: x[1]["percent_increase"], reverse=True)
        
        # Store the top skills for this stage
        development_paths[stage] = {
            "top_skills": [(skill, values) for skill, values in sorted_skills[:3]],
            "sample_size": len(stage_samples)
        }
    
    # Create visualization of optimal paths by career stage
    plt.figure(figsize=(14, 10))
    
    # Prepare data for grouped bar chart
    stages = list(development_paths.keys())
    skill_data = {skill: [] for skill in skill_categories.keys()}
    
    for stage in stages:
        # Get the percentage increase for each skill in this stage
        stage_data = development_paths[stage]
        top_dict = {skill: values["percent_increase"] for skill, values in stage_data["top_skills"]}
        
        # Add data for each skill
        for skill in skill_categories.keys():
            skill_data[skill].append(top_dict.get(skill, 0))
    
    # Plot grouped bar chart
    x = np.arange(len(stages))
    width = 0.15
    multiplier = 0
    
    # Set color palette
    colors = sns.color_palette("viridis", len(skill_categories))
    
    # Plot each skill group
    for i, (skill, values) in enumerate(skill_data.items()):
        offset = width * multiplier
        bars = plt.bar(x + offset, values, width, label=skill, color=colors[i])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90)
        
        multiplier += 1
    
    # Add labels and legend
    plt.title("Optimal Skill Development Paths by Career Stage", fontsize=14)
    plt.xlabel("Career Stage", fontsize=12)
    plt.ylabel("Estimated Salary Increase (%)", fontsize=12)
    plt.xticks(x + width * (len(skill_categories) - 1) / 2, stages)
    plt.legend(title="Skill Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved optimal skill development paths to {output_path}")
    
    plt.close()
    
    # Return results
    return development_paths

def main(year=2024):
    """
    Main function to run the model interpretation.
    
    Parameters:
    -----------
    year : int
        Survey year
    """
    logger.info("Running model interpretation for Stack Overflow Developer Survey {}".format(year))
    
    # Load data and models
    model_data = load_data_and_models(year)
    if model_data is None:
        logger.error("Failed to load data and models")
        return
    
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test, best_classifier, best_regressor, feature_names = model_data
    
    # Get current timestamp for file naming
    timestamp = int(time.time())
    
    # Process the classification model
    if best_classifier is not None:
        logger.info("Computing SHAP values for classifier")
        classifier_info = {"model": best_classifier, "model_type": "classifier"}
        classifier_explainer, classifier_shap_values, classifier_shap_values_test = compute_shap_values(
            classifier_info, X_train, X_test, model_type='classifier')
        
        # Create SHAP summary plot for classifier
        shap_summary_path = REPORTS_FIGURES_DIR / f"shap_summary_classification_{timestamp}.png"
        plot_shap_summary(classifier_shap_values_test, X_test, feature_names, 'classifier', shap_summary_path)
        
        # Non-timestamped version for reference
        shap_summary_path_ref = REPORTS_FIGURES_DIR / "shap_summary_classification.png"
        plot_shap_summary(classifier_shap_values_test, X_test, feature_names, 'classifier', shap_summary_path_ref)
        
        # Update latest shap files reference
        with open(REPORTS_DIR / "latest_shap_files.txt", "r") as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            if line.startswith("classification:"):
                updated_lines.append(f"classification:shap_summary_classification_{timestamp}.png\n")
            else:
                updated_lines.append(line)
        
        with open(REPORTS_DIR / "latest_shap_files.txt", "w") as f:
            f.writelines(updated_lines)
    
    # Process the regression model
    if best_regressor is not None:
        logger.info("Computing SHAP values for regressor")
        regressor_info = {"model": best_regressor, "model_type": "regressor"}
        regressor_explainer, regressor_shap_values, regressor_shap_values_test = compute_shap_values(
            regressor_info, X_train, X_test, model_type='regressor')
        
        # Create SHAP summary plot for regressor
        shap_summary_path = REPORTS_FIGURES_DIR / f"shap_summary_regression_{timestamp}.png"
        plot_shap_summary(regressor_shap_values_test, X_test, feature_names, 'regressor', shap_summary_path)
        
        # Non-timestamped version for reference
        shap_summary_path_ref = REPORTS_FIGURES_DIR / "shap_summary_regression.png"
        plot_shap_summary(regressor_shap_values_test, X_test, feature_names, 'regressor', shap_summary_path_ref)
        
        # Update latest shap files reference
        with open(REPORTS_DIR / "latest_shap_files.txt", "r") as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            if line.startswith("regression:"):
                updated_lines.append(f"regression:shap_summary_regression_{timestamp}.png\n")
            else:
                updated_lines.append(line)
        
        with open(REPORTS_DIR / "latest_shap_files.txt", "w") as f:
            f.writelines(updated_lines)
            
        # Map SHAP values to Human Capital Theory framework
        logger.info("Mapping SHAP values to Human Capital Theory framework")
        hct_output_path = REPORTS_FIGURES_DIR / f"hct_framework_analysis_{timestamp}.png"
        hct_results = map_shap_values_to_hct_framework(
            regressor_shap_values_test, X_test, feature_names, 'regressor', hct_output_path)
        
        # Visualize skill development ROI
        logger.info("Visualizing skill development ROI")
        roi_output_path = REPORTS_FIGURES_DIR / f"skill_development_roi_{timestamp}.png"
        roi_results = visualize_skill_development_roi(
            regressor_shap_values_test, X_test, feature_names, 'regressor', roi_output_path)
        
        # Identify optimal skill development paths
        logger.info("Identifying optimal skill development paths")
        paths_output_path = REPORTS_FIGURES_DIR / f"optimal_skill_paths_{timestamp}.png"
        paths_results = identify_optimal_skill_development_paths(
            regressor_shap_values_test, X_test, feature_names, 'regressor', paths_output_path)
        
        # Save the HRD analysis results
        hrd_results = {
            "hct_framework": hct_results,
            "skill_development_roi": roi_results,
            "optimal_paths": paths_results
        }
        
        with open(REPORTS_DIR / f"hrd_analysis_results_{timestamp}.json", "w") as f:
            json.dump(hrd_results, f, indent=2, default=str)
    
    logger.info("Model interpretation complete")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Interpret models using SHAP')
    parser.add_argument('--year', type=int, default=2024, help='Survey year')
    
    args = parser.parse_args()
    
    # Run main function
    main(year=args.year)
