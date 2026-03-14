"""
Student Performance ML - Training Script
Trains multiple models and saves the best one.
"""

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'StudentPerformanceFactors.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)
TARGET = 'Exam_Score'


def load_and_preprocess(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET])
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    feature_cols = [c for c in df.columns if c != TARGET]
    X = df[feature_cols]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, feature_cols, le_dict, df


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        'MAE':  round(mean_absolute_error(y_test, preds), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        'R2':   round(r2_score(y_test, preds), 4),
    }


def get_models():
    models = {
        'Linear Regression':  LinearRegression(),
        'Decision Tree':      DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest':      RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        'Gradient Boosting':  GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42),
        'MLP Neural Network': MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42),
    }
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(n_estimators=150, learning_rate=0.1, random_state=42, verbosity=0)
    except ImportError:
        print("  XGBoost not installed — skipping.")
    return models


def train_all(X_train, X_test, y_train, y_test):
    models = get_models()
    results = {}
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    for name, model in models.items():
        print(f"  Training {name}...", end=' ', flush=True)
        needs_scaling = ('MLP' in name or 'Linear' in name)
        if needs_scaling:
            model.fit(X_train_sc, y_train)
            metrics = evaluate(model, X_test_sc, y_test)
            results[name] = {'model': model, 'scaler': scaler, 'metrics': metrics}
        else:
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)
            results[name] = {'model': model, 'scaler': None, 'metrics': metrics}
        print(f"R²={metrics['R2']:.3f}  MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}")
    return results


def get_feature_importance(model, X_train, y_train, feature_names):
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
    return dict(zip(feature_names, result.importances_mean))


def main():
    print("=" * 60)
    print("  Student Performance ML — Training Pipeline")
    print("=" * 60)
    print("\n[1/4] Loading & preprocessing data...")
    X_train, X_test, y_train, y_test, feature_cols, le_dict, df = load_and_preprocess()
    print(f"  Features : {feature_cols}")
    print(f"  Train    : {X_train.shape}  |  Test : {X_test.shape}")
    print("\n[2/4] Training models...")
    results = train_all(X_train, X_test, y_train, y_test)
    print("\n[3/4] Selecting best model (by R²)...")
    best_name = max(results, key=lambda k: results[k]['metrics']['R2'])
    best_info = results[best_name]
    print(f"  Best model : {best_name}  R²={best_info['metrics']['R2']}")
    tree_models = [k for k in results if k in ('Random Forest','Gradient Boosting','XGBoost','Decision Tree')]
    fi_name = best_name if best_name in tree_models else (tree_models[0] if tree_models else None)
    fi = get_feature_importance(results[fi_name]['model'], X_train, y_train, feature_cols) if fi_name else {}
    print("\n[4/4] Saving artifacts...")
    payload = {
        'best_model_name':    best_name,
        'best_model':         best_info['model'],
        'scaler':             best_info['scaler'],
        'all_results':        {k: v['metrics'] for k, v in results.items()},
        'feature_names':      feature_cols,
        'feature_importance': fi,
        'le_dict':            le_dict,
        'df_shape':           df.shape,
    }
    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    joblib.dump(payload, model_path)
    print(f"  Saved → {model_path}")
    print("\n" + "=" * 60 + "\n  Training complete!\n" + "=" * 60)
    return payload


if __name__ == '__main__':
    main()