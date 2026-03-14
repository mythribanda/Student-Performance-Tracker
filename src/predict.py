"""
Student Performance ML - Prediction Module
Works with ANY column set saved in model.pkl — no hardcoded feature names.
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')


def load_payload(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run train.py first.")
    return joblib.load(path)


def _encode_val(val, feat, le_dict):
    """Label-encode a single value if the feature is categorical."""
    if feat not in le_dict:
        return val
    le = le_dict[feat]
    val_str = str(val)
    if val_str in le.classes_:
        return int(le.transform([val_str])[0])
    # Unknown category → use the most frequent class (index 0 after fit)
    return int(le.transform([le.classes_[0]])[0])


def predict_single(model_path=MODEL_PATH, **kwargs):
    """
    Predict exam score for one student.

    Pass feature values as keyword arguments matching the column names in the
    trained model, e.g.:

        predict_single(Hours_Studied=7, Attendance=85, Sleep_Hours=7,
                       Previous_Scores=75, Motivation_Level='High', ...)

    Any feature not supplied defaults to the median (numeric) or first class
    (categorical) seen during training — so partial inputs are safe.

    Returns
    -------
    float : predicted exam score (clipped 0–100)
    """
    payload  = load_payload(model_path)
    model    = payload['best_model']
    scaler   = payload['scaler']
    le_dict  = payload['le_dict']
    features = payload['feature_names']

    # Build defaults from le_dict / numeric median placeholder (0)
    row = {}
    for feat in features:
        if feat in kwargs:
            row[feat] = _encode_val(kwargs[feat], feat, le_dict)
        elif feat in le_dict:
            # default: first class
            row[feat] = int(le_dict[feat].transform([le_dict[feat].classes_[0]])[0])
        else:
            row[feat] = 0  # numeric default

    X = pd.DataFrame([row])[features]
    if scaler is not None:
        X = scaler.transform(X)

    score = float(model.predict(X)[0])
    return round(float(np.clip(score, 0, 100)), 2)


def predict_batch(df_input, model_path=MODEL_PATH):
    """
    Predict exam scores for a DataFrame of students.
    Missing columns are filled with 0 / first class automatically.
    """
    payload  = load_payload(model_path)
    model    = payload['best_model']
    scaler   = payload['scaler']
    le_dict  = payload['le_dict']
    features = payload['feature_names']

    df = df_input.copy()

    # Encode categoricals
    for col in le_dict:
        if col in df.columns:
            le = le_dict[col]
            df[col] = df[col].apply(
                lambda v: int(le.transform([str(v)])[0])
                if str(v) in le.classes_
                else int(le.transform([le.classes_[0]])[0])
            )

    # Add any missing columns with 0
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0

    X = df[features]
    if scaler is not None:
        X = scaler.transform(X)

    preds = model.predict(X)
    return np.clip(preds, 0, 100).round(2)


if __name__ == '__main__':
    score = predict_single(
        Hours_Studied=7,
        Attendance=85,
        Sleep_Hours=7,
        Previous_Scores=75,
        Motivation_Level='High',
        Internet_Access='Yes',
        Tutoring_Sessions=3,
        Family_Income='Medium',
        Teacher_Quality='High',
        School_Type='Private',
        Parental_Involvement='High',
        Access_to_Resources='High',
        Extracurricular_Activities='Yes',
        Peer_Influence='Positive',
        Physical_Activity=4,
        Learning_Disabilities='No',
        Parental_Education_Level='College',
        Distance_from_Home='Near',
        Gender='Male',
    )
    print(f"Predicted Exam Score: {score}")