import os
import warnings
import numpy as np
import pandas as pd
import joblib

from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

RAW_FEATURES = [
    'gender', 'age', 'hypertension', 'heart_disease',
    'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'
]

TARGET = 'diabetes'

MODEL_PATH = 'diabetes_model.pkl'


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['gender']          = df['gender'].str.strip().str.lower()
    df['smoking_history'] = df['smoking_history'].str.strip().str.lower()

    df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=False)
    return df


def align_columns(df: pd.DataFrame, train_columns: list) -> pd.DataFrame:
    for col in train_columns:
        if col not in df.columns:
            df[col] = 0
    return df[train_columns]


def train_model(csv_path: str = 'diabetes_prediction_dataset.csv') -> bool:
    global model, train_columns

    try:
        df = pd.read_csv(csv_path).dropna()
    except FileNotFoundError:
        print(f"ERROR: '{csv_path}' not found (cwd={os.getcwd()})")
        return False

    print(f"Dataset: {df.shape[0]} rows  |  "
          f"Diabetic: {df[TARGET].sum()}  "
          f"Non-diabetic: {(df[TARGET] == 0).sum()}")

    X = preprocess(df[RAW_FEATURES])
    y = df[TARGET].values
    train_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Evaluation (test set)".center(55, "="))
    print(classification_report(y_test, y_pred, target_names=['Not Diabetic', 'Diabetic']))

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}\n")

    importances = pd.Series(model.feature_importances_, index=train_columns)
    print("Top features".center(55, "="))
    print(importances.nlargest(8).to_string())
    print()

    joblib.dump({'model': model, 'train_columns': train_columns}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}\n")
    return True


def predict_one(input_dict: dict) -> dict:
    row = pd.DataFrame([input_dict])
    row = preprocess(row)
    row = align_columns(row, train_columns)

    proba = model.predict_proba(row)[0]
    prob_diabetic     = float(proba[1])
    prob_non_diabetic = float(proba[0])
    prediction        = 'Diabetic' if prob_diabetic >= 0.5 else 'Not Diabetic'

    confidence = max(prob_diabetic, prob_non_diabetic) * 100

    return {
        'prediction': prediction,
        'probability_diabetic': round(prob_diabetic * 100, 1),
        'probability_non_diabetic': round(prob_non_diabetic * 100, 1),
        'confidence': round(confidence, 1)
    }


VALID_GENDERS  = {'male', 'female', 'other'}
VALID_SMOKING  = {'never', 'no info', 'current', 'former', 'ever', 'not current'}

def validate(data: dict) -> str | None:
    if str(data.get('gender', '')).strip().lower() not in VALID_GENDERS:
        return f"gender must be one of {sorted(VALID_GENDERS)}"
    if str(data.get('smoking_history', '')).strip().lower() not in VALID_SMOKING:
        return f"smoking_history must be one of {sorted(VALID_SMOKING)}"
    try:
        if not (0 <= float(data['age']) <= 120):
            return "age must be 0–120"
        if not (10 <= float(data['bmi']) <= 70):
            return "bmi must be 10–70"
        if not (0 <= float(data['HbA1c_level']) <= 20):
            return "HbA1c_level must be 0–20"
        if not (0 <= float(data['blood_glucose_level']) <= 500):
            return "blood_glucose_level must be 0–500"
        if int(data['hypertension']) not in (0, 1):
            return "hypertension must be 0 or 1"
        if int(data['heart_disease']) not in (0, 1):
            return "heart_disease must be 0 or 1"
    except (KeyError, ValueError, TypeError) as exc:
        return f"Invalid or missing field: {exc}"
    return None


app = Flask(__name__)
model        = None
train_columns = []


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not ready'}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    error = validate(data)
    if error:
        return jsonify({'error': error}), 400

    try:
        result = predict_one(data)
        return jsonify(result)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


print("DIABETES PREDICTION - STARTING".center(55, "="))

if os.path.exists(MODEL_PATH):
    artefact      = joblib.load(MODEL_PATH)
    model         = artefact['model']
    train_columns = artefact['train_columns']
    print(f"Loaded saved model from {MODEL_PATH}")
else:
    success = train_model()
    if not success:
        print("WARNING: training failed — /predict will return 503")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\nVisit http://127.0.0.1:{port} in your browser\n")
    app.run(debug=False, host='0.0.0.0', port=port)