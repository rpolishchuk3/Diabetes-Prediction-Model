from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)

model = None
scaler = None

feature_names = [
    'gender', 'age', 'hypertension', 'heart_disease',
    'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'
]

GENDER_MAP = {'female': 0, 'male': 1, 'other': 2}
SMOKING_MAP = {
    'never': 0,
    'no info': 1,
    'current': 2,
    'former': 3,
    'ever': 4,
    'not current': 5
}


def encode_features(df):
    df = df.copy()
    df['gender'] = df['gender'].str.strip().str.lower().map(GENDER_MAP).fillna(2)
    df['smoking_history'] = df['smoking_history'].str.strip().str.lower().map(SMOKING_MAP).fillna(1)
    return df


def load_data_from_csv(csv_file_path='diabetes_prediction_dataset.csv'):
    try:
        print(f"Loading data from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        print(f"Columns: {df.columns.tolist()}")
        print(f"Dataset shape: {df.shape}")

        missing_values = df.isnull().sum()
        if missing_values.any():
            df = df.dropna()
            print(f"After removing missing values: {df.shape}")

        required_columns = feature_names + ['diabetes']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        print(f"Diabetic cases:     {(df['diabetes'] == 1).sum()}")
        print(f"Non-diabetic cases: {(df['diabetes'] == 0).sum()}")
        return df

    except FileNotFoundError:
        print(f"ERROR: CSV file not found at '{csv_file_path}'")
        print(f"Current working directory: {os.getcwd()}")
        raise
    except Exception as e:
        print(f"ERROR loading CSV: {str(e)}")
        raise


def train_model(csv_file_path='diabetes_prediction_dataset.csv'):
    global model, scaler

    try:
        print("=" * 50)
        print("Loading dataset...")
        print("=" * 50)

        df = load_data_from_csv(csv_file_path)
        df = encode_features(df)

        X = df[feature_names].values
        y = df['diabetes'].values

        # ── NO noise added ── the data is clean enough
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Fix class imbalance with class_weight, use a stronger model
        print("\nTraining Random Forest (balanced)...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            class_weight='balanced',   # ← handles 9:1 imbalance
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_s, y_train)

        train_score = model.score(X_train_s, y_train)
        test_score  = model.score(X_test_s,  y_test)
        print(f"\nTraining accuracy: {train_score*100:.2f}%")
        print(f"Testing accuracy:  {test_score*100:.2f}%")

        # Sanity-check with the 5 test cases
        print(f"\n{'='*50}")
        print("SANITY CHECK")
        print(f"{'='*50}")
        sanity_cases = [
            ({'gender':'female','age':22,'hypertension':0,'heart_disease':0,
              'smoking_history':'never','bmi':21.5,'HbA1c_level':4.8,'blood_glucose_level':85},
             "1 – Definitely NOT Diabetic"),
            ({'gender':'male','age':40,'hypertension':0,'heart_disease':0,
              'smoking_history':'former','bmi':27.0,'HbA1c_level':5.9,'blood_glucose_level':120},
             "2 – Probably Not Diabetic"),
            ({'gender':'male','age':52,'hypertension':1,'heart_disease':0,
              'smoking_history':'current','bmi':30.5,'HbA1c_level':6.2,'blood_glucose_level':155},
             "3 – Borderline"),
            ({'gender':'female','age':60,'hypertension':1,'heart_disease':0,
              'smoking_history':'former','bmi':34.0,'HbA1c_level':6.8,'blood_glucose_level':190},
             "4 – Probably Diabetic"),
            ({'gender':'male','age':68,'hypertension':1,'heart_disease':1,
              'smoking_history':'current','bmi':40.0,'HbA1c_level':9.5,'blood_glucose_level':280},
             "5 – Definitely Diabetic"),
        ]
        for case_dict, label in sanity_cases:
            row = encode_features(pd.DataFrame([case_dict]))
            scaled = scaler.transform(row[feature_names].values)
            proba = model.predict_proba(scaled)[0]
            pred = "Diabetic" if proba[1] > 0.5 else "Not Diabetic"
            print(f"{label}: {pred} | Non-D: {proba[0]*100:.1f}%  D: {proba[1]*100:.1f}%")

        print(f"\n{'='*50}")
        print("Model training complete!")
        print(f"{'='*50}\n")
        return True

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not trained. Please check server logs.'}), 500

        data = request.get_json()

        gender          = str(data.get('gender', 'other')).strip()
        age             = float(data.get('age', 0))
        hypertension    = int(data.get('hypertension', 0))
        heart_disease   = int(data.get('heart_disease', 0))
        smoking_history = str(data.get('smoking_history', 'No Info')).strip()
        bmi             = float(data.get('bmi', 0))
        hba1c           = float(data.get('HbA1c_level', 0))
        blood_glucose   = float(data.get('blood_glucose_level', 0))

        # Validation
        if not (0 <= age <= 120):
            return jsonify({'error': 'Age must be between 0 and 120'}), 400
        if not (10 <= bmi <= 70):
            return jsonify({'error': 'BMI must be between 10 and 70'}), 400
        if not (0 <= hba1c <= 20):
            return jsonify({'error': 'HbA1c must be between 0 and 20'}), 400
        if not (0 <= blood_glucose <= 500):
            return jsonify({'error': 'Blood glucose must be between 0 and 500'}), 400
        if hypertension not in (0, 1):
            return jsonify({'error': 'Hypertension must be 0 or 1'}), 400
        if heart_disease not in (0, 1):
            return jsonify({'error': 'Heart disease must be 0 or 1'}), 400

        gender_enc  = GENDER_MAP.get(gender.lower(), 2)
        smoking_enc = SMOKING_MAP.get(smoking_history.lower(), 1)

        input_features = np.array([[
            gender_enc, age, hypertension, heart_disease,
            smoking_enc, bmi, hba1c, blood_glucose
        ]])

        input_scaled = scaler.transform(input_features)
        proba = model.predict_proba(input_scaled)[0]

        prob_non_diabetic = float(proba[0])
        prob_diabetic     = float(proba[1])

        final_prediction = 1 if prob_diabetic > 0.5 else 0
        confidence = prob_diabetic if final_prediction == 1 else prob_non_diabetic

        result = {
            'prediction':             'Diabetic' if final_prediction == 1 else 'Not Diabetic',
            'confidence':             round(confidence * 100, 1),
            'probability_diabetic':   round(prob_diabetic * 100, 1),
            'probability_non_diabetic': round(prob_non_diabetic * 100, 1)
        }
        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


# ── Train on startup ──
csv_file = 'diabetes_prediction_dataset.csv'
print("\n" + "=" * 60)
print("INITIALIZING DIABETES PREDICTION SYSTEM")
print("=" * 60)

success = train_model(csv_file)

if not success:
    print("\nWARNING: Model training failed! Predictions will not work.\n")

if __name__ == '__main__':
    print("\nStarting Flask development server...")
    print("Visit http://127.0.0.1:5000 in your browser\n")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)