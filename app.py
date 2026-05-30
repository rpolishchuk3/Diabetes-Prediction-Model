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

# Global variables for model and scaler
model = None
scaler = None

feature_names = [
    'gender', 'age', 'hypertension', 'heart_disease',
    'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'
]

# Encoding maps for categorical columns
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
    """Encode categorical columns to numeric."""
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
        print(f"First few rows:\n{df.head()}")

        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print(f"Warning: Missing values found:\n{missing_values[missing_values > 0]}")
            df = df.dropna()
            print(f"After removing missing values, dataset shape: {df.shape}")

        # Verify required columns
        required_columns = feature_names + ['diabetes']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        diabetic_count = (df['diabetes'] == 1).sum()
        non_diabetic_count = (df['diabetes'] == 0).sum()
        print(f"Diabetic cases: {diabetic_count}")
        print(f"Non-diabetic cases: {non_diabetic_count}")

        return df

    except FileNotFoundError:
        print(f"ERROR: CSV file not found at '{csv_file_path}'")
        print(f"Current working directory: {os.getcwd()}")
        raise
    except Exception as e:
        print(f"ERROR loading CSV file: {str(e)}")
        raise


def train_model(csv_file_path='diabetes_prediction_dataset.csv'):
    """Train the Random Forest model."""
    global model, scaler

    try:
        print("=" * 50)
        print("Loading dataset from CSV file...")
        print("=" * 50)

        df = load_data_from_csv(csv_file_path)
        df = encode_features(df)

        X = df[feature_names].values
        y = df['diabetes'].values

        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")

        # Add noise to simulate measurement uncertainty
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, X.shape)
        X_noisy = X * (1 + noise)
        print("Added 5% random noise to simulate measurement uncertainty")

        X_train, X_test, y_train, y_test = train_test_split(
            X_noisy, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"\nTraining set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("\nTraining Random Forest with regularization...")
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            min_samples_split=100,
            min_samples_leaf=50,
            max_features=2,
            min_impurity_decrease=0.01,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_scaled, y_train)

        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)

        print(f"\n{'=' * 50}")
        print("MODEL PERFORMANCE")
        print(f"{'=' * 50}")
        print(f"Training accuracy: {train_score:.4f} ({train_score * 100:.2f}%)")
        print(f"Testing accuracy: {test_score:.4f} ({test_score * 100:.2f}%)")

        # Quick sanity-check predictions
        print(f"\n{'=' * 50}")
        print("PROBABILITY SANITY CHECK")
        print(f"{'=' * 50}")

        test_cases = [
            # gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level
            ({'gender': 'female', 'age': 25, 'hypertension': 0, 'heart_disease': 0,
              'smoking_history': 'never', 'bmi': 22.0, 'HbA1c_level': 5.0, 'blood_glucose_level': 90},
             "Clearly Non-Diabetic"),
            ({'gender': 'male', 'age': 50, 'hypertension': 1, 'heart_disease': 0,
              'smoking_history': 'former', 'bmi': 28.5, 'HbA1c_level': 6.1, 'blood_glucose_level': 140},
             "Borderline"),
            ({'gender': 'male', 'age': 60, 'hypertension': 1, 'heart_disease': 1,
              'smoking_history': 'current', 'bmi': 35.0, 'HbA1c_level': 7.5, 'blood_glucose_level': 200},
             "Likely Diabetic"),
        ]

        for case_dict, description in test_cases:
            row = pd.DataFrame([case_dict])
            row = encode_features(row)
            vals = row[feature_names].values
            vals_noisy = vals * (1 + np.random.normal(0, 0.05, vals.shape))
            scaled = scaler.transform(vals_noisy)
            proba = model.predict_proba(scaled)[0]
            print(f"\n{description}:")
            print(f"  Non-Diabetic: {proba[0] * 100:.1f}% | Diabetic: {proba[1] * 100:.1f}%")

        print(f"\n{'=' * 50}")
        print("Model training completed successfully!")
        print(f"{'=' * 50}\n")
        return True

    except Exception as e:
        print(f"\n{'=' * 50}")
        print("ERROR DURING MODEL TRAINING")
        print(f"{'=' * 50}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/')
def home():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not trained. Please check server logs.'}), 500

        data = request.get_json()

        # Extract and validate inputs
        gender = str(data.get('gender', 'other')).strip()
        age = float(data.get('age', 0))
        hypertension = int(data.get('hypertension', 0))
        heart_disease = int(data.get('heart_disease', 0))
        smoking_history = str(data.get('smoking_history', 'No Info')).strip()
        bmi = float(data.get('bmi', 0))
        hba1c = float(data.get('HbA1c_level', 0))
        blood_glucose = float(data.get('blood_glucose_level', 0))

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

        # Encode categoricals
        gender_enc = GENDER_MAP.get(gender.lower(), 2)
        smoking_enc = SMOKING_MAP.get(smoking_history.lower(), 1)

        input_features = np.array([[
            gender_enc, age, hypertension, heart_disease,
            smoking_enc, bmi, hba1c, blood_glucose
        ]])

        # Borderline detection on clinical markers
        hba1c_borderline = 5.7 <= hba1c <= 7.0
        glucose_borderline = 140 <= blood_glucose <= 199
        bmi_borderline = 25 <= bmi <= 35
        borderline_count = sum([hba1c_borderline, glucose_borderline, bmi_borderline])

        # Add measurement noise
        noise = np.random.normal(0, 0.05, input_features.shape)
        input_noisy = input_features * (1 + noise)

        input_scaled = scaler.transform(input_noisy)
        prediction_proba = model.predict_proba(input_scaled)[0]

        MIN_PROB, MAX_PROB = 0.02, 0.98
        prob_non_diabetic = np.clip(prediction_proba[0], MIN_PROB, MAX_PROB)
        prob_diabetic = np.clip(prediction_proba[1], MIN_PROB, MAX_PROB)

        # Pull borderline cases toward 50-50
        if borderline_count >= 2:
            uncertainty_factor = 0.6
            prob_non_diabetic = prob_non_diabetic * (1 - uncertainty_factor) + 0.5 * uncertainty_factor
            prob_diabetic = prob_diabetic * (1 - uncertainty_factor) + 0.5 * uncertainty_factor

        total = prob_non_diabetic + prob_diabetic
        prob_non_diabetic /= total
        prob_diabetic /= total

        final_prediction = 1 if prob_diabetic > 0.5 else 0
        confidence = prob_diabetic if final_prediction == 1 else prob_non_diabetic

        result = {
            'prediction': 'Diabetic' if final_prediction == 1 else 'Not Diabetic',
            'confidence': float(confidence * 100),
            'probability_diabetic': float(prob_diabetic * 100),
            'probability_non_diabetic': float(prob_non_diabetic * 100)
        }

        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


# Train the model when the module loads
csv_file = 'diabetes_prediction_dataset.csv'
print("\n" + "=" * 60)
print("INITIALIZING DIABETES PREDICTION SYSTEM")
print("=" * 60)

success = train_model(csv_file)

if not success:
    print("\n" + "=" * 60)
    print("WARNING: Model training failed!")
    print("The server will start but predictions will not work.")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    print("\nStarting Flask development server...")
    print("Visit http://127.0.0.1:5000 in your browser\n")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)