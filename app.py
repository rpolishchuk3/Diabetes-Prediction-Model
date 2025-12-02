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

# Global variables
model = None
scaler = None
feature_names = ['FPG', 'OGTT', 'Random_Plasma_Glucose', 'HbA1c']

def load_data_from_csv(csv_file_path='diabetes_dataset_500.csv'):
    try:
        print(f"Loading data from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        # Normalize column names to avoid whitespace issues
        df.columns = df.columns.str.strip()
        
        # Robust column mapping
        column_mapping = {
            'Fasting Plasma Glucose (FPG)': 'FPG',
            'Oral Glucose Tolerance Test (OGTT) – 2-hour plasma glucose': 'OGTT',
            'Random Plasma Glucose': 'Random_Plasma_Glucose',
            'Hemoglobin A1c (HbA1c)': 'HbA1c',
            'Diagnosed diabetes': 'Diagnosed_Diabetes'
        }
        
        # Rename only if columns exist
        df = df.rename(columns=column_mapping)
        
        # Clean the Target Column (Handle Yes/No, 1/0, Pos/Neg)
        df['Diagnosed_Diabetes'] = df['Diagnosed_Diabetes'].astype(str).str.strip().str.lower()
        
        # Map various "positive" indicators to 1 and "negative" to 0
        diagnosis_map = {
            'yes': 1, 'no': 0,
            'positive': 1, 'negative': 0,
            '1': 1, '0': 0,
            '1.0': 1, '0.0': 0
        }
        
        df['target'] = df['Diagnosed_Diabetes'].map(diagnosis_map)
        
        # Remove rows where target couldn't be mapped
        df = df.dropna(subset=['target'])
        
        diabetic_count = (df['target'] == 1).sum()
        non_diabetic_count = (df['target'] == 0).sum()
        
        print(f"Data Loaded -> Diabetic: {diabetic_count}, Non-Diabetic: {non_diabetic_count}")
        
        if diabetic_count == 0 or non_diabetic_count == 0:
            print("CRITICAL WARNING: Dataset contains only one class. Model will produce constant predictions.")
        
        return df
        
    except Exception as e:
        print(f"ERROR loading CSV file: {str(e)}")
        raise

def train_model(csv_file_path='diabetes_dataset_500.csv'):
    global model, scaler
    try:
        df = load_data_from_csv(csv_file_path)
        
        X = df[feature_names].values
        y = df['target'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        score = model.score(X_test_scaled, y_test)
        print(f"Model trained successfully. Accuracy: {score:.2f}")
        return True
        
    except Exception as e:
        print(f"Training Failed: {str(e)}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not trained'}), 500
        
        # 1. Get the raw JSON
        data = request.get_json()
        print(f"\n--- INCOMING REQUEST ---")
        print(f"Raw JSON received: {data}")  # Debugging line
        
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        # 2. Normalize keys to lowercase to ensure matching
        # This fixes the issue if Frontend sends 'fpg' but we expect 'FPG'
        data_lower = {k.lower(): v for k, v in data.items()}
        
        # 3. Extract features safely using lowercase keys
        try:
            fpg = float(data_lower.get('fpg', 0))
            ogtt = float(data_lower.get('ogtt', 0))
            random_pg = float(data_lower.get('random_plasma_glucose', 0))
            hba1c = float(data_lower.get('hba1c', 0))
        except ValueError:
            return jsonify({'error': 'Inputs must be numbers'}), 400

        print(f"Parsed inputs -> FPG: {fpg}, OGTT: {ogtt}, RPG: {random_pg}, HbA1c: {hba1c}")

        # Check for the "Zero Input" trap
        if fpg == 0 and ogtt == 0 and random_pg == 0 and hba1c == 0:
            print("WARNING: All inputs are 0. Possible key mismatch or empty input.")

        # 4. Predict
        input_features = np.array([[fpg, ogtt, random_pg, hba1c]])
        input_scaled = scaler.transform(input_features)
        
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # 5. Handle probability array safely
        # Note: prediction_proba is [prob_class_0, prob_class_1]
        prob_non_diabetic = float(prediction_proba[0]) * 100
        prob_diabetic = float(prediction_proba[1]) * 100
        
        result = {
            'prediction': 'Diabetic' if prediction == 1 else 'Not Diabetic',
            'confidence': prob_diabetic if prediction == 1 else prob_non_diabetic,
            'probability_diabetic': prob_diabetic,
            'probability_non_diabetic': prob_non_diabetic
        }
        
        print(f"Result sent: {result}")
        return jsonify(result)
    
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

# Initialization
csv_file = 'diabetes_dataset_500.csv'
if os.path.exists(csv_file):
    train_model(csv_file)
else:
    print(f"WARNING: {csv_file} not found. Model not trained.")

if __name__ == '__main__':
    app.run(debug=True, port=5000)