from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None
feature_names = ['FPG', 'OGTT', 'Random_Plasma_Glucose', 'HbA1c']

def generate_synthetic_data(n_samples=500):
    """
    Generate synthetic diabetes dataset with medically realistic values.
    
    Normal ranges:
    - FPG (Fasting Plasma Glucose): < 100 mg/dL (normal), 100-125 (prediabetes), >= 126 (diabetes)
    - OGTT (Oral Glucose Tolerance Test): < 140 mg/dL (normal), 140-199 (prediabetes), >= 200 (diabetes)
    - Random Plasma Glucose: < 140 mg/dL (normal), >= 200 (diabetes if symptomatic)
    - HbA1c: < 5.7% (normal), 5.7-6.4% (prediabetes), >= 6.5% (diabetes)
    """
    np.random.seed(42)
    
    data = []
    
    # Generate diabetic patients (40% of dataset)
    n_diabetic = int(n_samples * 0.4)
    for _ in range(n_diabetic):
        # Diabetic patients - higher values with some variance
        fpg = np.random.normal(150, 25)  # Mean 150, std 25
        ogtt = np.random.normal(220, 35)  # Mean 220, std 35
        random_pg = np.random.normal(210, 30)  # Mean 210, std 30
        hba1c = np.random.normal(7.5, 1.0)  # Mean 7.5%, std 1.0
        
        # Ensure values stay in medically reasonable ranges
        fpg = max(80, min(fpg, 300))
        ogtt = max(100, min(ogtt, 400))
        random_pg = max(100, min(random_pg, 400))
        hba1c = max(4.5, min(hba1c, 14.0))
        
        data.append([fpg, ogtt, random_pg, hba1c, 'Yes'])
    
    # Generate non-diabetic patients (60% of dataset)
    n_non_diabetic = n_samples - n_diabetic
    for _ in range(n_non_diabetic):
        # Non-diabetic patients - lower values with some variance
        fpg = np.random.normal(95, 15)  # Mean 95, std 15
        ogtt = np.random.normal(120, 25)  # Mean 120, std 25
        random_pg = np.random.normal(110, 20)  # Mean 110, std 20
        hba1c = np.random.normal(5.3, 0.5)  # Mean 5.3%, std 0.5
        
        # Ensure values stay in medically reasonable ranges
        fpg = max(60, min(fpg, 125))
        ogtt = max(70, min(ogtt, 180))
        random_pg = max(70, min(random_pg, 180))
        hba1c = max(4.0, min(hba1c, 6.3))
        
        data.append([fpg, ogtt, random_pg, hba1c, 'No'])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['FPG', 'OGTT', 'Random_Plasma_Glucose', 'HbA1c', 'Diagnosed_Diabetes'])
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def train_model():
    """Train the Random Forest model on synthetic data."""
    global model, scaler
    
    print("Generating synthetic dataset...")
    df = generate_synthetic_data(500)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Diabetic cases: {(df['Diagnosed_Diabetes'] == 'Yes').sum()}")
    print(f"Non-diabetic cases: {(df['Diagnosed_Diabetes'] == 'No').sum()}")
    
    # Prepare features and target
    X = df[feature_names].values
    y = (df['Diagnosed_Diabetes'] == 'Yes').astype(int).values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest Classifier
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    print("Model training completed successfully!")

@app.route('/')
def home():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint that accepts JSON data with 4 medical features
    and returns diabetes prediction.
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features
        fpg = float(data.get('FPG', 0))
        ogtt = float(data.get('OGTT', 0))
        random_pg = float(data.get('Random_Plasma_Glucose', 0))
        hba1c = float(data.get('HbA1c', 0))
        
        # Validate input ranges
        if not (50 <= fpg <= 400):
            return jsonify({'error': 'FPG must be between 50 and 400 mg/dL'}), 400
        if not (50 <= ogtt <= 500):
            return jsonify({'error': 'OGTT must be between 50 and 500 mg/dL'}), 400
        if not (50 <= random_pg <= 500):
            return jsonify({'error': 'Random Plasma Glucose must be between 50 and 500 mg/dL'}), 400
        if not (3.0 <= hba1c <= 15.0):
            return jsonify({'error': 'HbA1c must be between 3.0% and 15.0%'}), 400
        
        # Prepare input for prediction
        input_features = np.array([[fpg, ogtt, random_pg, hba1c]])
        
        # Scale the input
        input_scaled = scaler.transform(input_features)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Prepare response
        result = {
            'prediction': 'Diabetic' if prediction == 1 else 'Not Diabetic',
            'confidence': float(prediction_proba[prediction]) * 100,
            'probability_diabetic': float(prediction_proba[1]) * 100,
            'probability_non_diabetic': float(prediction_proba[0]) * 100
        }
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/model-info')
def model_info():
    """Return information about the trained model."""
    if model is None:
        return jsonify({'error': 'Model not trained'}), 500
    
    info = {
        'model_type': 'Random Forest Classifier',
        'n_estimators': model.n_estimators,
        'features': feature_names,
        'feature_importances': {
            feature: float(importance) 
            for feature, importance in zip(feature_names, model.feature_importances_)
        }
    }
    
    return jsonify(info)


# Train model when module loads (outside if __name__)
train_model()

if __name__ == '__main__':
    # This only runs when testing locally
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)