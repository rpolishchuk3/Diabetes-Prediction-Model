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
feature_names = ['FPG', 'OGTT', 'Random_Plasma_Glucose', 'HbA1c']

def load_data_from_csv(csv_file_path='diabetes_data.csv'):
    """
    Load diabetes dataset from CSV file.
    
    Expected CSV format:
    Fasting Plasma Glucose (FPG),Oral Glucose Tolerance Test (OGTT) – 2-hour plasma glucose,Random Plasma Glucose,Hemoglobin A1c (HbA1c),Diagnosed diabetes
    6.1,9.32,11.2,6.58,yes
    """
    try:
        print(f"Loading data from: {csv_file_path}")
        
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        print(f"Original columns: {df.columns.tolist()}")
        
        # Rename columns to match our expected format
        column_mapping = {
            'Fasting Plasma Glucose (FPG)': 'FPG',
            'Oral Glucose Tolerance Test (OGTT) – 2-hour plasma glucose': 'OGTT',
            'Random Plasma Glucose': 'Random_Plasma_Glucose',
            'Hemoglobin A1c (HbA1c)': 'HbA1c',
            'Diagnosed diabetes': 'Diagnosed_Diabetes'
        }
        
        df = df.rename(columns=column_mapping)
        
        print(f"Renamed columns: {df.columns.tolist()}")
        print(f"Dataset shape: {df.shape}")
        print(f"First few rows:\n{df.head()}")
        
        # Convert 'Diagnosed_Diabetes' to consistent format (Yes/No)
        # Handle variations: yes/Yes/YES, no/No/NO
        df['Diagnosed_Diabetes'] = df['Diagnosed_Diabetes'].astype(str).str.strip().str.lower()
        df['Diagnosed_Diabetes'] = df['Diagnosed_Diabetes'].replace({'yes': 'Yes', 'no': 'No'})
        
        # Check for any unexpected values
        unique_values = df['Diagnosed_Diabetes'].unique()
        print(f"Unique values in Diagnosed_Diabetes: {unique_values}")
        
        # Count diabetic vs non-diabetic
        diabetic_count = (df['Diagnosed_Diabetes'] == 'Yes').sum()
        non_diabetic_count = (df['Diagnosed_Diabetes'] == 'No').sum()
        
        print(f"Diabetic cases: {diabetic_count}")
        print(f"Non-diabetic cases: {non_diabetic_count}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print(f"Warning: Missing values found:\n{missing_values[missing_values > 0]}")
            # Drop rows with missing values
            df = df.dropna()
            print(f"After removing missing values, dataset shape: {df.shape}")
        
        # Verify all required columns are present
        required_columns = ['FPG', 'OGTT', 'Random_Plasma_Glucose', 'HbA1c', 'Diagnosed_Diabetes']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
        
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at '{csv_file_path}'")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Please ensure the CSV file is in the same directory as app.py")
        raise
    except Exception as e:
        print(f"ERROR loading CSV file: {str(e)}")
        raise

def train_model(csv_file_path='diabetes_data.csv'):
    """Train the Random Forest model on data from CSV file."""
    global model, scaler
    
    try:
        print("="*50)
        print("Loading dataset from CSV file...")
        print("="*50)
        
        # Load data from CSV
        df = load_data_from_csv(csv_file_path)
        
        # Prepare features and target
        X = df[feature_names].values
        y = (df['Diagnosed_Diabetes'] == 'Yes').astype(int).values
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Feature statistics:\n{pd.DataFrame(X, columns=feature_names).describe()}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Classifier
        print("\nTraining Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"\n{'='*50}")
        print(f"MODEL PERFORMANCE")
        print(f"{'='*50}")
        print(f"Training accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
        print(f"Testing accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nFeature Importance:")
        for idx, row in feature_importance.iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")
        
        print(f"\n{'='*50}")
        print("Model training completed successfully!")
        print(f"{'='*50}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"ERROR DURING MODEL TRAINING")
        print(f"{'='*50}")
        print(f"Error: {str(e)}")
        print(f"\nPlease check:")
        print(f"1. CSV file exists in the same directory as app.py")
        print(f"2. CSV file has the correct column names")
        print(f"3. CSV file has data in the correct format")
        return False

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
        # Check if model is trained
        if model is None or scaler is None:
            return jsonify({'error': 'Model not trained. Please check server logs.'}), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features
        fpg = float(data.get('FPG', 0))
        ogtt = float(data.get('OGTT', 0))
        random_pg = float(data.get('Random_Plasma_Glucose', 0))
        hba1c = float(data.get('HbA1c', 0))
        
        # Validate input ranges (allowing wider ranges for your data)
        if not (0 <= fpg <= 500):
            return jsonify({'error': 'FPG must be between 0 and 500'}), 400
        if not (0 <= ogtt <= 500):
            return jsonify({'error': 'OGTT must be between 0 and 500'}), 400
        if not (0 <= random_pg <= 500):
            return jsonify({'error': 'Random Plasma Glucose must be between 0 and 500'}), 400
        if not (0 <= hba1c <= 20):
            return jsonify({'error': 'HbA1c must be between 0% and 20%'}), 400
        
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

if __name__ == '__main__':
    # Train the model when the application starts
    # Make sure 'diabetes_data.csv' is in the same directory as app.py
    csv_file = 'diabetes_data.csv'
    
    success = train_model(csv_file)
    
    if success:
        # Run the Flask application
        print("\nStarting Flask server...")
        print("Visit http://127.0.0.1:5000 in your browser\n")
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("\nFailed to train model. Server not started.")
        print("Please fix the errors above and try again.")