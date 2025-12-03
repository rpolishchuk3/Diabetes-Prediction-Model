from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None
feature_names = ['FPG', 'OGTT', 'Random_Plasma_Glucose', 'HbA1c']

def load_data_from_csv(csv_file_path='diabetes_dataset_1000.csv'):
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
        df['Diagnosed_Diabetes'] = df['Diagnosed_Diabetes'].astype(str).str.strip().str.lower()
        df['Diagnosed_Diabetes'] = df['Diagnosed_Diabetes'].replace({'yes': 'Yes', 'no': 'No'})
        
        # Check for any unexpected values
        unique_values = df['Diagnosed_Diabetes'].unique()
        print(f"Unique values in Diagnosed_Diabetes: {unique_values}")
    
        diabetic_count = (df['Diagnosed_Diabetes'] == 'Yes').sum()
        non_diabetic_count = (df['Diagnosed_Diabetes'] == 'No').sum()
        
        print(f"Diabetic cases: {diabetic_count}")
        print(f"Non-diabetic cases: {non_diabetic_count}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print(f"Warning: Missing values found:\n{missing_values[missing_values > 0]}")
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

def train_model(csv_file_path='diabetes_dataset_1000.csv'):
    """Train the Random Forest model with aggressive regularization."""
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
        
        # ADD NOISE to create more uncertainty
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, X.shape)
        X_noisy = X * (1 + noise)
        
        print(f"Added 5% random noise to simulate measurement uncertainty")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_noisy, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # VERY CONSERVATIVE Random Forest settings
        print("\nTraining Random Forest with extreme regularization...")
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
        
        # Evaluate the model
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"\n{'='*50}")
        print(f"MODEL PERFORMANCE")
        print(f"{'='*50}")
        print(f"Training accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
        print(f"Testing accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
        
        # Test with known values
        print(f"\n{'='*50}")
        print("PROBABILITY CALIBRATION TEST")
        print(f"{'='*50}")
        
        test_cases = [
            ([82, 108, 95, 4.8], "Clearly Non-Diabetic"),
            ([110, 165, 180, 6.1], "Borderline (SHOULD BE ~50%)"),
            ([130, 200, 210, 6.7], "Uncertain"),
            ([160, 240, 280, 7.5], "Likely Diabetic"),
            ([200, 300, 350, 9.0], "Clearly Diabetic")
        ]
        
        for test_vals, description in test_cases:
            # Extract values
            fpg, ogtt, random_pg, hba1c = test_vals
            
            # Check borderline status
            fpg_borderline = 100 <= fpg <= 130
            ogtt_borderline = 140 <= ogtt <= 210
            random_pg_borderline = 140 <= random_pg <= 210
            hba1c_borderline = 5.7 <= hba1c <= 7.0
            borderline_count = sum([fpg_borderline, ogtt_borderline, random_pg_borderline, hba1c_borderline])
            
            # Add noise
            test_input_noisy = np.array(test_vals) * (1 + np.random.normal(0, 0.05, 4))
            test_input = scaler.transform([test_input_noisy])
            proba = model.predict_proba(test_input)[0]
            
            # Apply smoothing
            MIN_PROB = 0.02
            MAX_PROB = 0.98
            prob_0 = np.clip(proba[0], MIN_PROB, MAX_PROB)
            prob_1 = np.clip(proba[1], MIN_PROB, MAX_PROB)
            
            # Apply borderline adjustment if 2+ markers are borderline
            if borderline_count >= 2:
                uncertainty_factor = 0.6  # ← Change from 0.4 to 0.6
                prob_0 = prob_0 * (1 - uncertainty_factor) + 0.5 * uncertainty_factor
                prob_1 = prob_1 * (1 - uncertainty_factor) + 0.5 * uncertainty_factor
            
            # Normalize
            total = prob_0 + prob_1
            prob_0 = prob_0 / total
            prob_1 = prob_1 / total
            
            print(f"\n{description}: {test_vals}")
            print(f"  Borderline markers: {borderline_count}/4")
            print(f"  Non-Diabetic: {prob_0*100:.1f}% | Diabetic: {prob_1*100:.1f}%")
        
        print(f"\n{'='*50}")
        print("Model training completed successfully!")
        print(f"{'='*50}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"ERROR DURING MODEL TRAINING")
        print(f"{'='*50}")
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
    """Prediction endpoint with smart borderline detection."""
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not trained. Please check server logs.'}), 500
        
        data = request.get_json()
        
        # Extract features
        fpg = float(data.get('FPG', 0))
        ogtt = float(data.get('OGTT', 0))
        random_pg = float(data.get('Random_Plasma_Glucose', 0))
        hba1c = float(data.get('HbA1c', 0))
        
        # Validate input ranges
        if not (0 <= fpg <= 500):
            return jsonify({'error': 'FPG must be between 0 and 500'}), 400
        if not (0 <= ogtt <= 500):
            return jsonify({'error': 'OGTT must be between 0 and 500'}), 400
        if not (0 <= random_pg <= 500):
            return jsonify({'error': 'Random Plasma Glucose must be between 0 and 500'}), 400
        if not (0 <= hba1c <= 20):
            return jsonify({'error': 'HbA1c must be between 0% and 20%'}), 400
        
        # ============================================
        # BORDERLINE DETECTION
        # Medical guidelines for prediabetes/borderline cases
        # ============================================
        
        # Check if values are in prediabetic/borderline ranges
        fpg_borderline = 100 <= fpg <= 130
        ogtt_borderline = 140 <= ogtt <= 210
        random_pg_borderline = 140 <= random_pg <= 210
        hba1c_borderline = 5.7 <= hba1c <= 7.0
        
        # Count how many markers are borderline
        borderline_count = sum([fpg_borderline, ogtt_borderline, random_pg_borderline, hba1c_borderline])
        
        # Prepare input
        input_features = np.array([[fpg, ogtt, random_pg, hba1c]])
        
        # Add measurement noise
        noise = np.random.normal(0, 0.05, input_features.shape)
        input_noisy = input_features * (1 + noise)
        
        # Scale the input
        input_scaled = scaler.transform(input_noisy)
        
        # Make prediction
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # ============================================
        # SMART PROBABILITY ADJUSTMENT
        # ============================================
        
        MIN_PROB = 0.02
        MAX_PROB = 0.98
        
        prob_non_diabetic = np.clip(prediction_proba[0], MIN_PROB, MAX_PROB)
        prob_diabetic = np.clip(prediction_proba[1], MIN_PROB, MAX_PROB)
        
        # If 2+ markers are borderline, force more uncertainty
        if borderline_count >= 2:
            # Pull probabilities toward 50-50 for borderline cases
            uncertainty_factor = 0.6  # ← Change from 0.4 to 0.6
            
            prob_non_diabetic = prob_non_diabetic * (1 - uncertainty_factor) + 0.5 * uncertainty_factor
            prob_diabetic = prob_diabetic * (1 - uncertainty_factor) + 0.5 * uncertainty_factor
        
        # Normalize
        total = prob_non_diabetic + prob_diabetic
        prob_non_diabetic = prob_non_diabetic / total
        prob_diabetic = prob_diabetic / total
        
        # Final prediction
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
csv_file = 'diabetes_dataset_1000.csv'
print("\n" + "="*60)
print("INITIALIZING DIABETES PREDICTION SYSTEM")
print("="*60)

success = train_model(csv_file)

if not success:
    print("\n" + "="*60)
    print("WARNING: Model training failed!")
    print("The server will start but predictions will not work.")
    print("="*60 + "\n")

# This only runs when using 'python app.py' directly
if __name__ == '__main__':
    print("\nStarting Flask development server...")
    print("Visit http://127.0.0.1:5000 in your browser\n")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)