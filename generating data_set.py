import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

def generate_realistic_diabetes_data(n_samples=1000):
    """
    Generate realistic diabetes dataset with proper overlap between classes.
    This will create more nuanced predictions instead of 100% confidence.
    """
    
    data = []
    
    # Generate NON-DIABETIC patients (60% = 300 patients)
    n_non_diabetic = int(n_samples * 0.6)
    
    for i in range(n_non_diabetic):
        # 70% clearly normal
        if i < n_non_diabetic * 0.7:
            fpg = np.random.normal(90, 12)  # Mean 90, std 12
            ogtt = np.random.normal(115, 20)  # Mean 115, std 20
            random_pg = np.random.normal(110, 25)  # Mean 110, std 25
            hba1c = np.random.normal(5.1, 0.4)  # Mean 5.1%, std 0.4
        
        # 20% borderline/prediabetic (higher uncertainty)
        elif i < n_non_diabetic * 0.9:
            fpg = np.random.normal(108, 10)
            ogtt = np.random.normal(155, 18)
            random_pg = np.random.normal(165, 22)
            hba1c = np.random.normal(5.9, 0.3)
        
        # 10% high-normal (even more uncertainty)
        else:
            fpg = np.random.normal(118, 8)
            ogtt = np.random.normal(175, 15)
            random_pg = np.random.normal(185, 20)
            hba1c = np.random.normal(6.2, 0.25)
        
        # Ensure values stay in reasonable ranges
        fpg = np.clip(fpg, 70, 125)
        ogtt = np.clip(ogtt, 80, 199)
        random_pg = np.clip(random_pg, 80, 199)
        hba1c = np.clip(hba1c, 4.0, 6.4)
        
        data.append([round(fpg, 2), round(ogtt, 2), round(random_pg, 2), round(hba1c, 2), 'no'])
    
    # Generate DIABETIC patients (40% = 200 patients)
    n_diabetic = n_samples - n_non_diabetic
    
    for i in range(n_diabetic):
        # 70% clearly diabetic
        if i < n_diabetic * 0.7:
            fpg = np.random.normal(165, 25)  # Mean 165, std 25
            ogtt = np.random.normal(250, 35)  # Mean 250, std 35
            random_pg = np.random.normal(280, 40)  # Mean 280, std 40
            hba1c = np.random.normal(8.2, 1.2)  # Mean 8.2%, std 1.2
        
        # 20% early diabetic (overlaps with prediabetic range)
        elif i < n_diabetic * 0.9:
            fpg = np.random.normal(138, 15)
            ogtt = np.random.normal(215, 25)
            random_pg = np.random.normal(230, 30)
            hba1c = np.random.normal(7.0, 0.5)
        
        # 10% borderline diabetic (high uncertainty)
        else:
            fpg = np.random.normal(128, 10)
            ogtt = np.random.normal(195, 20)
            random_pg = np.random.normal(205, 25)
            hba1c = np.random.normal(6.6, 0.3)
        
        # Ensure values stay in reasonable ranges
        fpg = np.clip(fpg, 126, 400)
        ogtt = np.clip(ogtt, 200, 450)
        random_pg = np.clip(random_pg, 200, 450)
        hba1c = np.clip(hba1c, 6.5, 14.0)
        
        data.append([round(fpg, 2), round(ogtt, 2), round(random_pg, 2), round(hba1c, 2), 'yes'])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[
        'Fasting Plasma Glucose (FPG)',
        'Oral Glucose Tolerance Test (OGTT) – 2-hour plasma glucose',
        'Random Plasma Glucose',
        'Hemoglobin A1c (HbA1c)',
        'Diagnosed diabetes'
    ])
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# Generate the dataset
df = generate_realistic_diabetes_data(1000)

# Save to CSV
df.to_csv('diabetes_dataset_1000.csv', index=False)

print("Dataset generated successfully!")
print(f"\nDataset shape: {df.shape}")
print(f"Diabetic cases: {(df['Diagnosed diabetes'] == 'yes').sum()}")
print(f"Non-diabetic cases: {(df['Diagnosed diabetes'] == 'no').sum()}")
print("\nFirst 10 rows:")
print(df.head(10))
print("\nLast 10 rows:")
print(df.tail(10))
print("\nStatistics:")
print(df.describe())
print("\nDataset saved as 'diabetes_dataset_1000.csv'")