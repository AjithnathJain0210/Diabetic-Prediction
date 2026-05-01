import os

# ==========================================
# Base Directory Identification
# ==========================================
# This identifies the root directory: C:\Users\ajith\Desktop\Diabetic Predictor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# Data Paths (Input & Output)
# ==========================================
# Original clinical data source
CLINICAL_DATA_PATH = r"C:\Users\ajith\Desktop\processed_dataset.xlsx"

# Fingerprint image directories
FINGERPRINT_RAW_DIR = os.path.join(BASE_DIR, 'data', 'fingerprint_images')
FINGERPRINT_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'fingerprint_images')

# Output paths for processed CSV files
CLINICAL_PREPROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'clinical_cleaned.csv')

# The "Seed" data (The 30 real-world integrated samples)
INTEGRATED_FEATURES_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'integrated_features.csv')

# The "Expanded" data (The 300 augmented samples for training)
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'augmented_data.csv')

# ==========================================
# Model & Transformer Paths (The Backend)
# ==========================================
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Standardized to .pkl for compatibility with Module 6 & 7
MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
FEATURES_PKL_PATH = os.path.join(MODELS_DIR, 'features.pkl')

# Secondary transformers (if needed for older logic)
IMPUTER_PATH = os.path.join(MODELS_DIR, 'imputer.pkl')
LABEL_ENCODERS_PATH = os.path.join(MODELS_DIR, 'label_encoders.pkl')

# ==========================================
# Feature Definitions (13 Column Logic)
# ==========================================
CATEGORICAL_FEATURES = [
    'gender', 
    'smoking_status',
    'physical_activity_level', 
    'family_history', 
    'fingerprint_type'
]

NUMERICAL_FEATURES = [
    'age',
    'bmi',
    'blood_pressure_systolic',
    'blood_pressure_diastolic',
    'ridge_count',
    'ridge_density',
    'minutiae_count'
]

# The 'risk_level' is our Target (Y), not a feature (X)

# ==========================================
# Training Hyperparameters (Pruned for 93-95% Accuracy)
# ==========================================
N_ESTIMATORS = 50
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 15
MAX_FEATURES = 0.3
TEST_SIZE = 0.3
RANDOM_STATE = 42

# ==========================================
# ADDITIONAL ENHANCEMENTS (Login & History)
# ==========================================
# This points to the patient history database in your root folder
DB_PATH = os.path.join(BASE_DIR, 'patient_records.db')