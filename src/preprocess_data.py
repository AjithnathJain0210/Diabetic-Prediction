import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from config import CLINICAL_DATA_PATH, CLINICAL_PREPROCESSED_PATH

def preprocess_clinical():
    print("\n" + "="*60)
    print("🚀 MODULE 2: CLINICAL DATA PREPROCESSING (ALL CATEGORIES)")
    print("="*60)

    # 1. Load Data
    if not os.path.exists(CLINICAL_DATA_PATH):
        print(f"❌ Error: Raw file not found at {CLINICAL_DATA_PATH}")
        return
        
    df = pd.read_excel(CLINICAL_DATA_PATH) if CLINICAL_DATA_PATH.endswith('.xlsx') else pd.read_csv(CLINICAL_DATA_PATH)
    initial_shape = df.shape
    print(f"✓ Loaded {initial_shape[0]} clinical records")

    # 2. Normalize Headers (Standardize: 'Family History' -> 'family_history')
    df.columns = [str(c).lower().strip().replace(' ', '_').replace('(', '').replace(')', '') for c in df.columns]

    # 3. Handle Duplicates
    df = df.drop_duplicates().copy()
    removed_dupes = initial_shape[0] - len(df)
    print(f"✓ Removed {removed_dupes} duplicate records")

    # 4. Handle Missing Values (Median Imputation for Numbers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        print(f"✓ Handled missing values using Median Imputation")

    # 5. ENCODING ALL CATEGORICAL DATA (The 2026 Batch Final Logic)
    # We map every category from your acquisition module into integers.
    mapping = {
        'gender': {'female': 1, 'male': 2},
        'ethnicity_indian_regions': {'north_india': 0, 'south_india': 1, 'east_india': 2, 'west_india': 3, 'central_india': 4,
                                    'north india': 0, 'south india': 1, 'east india': 2, 'west india': 3, 'central india': 4},
        'smoking_status': {'non-smoker': 0, 'former_smoker': 1, 'current_smoker': 2, 
                          'former smoker': 1, 'current smoker': 2},
        'physical_activity_level': {'low': 0, 'moderate': 1, 'high': 2},
        'family_history': {'no': 0, 'yes': 1},
        'risk_level': {'low': 0, 'medium': 1, 'high': 2}
    }

    print("✓ Encoding Categorical Features (Whole Numbers Only):")
    for category, m_map in mapping.items():
        # This checks for the best column name match
        actual_col = next((c for c in df.columns if category in c), None)
        
        if actual_col:
            # Clean the text and apply the map
            df[actual_col] = df[actual_col].astype(str).str.strip().str.lower().replace(' ', '_', regex=True).map(m_map).fillna(0).astype(int)
            print(f"   - {actual_col}: Encoded Successfully")
        else:
            print(f"   ⚠️ Warning: {category} column not found in dataset!")

    # 6. Final Cleaning: Keeping ID and Age Raw
    for col in ['id', 'age']:
        if col in df.columns:
            df[col] = df[col].astype(float).astype(int)

    # 7. Save and Summary (Terminal Style)
    os.makedirs(os.path.dirname(CLINICAL_PREPROCESSED_PATH), exist_ok=True)
    df.to_csv(CLINICAL_PREPROCESSED_PATH, index=False)
    
    print("\n" + "-"*60)
    print(f"Clinical Data Shape: {df.shape}")
    print("First 5 rows (Processed Multi-Category View):")
    
    # Selecting the specific columns you asked for
    preview_cols = ['id', 'gender', 'ethnicity_indian_regions', 'age', 'smoking_status', 'physical_activity_level', 'family_history', 'risk_level']
    existing_preview = [c for c in preview_cols if c in df.columns]
    print(df[existing_preview].head())
    
    print(f"\n[{len(df)} rows x {len(df.columns)} columns]")
    print(f"✓ Preprocessed file saved to: {CLINICAL_PREPROCESSED_PATH}")
    print("="*60 + "\n")

if __name__ == "__main__":
    preprocess_clinical()