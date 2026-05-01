import pandas as pd
import os
from config import CLINICAL_PREPROCESSED_PATH, INTEGRATED_FEATURES_PATH, FINGERPRINT_PROCESSED_DIR

def integrate_data():
    print("\n" + "="*60)
    print("🚀 MODULE 4: POSITIONAL FORCE-JOIN (30 SAMPLES)")
    print("="*60)

    # 1. Load Files
    clinical_df = pd.read_csv(CLINICAL_PREPROCESSED_PATH).head(30).reset_index(drop=True)
    finger_path = os.path.join(FINGERPRINT_PROCESSED_DIR, 'extracted_features.csv')
    finger_df = pd.read_csv(finger_path).head(30).reset_index(drop=True)

    # 2. Clean Fingerprint Headers (lowercase, no spaces)
    finger_df.columns = [c.lower().strip().replace(' ', '_') for c in finger_df.columns]
    
    # 3. Drop 'id' from fingerprint so it doesn't duplicate
    if 'id' in finger_df.columns:
        finger_df = finger_df.drop(columns=['id'])

    # 4. Remove Biometric columns from Clinical if they exist (Clean Slate)
    biometric_keys = ['fingerprint_type', 'ridge_count', 'ridge_density', 'minutiae_count']
    clinical_df = clinical_df.drop(columns=[c for c in biometric_keys if c in clinical_df.columns], errors='ignore')

    # 5. THE FORCE JOIN (Axis 1)
    # This glues them side-by-side regardless of IDs
    final_df = pd.concat([clinical_df, finger_df], axis=1)

    # 6. Strict 13-Column Target List (Ethnicity, Waist, Hip already removed)
    target_features = [
        'gender', 'age', 'bmi', 'blood_pressure_systolic', 
        'blood_pressure_diastolic', 'smoking_status', 
        'physical_activity_level', 'family_history', 
        'fingerprint_type', 'ridge_count', 'ridge_density', 'minutiae_count',
        'risk_level'
    ]

    # 7. Map Fingerprint Type (Arch=0, Loop=1, Whorl=2)
    if 'fingerprint_type' in final_df.columns:
        f_map = {'arch': 0, 'loop': 1, 'whorl': 2}
        final_df['fingerprint_type'] = final_df['fingerprint_type'].astype(str).str.lower().str.strip().map(f_map).fillna(1).astype(int)

    # 8. Reorder and Filter to exactly 13 columns
    existing_cols = [c for c in target_features if c in final_df.columns]
    final_df = final_df[existing_cols].copy()

    # 9. Final Data Type Casting
    int_cols = ['gender', 'age', 'smoking_status', 'physical_activity_level', 'family_history', 'fingerprint_type', 'risk_level']
    for col in int_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0).astype(float).round().astype(int)

    # 10. Save
    os.makedirs(os.path.dirname(INTEGRATED_FEATURES_PATH), exist_ok=True)
    final_df.to_csv(INTEGRATED_FEATURES_PATH, index=False)

    print(f"✓ Integration Results: {final_df.shape[0]} rows x {final_df.shape[1]} columns.")
    
    if final_df.shape[1] == 13:
        print("✅ SUCCESS: 30 samples integrated with all 13 features!")
        print(f"📊 Headers: {final_df.columns.tolist()}")
    else:
        print(f"❌ ERROR: Found {final_df.shape[1]} features. Missing: {set(target_features) - set(final_df.columns)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    integrate_data()