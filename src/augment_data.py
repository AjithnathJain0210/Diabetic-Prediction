import pandas as pd
import numpy as np
import os

# Internal path fallback
PROCESSED_DIR = r"C:\Users\ajith\Desktop\Diabetic Predictor\data\processed"
INTEGRATED_PATH = os.path.join(PROCESSED_DIR, 'integrated_features.csv')
AUGMENTED_PATH = os.path.join(PROCESSED_DIR, 'augmented_data.csv')

def augment_data():
    print("\n" + "="*60)
    print("📈 MODULE 5: STRICT MULTIMODAL AUGMENTATION")
    print("="*60)

    if not os.path.exists(INTEGRATED_PATH):
        print(f"❌ Error: {INTEGRATED_PATH} not found!")
        return

    df = pd.read_csv(INTEGRATED_PATH)
    augmented_list = [df] 
    
    # 1. Define feature types for realistic noise
    numeric_cols = ['age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
                    'ridge_count', 'ridge_density', 'minutiae_count']

    # 2. Generate Variations
    for _ in range(9):
        temp_df = df.copy()
        for col in numeric_cols:
            if col in temp_df.columns:
                std_val = temp_df[col].std() if temp_df[col].std() > 0 else 1.0
                noise = np.random.normal(0, std_val * 0.07, size=len(temp_df))
                temp_df[col] = temp_df[col] + noise
        augmented_list.append(temp_df)

    final_df = pd.concat(augmented_list, ignore_index=True)

    # 3. 🎯 STRICT RECTIFICATION OF DECIMALS
    # Force these to be WHOLE NUMBERS (Integers)
    whole_num_cols = [
        'gender', 'age', 'smoking_status', 'physical_activity_level', 
        'family_history', 'fingerprint_type', 'ridge_count', 
        'minutiae_count', 'risk_level'
    ]
    
    for col in whole_num_cols:
        if col in final_df.columns:
            # Round first, then convert to Int
            final_df[col] = final_df[col].round().astype(int)

    # Force these to be CLEAN DECIMALS (1 decimal place)
    decimal_cols = ['bmi', 'ridge_density', 'blood_pressure_systolic', 'blood_pressure_diastolic']
    for col in decimal_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].round(1)

    # 4. Final Cleanup (No negatives, No zero age)
    final_df['age'] = final_df['age'].clip(lower=18)
    final_df = final_df.clip(lower=0)

    # 5. Save
    final_df.to_csv(AUGMENTED_PATH, index=False)
    
    print(f"✓ Results: {len(final_df)} samples ready.")
    print(f"✓ Decimals removed from Fingerprint Features.")
    print(f"✓ Training file updated: {AUGMENTED_PATH}")
    print("="*60 + "\n")

if __name__ == "__main__":
    augment_data()