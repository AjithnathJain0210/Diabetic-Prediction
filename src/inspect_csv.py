import pandas as pd
import os
from config import FINGERPRINT_PROCESSED_DIR

path = os.path.join(FINGERPRINT_PROCESSED_DIR, 'extracted_features.csv')
df = pd.read_csv(path)

print("\n🔍 --- BIOMETRIC CSV INSPECTION ---")
print(f"File Path: {path}")
print(f"Exact Headers: {df.columns.tolist()}")
print(f"First Row Data:\n{df.iloc[0].to_dict()}")
print("------------------------------------\n")