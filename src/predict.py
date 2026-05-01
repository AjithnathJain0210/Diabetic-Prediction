import joblib
import numpy as np
import pandas as pd
import os
import sys

# Ensure the root directory is in the path so config can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_PATH, SCALER_PATH, FEATURES_PKL_PATH

class DiabeticPredictor:
    def __init__(self):
        # 1. Load the Brain (Model), the Filter (Scaler), and the Map (Features)
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.feature_names = joblib.load(FEATURES_PKL_PATH)
            print(f"✅ Predictor initialized with {len(self.feature_names)} features.")
        except Exception as e:
            print(f"❌ Error loading model artifacts: {e}")
            print("Hint: Ensure you ran train_model.py successfully.")
            sys.exit(1)

    def get_healthcare_suggestions(self, risk_level):
        """Tailored medical and lifestyle advice based on risk level."""
        suggestions = {
            0: {
                "label": "LOW RISK",
                "color": "green",
                "tips": [
                    "Maintain Fiber Intake: Aim for 25–30g daily from whole grains.",
                    "Hydration: Drink at least 3 liters of water daily.",
                    "Physical Consistency: Keep exercising at least 150 minutes per week.",
                    "Periodic Screening: Re-check markers annually.",
                    "Sleep Hygiene: Ensure 7–8 hours of rest for insulin sensitivity."
                ]
            },
            1: {
                "label": "MEDIUM RISK",
                "color": "orange",
                "tips": [
                    "Sugar Audit: Reduce refined sugars and beverages immediately.",
                    "Post-Meal Movement: 15-minute brisk walk after major meals.",
                    "Weight Target: Aim for a 5% reduction in body weight.",
                    "Active Monitoring: Check blood sugar every 6 months.",
                    "Stress Management: Use yoga to manage cortisol levels."
                ]
            },
            2: {
                "label": "HIGH RISK",
                "color": "red",
                "tips": [
                    "Medical Consult: Schedule a visit with an endocrinologist.",
                    "Diagnostic Testing: Request HbA1c and OGTT blood tests.",
                    "Dietary Overhaul: Shift strictly to low-GI foods.",
                    "BP Tracking: Monitor blood pressure daily.",
                    "Ocular Health: Schedule a retinal exam for early changes."
                ]
            }
        }
        return suggestions.get(risk_level, suggestions[0])

    def predict_risk(self, clinical_data, biometric_features):
        """
        Takes UI inputs, scales them, and returns prediction with confidence.
        """
        # 1. Merge and clean input keys
        raw_input = {**clinical_data, **biometric_features}
        full_input = {k.lower().strip().replace(' ', '_'): v for k, v in raw_input.items()}
        
        # 2. Encode Fingerprint Type (Arch=0, Loop=1, Whorl=2)
        mapping = {'arch': 0, 'loop': 1, 'whorl': 2}
        if 'fingerprint_type' in full_input:
            f_val = str(full_input['fingerprint_type']).lower()
            full_input['fingerprint_type'] = mapping.get(f_val, 1)

        # 3. Align with Training Features
        input_df = pd.DataFrame([full_input])
        for col in self.feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match exactly how the model was trained
        input_df = input_df[self.feature_names]

        # 4. Scaling
        scaled_array = self.scaler.transform(input_df)

        # 5. Prediction
        risk_code = int(self.model.predict(scaled_array)[0])
        probabilities = self.model.predict_proba(scaled_array)[0]
        confidence = round(float(np.max(probabilities)) * 100, 2)
        
        # 6. Get Advice
        advice = self.get_healthcare_suggestions(risk_code)

        return {
            'risk_level': risk_code,
            'label': advice['label'],
            'confidence': confidence,
            'tips': advice['tips'],
            'color': advice['color']
        }

if __name__ == "__main__":
    # Test Block for Command Line Verification
    predictor = DiabeticPredictor()
    
    sample_clinical = {
        'gender': 1, 'age': 45, 'bmi': 32.0, 
        'blood_pressure_systolic': 150, 'blood_pressure_diastolic': 95,
        'smoking_status': 1, 'physical_activity_level': 0, 'family_history': 1
    }
    
    sample_biometric = {
        'fingerprint_type': 'Whorl', 'ridge_count': 35, 'ridge_density': 17, 'minutiae_count': 70
    }
    
    result = predictor.predict_risk(sample_clinical, sample_biometric)
    
    print("\n" + "="*50)
    print(f"DIABETIC RISK ASSESSMENT: {result['label']}")
    print(f"Confidence: {result['confidence']}%")
    print("="*50)
    for i, tip in enumerate(result['tips'], 1):
        print(f"{i}. {tip}")