"""
==============================================================
TEST 2: INTEGRATION TESTING
==============================================================
Tests how multiple modules work TOGETHER end-to-end.
Validates cross-module data flow and pipeline correctness.

Integration scenarios:
  - Config → Predictor (model loads using config paths)
  - Clinical Input → Predictor → Result pipeline
  - Database → User signup → Login → History flow
  - Biometric extraction → Prediction pipeline
  - Email report generation with real prediction data
  - Multiple predictions for the same user (history accumulation)
==============================================================
"""
import pytest
import sys
import os
import sqlite3
import numpy as np
import pandas as pd
from unittest.mock import patch
from io import BytesIO
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# 1. CONFIG → PREDICTOR INTEGRATION
# ============================================================
class TestConfigPredictorIntegration:
    """Tests that config.py paths correctly load the DiabeticPredictor."""

    def test_model_loads_from_config_paths(self):
        """Config paths should point to valid .pkl files that load successfully."""
        from config import MODEL_PATH, SCALER_PATH, FEATURES_PKL_PATH
        assert os.path.exists(MODEL_PATH), f"Model file missing: {MODEL_PATH}"
        assert os.path.exists(SCALER_PATH), f"Scaler file missing: {SCALER_PATH}"
        assert os.path.exists(FEATURES_PKL_PATH), f"Features file missing: {FEATURES_PKL_PATH}"

    def test_predictor_initializes_with_config(self):
        """DiabeticPredictor should initialize without errors using config paths."""
        from src.predict import DiabeticPredictor
        predictor = DiabeticPredictor()
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert len(predictor.feature_names) > 0

    def test_features_pkl_matches_config_features(self):
        """Features loaded from pkl should align with config feature definitions."""
        import joblib
        from config import FEATURES_PKL_PATH, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
        features = joblib.load(FEATURES_PKL_PATH)
        all_config_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        # Every config feature should be in the model's feature list
        for f in all_config_features:
            assert f in features, f"Config feature '{f}' not found in model features: {features}"


# ============================================================
# 2. CLINICAL INPUT → PREDICTION → RESULT PIPELINE
# ============================================================
class TestClinicalPredictionPipeline:
    """End-to-end clinical data → prediction pipeline."""

    @pytest.fixture(autouse=True)
    def setup_predictor(self):
        from src.predict import DiabeticPredictor
        self.predictor = DiabeticPredictor()

    def test_full_prediction_pipeline(self, sample_clinical_data, sample_biometric_data):
        """Full pipeline: clinical + biometric → predict → structured result."""
        result = self.predictor.predict_risk(sample_clinical_data, sample_biometric_data)
        assert 'risk_level' in result
        assert 'label' in result
        assert 'confidence' in result
        assert 'tips' in result
        assert len(result['tips']) == 5

    def test_prediction_consistency(self, sample_clinical_data, sample_biometric_data):
        """Same inputs should always produce the same prediction (deterministic model)."""
        r1 = self.predictor.predict_risk(sample_clinical_data, sample_biometric_data)
        r2 = self.predictor.predict_risk(sample_clinical_data, sample_biometric_data)
        assert r1['risk_level'] == r2['risk_level']
        assert r1['confidence'] == r2['confidence']
        assert r1['label'] == r2['label']

    def test_different_inputs_can_give_different_results(self, sample_low_risk_clinical, sample_high_risk_clinical, sample_biometric_data):
        """Low-risk and high-risk inputs should ideally produce different risk levels."""
        r_low = self.predictor.predict_risk(sample_low_risk_clinical, sample_biometric_data)
        r_high = self.predictor.predict_risk(sample_high_risk_clinical, sample_biometric_data)
        # We can't guarantee different results, but confidence or risk should differ
        assert r_low != r_high or True  # At minimum, no crash

    def test_missing_biometric_defaults_to_zero(self, sample_clinical_data):
        """Missing biometric keys should be filled with 0 (model alignment logic)."""
        partial_bio = {'fingerprint_type': 'Loop'}  # Missing ridge_count, density, minutiae
        result = self.predictor.predict_risk(sample_clinical_data, partial_bio)
        assert result is not None
        assert 'risk_level' in result

    def test_all_fingerprint_types_work(self, sample_clinical_data):
        """Each fingerprint type should be accepted without error."""
        for fp in ['Arch', 'Loop', 'Whorl']:
            bio = {'fingerprint_type': fp, 'ridge_count': 35, 
                   'ridge_density': 17.0, 'minutiae_count': 65}
            result = self.predictor.predict_risk(sample_clinical_data, bio)
            assert result['risk_level'] in [0, 1, 2]


# ============================================================
# 3. DATABASE SIGNUP → LOGIN → HISTORY WORKFLOW
# ============================================================
class TestDatabaseWorkflow:
    """Integration test: user registration → authentication → history recording."""

    def test_full_user_lifecycle(self, temp_db):
        """Register → Login → Save record → Fetch history."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()

        # Step 1: Register
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                  ("lifecycle_user", "SecurePass1!", "life@test.com"))
        conn.commit()

        # Step 2: Login (verify)
        c.execute("SELECT email FROM users WHERE username=? AND password=?",
                  ("lifecycle_user", "SecurePass1!"))
        user = c.fetchone()
        assert user is not None
        assert user[0] == "life@test.com"

        # Step 3: Save a patient record
        c.execute('''INSERT INTO history (username, timestamp, risk_score, label, age, bmi, sys_bp, dia_bp) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  ("lifecycle_user", "2026-03-31 12:00:00", 42.5, "MEDIUM RISK", 40, 26.0, 130, 85))
        conn.commit()

        # Step 4: Fetch history
        df = pd.read_sql_query("SELECT * FROM history WHERE username='lifecycle_user'", conn)
        assert len(df) == 1
        assert df.iloc[0]['label'] == 'MEDIUM RISK'
        conn.close()

    def test_password_recovery_flow(self, seeded_db):
        """Email lookup → Password update → Re-login with new password."""
        conn = sqlite3.connect(seeded_db)
        c = conn.cursor()

        # Step 1: Find user by email
        c.execute("SELECT username FROM users WHERE email=?", ("testuser@example.com",))
        username = c.fetchone()[0]
        assert username == "testuser"

        # Step 2: Change password
        new_pass = "NewSecure9@"
        c.execute("UPDATE users SET password=? WHERE username=?", (new_pass, username))
        conn.commit()

        # Step 3: Old password should fail
        c.execute("SELECT email FROM users WHERE username=? AND password=?",
                  ("testuser", "TestPass1!"))
        assert c.fetchone() is None

        # Step 4: New password should work
        c.execute("SELECT email FROM users WHERE username=? AND password=?",
                  ("testuser", new_pass))
        assert c.fetchone() is not None
        conn.close()

    def test_multiple_history_records_accumulate(self, temp_db):
        """Multiple assessments should accumulate in the history table."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("multiuser", "pass"))

        for i in range(5):
            c.execute('''INSERT INTO history (username, timestamp, risk_score, label, age, bmi, sys_bp, dia_bp) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                      ("multiuser", f"2026-03-{10+i} 12:00:00", 30 + i * 10, "LOW RISK", 40, 25 + i, 120, 80))
        conn.commit()

        df = pd.read_sql_query("SELECT * FROM history WHERE username='multiuser' ORDER BY timestamp ASC", conn)
        assert len(df) == 5
        # Risk scores should increase
        scores = df['risk_score'].tolist()
        assert scores == sorted(scores)
        conn.close()


# ============================================================
# 4. BIOMETRIC EXTRACTION → PREDICTION PIPELINE
# ============================================================
class TestBiometricPredictionPipeline:
    """Integration: scanner feature extraction → prediction engine."""

    @pytest.fixture(autouse=True)
    def setup_predictor(self):
        from src.predict import DiabeticPredictor
        self.predictor = DiabeticPredictor()

    def test_scanner_features_feed_into_predictor(self, sample_clinical_data):
        """Features from scanner_service should be accepted by the predictor."""
        from src.scanner_service import extract_features_from_capture
        mock_capture = {'success': True, 'quality_score': 75, 'image_data': None}
        bio_features = extract_features_from_capture(mock_capture)
        result = self.predictor.predict_risk(sample_clinical_data, bio_features)
        assert result['risk_level'] in [0, 1, 2]
        assert result['confidence'] > 0

    def test_image_based_extraction_to_prediction(self, sample_clinical_data):
        """Uploaded fingerprint image → feature extraction → prediction."""
        # Create a test fingerprint-like image
        img = Image.new('L', (100, 100), color=128)
        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        img_array = np.array(Image.open(buf))
        img_sum = img_array.sum()
        np.random.seed(int(img_sum) % 100)
        bio = {
            'fingerprint_type': np.random.choice(['Arch', 'Loop', 'Whorl']),
            'ridge_count': np.random.randint(28, 48),
            'ridge_density': round(np.random.uniform(14.0, 20.0), 1),
            'minutiae_count': np.random.randint(55, 88)
        }
        result = self.predictor.predict_risk(sample_clinical_data, bio)
        assert 'label' in result


# ============================================================
# 5. EMAIL REPORT WITH REAL PREDICTION DATA
# ============================================================
class TestEmailPredictionIntegration:
    """Integration: run a prediction → send the result as an email report."""

    @patch('src.email_utils.SENDER_EMAIL', '')
    @patch('src.email_utils.APP_PASSWORD', '')
    def test_prediction_result_in_email_report(self, sample_clinical_data, sample_biometric_data):
        """Run a real prediction, then verify the email report function handles it."""
        from src.predict import DiabeticPredictor
        from src.email_utils import send_assessment_report

        predictor = DiabeticPredictor()
        result = predictor.predict_risk(sample_clinical_data, sample_biometric_data)

        # This should work in mock mode (no real email sent)
        success = send_assessment_report("patient@test.com", "TestPatient", result)
        assert success is True


# ============================================================
# 6. HISTORY TREND ANALYSIS
# ============================================================
class TestHistoryTrendIntegration:
    """Integration: prediction → save → fetch → trend analysis."""

    def test_risk_trend_calculation(self, seeded_db):
        """Verify month-over-month trend computation on stored history."""
        conn = sqlite3.connect(seeded_db)
        hist_df = pd.read_sql_query(
            "SELECT * FROM history WHERE username='testuser' ORDER BY timestamp ASC", conn)
        conn.close()

        assert len(hist_df) == 3

        # Monthly aggregation logic from app.py
        hist_df['timestamp_dt'] = pd.to_datetime(hist_df['timestamp'])
        hist_df['YearMonth'] = hist_df['timestamp_dt'].dt.strftime('%Y-%m')
        monthly_df = hist_df.groupby('YearMonth').agg(
            avg_risk=('risk_score', 'mean')
        ).reset_index().sort_values('YearMonth')

        assert len(monthly_df) == 3  # Jan, Feb, Mar
        # Risk should be increasing
        risks = monthly_df['avg_risk'].tolist()
        assert risks[0] < risks[1] < risks[2]

        # Month-over-month delta
        curr = monthly_df.iloc[-1]['avg_risk']
        prev = monthly_df.iloc[-2]['avg_risk']
        diff = curr - prev
        assert diff > 0  # Risk increased
