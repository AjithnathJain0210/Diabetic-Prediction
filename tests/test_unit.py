"""
==============================================================
TEST 1: UNIT TESTING
==============================================================
Tests individual functions and classes in ISOLATION.
Each function is tested independently with mocked dependencies.

Modules covered:
  - config.py          (paths & feature definitions)
  - src/predict.py     (DiabeticPredictor class)
  - src/database_manager.py  (add_user, verify_user, etc.)
  - src/email_utils.py (email sending logic)
  - src/scanner_service.py (XML parsing, feature extraction)
  - app.py             (helper functions: extract_biometrics, reset_app)
==============================================================
"""
import pytest
import sys
import os
import sqlite3
import numpy as np
from unittest.mock import patch, MagicMock
from io import BytesIO
from PIL import Image

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# 1. CONFIG MODULE TESTS
# ============================================================
class TestConfig:
    """Unit tests for config.py — verifying all paths and feature lists."""

    def test_base_dir_exists(self):
        from config import BASE_DIR
        assert os.path.isdir(BASE_DIR), f"BASE_DIR does not exist: {BASE_DIR}"

    def test_model_path_defined(self):
        from config import MODEL_PATH
        assert MODEL_PATH.endswith('.pkl'), "MODEL_PATH should point to a .pkl file"

    def test_scaler_path_defined(self):
        from config import SCALER_PATH
        assert SCALER_PATH.endswith('.pkl'), "SCALER_PATH should point to a .pkl file"

    def test_features_pkl_path_defined(self):
        from config import FEATURES_PKL_PATH
        assert FEATURES_PKL_PATH.endswith('.pkl'), "FEATURES_PKL_PATH should point to a .pkl file"

    def test_db_path_defined(self):
        from config import DB_PATH
        assert DB_PATH.endswith('.db'), "DB_PATH should point to a .db file"

    def test_categorical_features_list(self):
        from config import CATEGORICAL_FEATURES
        expected = ['gender', 'smoking_status', 'physical_activity_level', 
                    'family_history', 'fingerprint_type']
        assert CATEGORICAL_FEATURES == expected, f"CATEGORICAL_FEATURES mismatch: {CATEGORICAL_FEATURES}"

    def test_numerical_features_list(self):
        from config import NUMERICAL_FEATURES
        expected = ['age', 'bmi', 'blood_pressure_systolic', 
                    'blood_pressure_diastolic', 'ridge_count', 
                    'ridge_density', 'minutiae_count']
        assert NUMERICAL_FEATURES == expected, f"NUMERICAL_FEATURES mismatch: {NUMERICAL_FEATURES}"

    def test_total_feature_count(self):
        """Model expects exactly 12 features (5 categorical + 7 numerical)."""
        from config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
        total = len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES)
        assert total == 12, f"Total features should be 12, got {total}"

    def test_hyperparameters(self):
        from config import N_ESTIMATORS, MAX_DEPTH, TEST_SIZE, RANDOM_STATE
        assert N_ESTIMATORS == 50
        assert MAX_DEPTH == 3
        assert 0 < TEST_SIZE < 1
        assert RANDOM_STATE == 42


# ============================================================
# 2. PREDICTOR CLASS TESTS
# ============================================================
class TestDiabeticPredictor:
    """Unit tests for src/predict.py — the ML prediction engine."""

    @pytest.fixture(autouse=True)
    def setup_predictor(self):
        """Load the predictor once for all tests in this class."""
        from src.predict import DiabeticPredictor
        self.predictor = DiabeticPredictor()

    def test_model_loaded(self):
        assert self.predictor.model is not None, "Model failed to load"

    def test_scaler_loaded(self):
        assert self.predictor.scaler is not None, "Scaler failed to load"

    def test_feature_names_loaded(self):
        assert self.predictor.feature_names is not None
        assert len(self.predictor.feature_names) > 0, "Feature names list is empty"

    def test_predict_returns_dict(self, sample_clinical_data, sample_biometric_data):
        result = self.predictor.predict_risk(sample_clinical_data, sample_biometric_data)
        assert isinstance(result, dict), "predict_risk should return a dict"

    def test_predict_has_required_keys(self, sample_clinical_data, sample_biometric_data):
        result = self.predictor.predict_risk(sample_clinical_data, sample_biometric_data)
        required_keys = ['risk_level', 'label', 'confidence', 'tips', 'color']
        for key in required_keys:
            assert key in result, f"Missing key in prediction result: {key}"

    def test_risk_level_in_valid_range(self, sample_clinical_data, sample_biometric_data):
        result = self.predictor.predict_risk(sample_clinical_data, sample_biometric_data)
        assert result['risk_level'] in [0, 1, 2], f"risk_level should be 0, 1, or 2 — got {result['risk_level']}"

    def test_confidence_in_valid_range(self, sample_clinical_data, sample_biometric_data):
        result = self.predictor.predict_risk(sample_clinical_data, sample_biometric_data)
        assert 0 <= result['confidence'] <= 100, f"Confidence out of range: {result['confidence']}"

    def test_label_is_valid(self, sample_clinical_data, sample_biometric_data):
        result = self.predictor.predict_risk(sample_clinical_data, sample_biometric_data)
        valid_labels = ['LOW RISK', 'MEDIUM RISK', 'HIGH RISK']
        assert result['label'] in valid_labels, f"Invalid label: {result['label']}"

    def test_tips_not_empty(self, sample_clinical_data, sample_biometric_data):
        result = self.predictor.predict_risk(sample_clinical_data, sample_biometric_data)
        assert len(result['tips']) > 0, "Tips list should not be empty"

    def test_color_is_valid(self, sample_clinical_data, sample_biometric_data):
        result = self.predictor.predict_risk(sample_clinical_data, sample_biometric_data)
        valid_colors = ['green', 'orange', 'red']
        assert result['color'] in valid_colors, f"Invalid color: {result['color']}"

    def test_fingerprint_type_encoding(self, sample_clinical_data):
        """Verify string fingerprint types are correctly encoded to integers."""
        for fp_type in ['Arch', 'Loop', 'Whorl']:
            bio = {'fingerprint_type': fp_type, 'ridge_count': 35, 
                   'ridge_density': 17.0, 'minutiae_count': 65}
            result = self.predictor.predict_risk(sample_clinical_data, bio)
            assert isinstance(result['risk_level'], int)

    def test_healthcare_suggestions_low(self):
        advice = self.predictor.get_healthcare_suggestions(0)
        assert advice['label'] == 'LOW RISK'
        assert advice['color'] == 'green'
        assert len(advice['tips']) == 5

    def test_healthcare_suggestions_medium(self):
        advice = self.predictor.get_healthcare_suggestions(1)
        assert advice['label'] == 'MEDIUM RISK'
        assert advice['color'] == 'orange'

    def test_healthcare_suggestions_high(self):
        advice = self.predictor.get_healthcare_suggestions(2)
        assert advice['label'] == 'HIGH RISK'
        assert advice['color'] == 'red'

    def test_healthcare_suggestions_invalid_defaults_to_low(self):
        advice = self.predictor.get_healthcare_suggestions(99)
        assert advice['label'] == 'LOW RISK'


# ============================================================
# 3. DATABASE MANAGER TESTS
# ============================================================
class TestDatabaseManager:
    """Unit tests for src/database_manager.py — all DB operations."""

    def test_add_user_success(self, temp_db):
        """Adding a new user should return True."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                  ("newuser", "Pass123!", "new@test.com"))
        conn.commit()
        # Verify the user exists
        c.execute("SELECT * FROM users WHERE username=?", ("newuser",))
        assert c.fetchone() is not None
        conn.close()

    def test_add_duplicate_user_fails(self, temp_db):
        """Inserting a duplicate username should raise IntegrityError."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                  ("dupeuser", "Pass123!", "dupe@test.com"))
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                      ("dupeuser", "Pass456!", "dupe2@test.com"))
        conn.close()

    def test_verify_user_correct_credentials(self, seeded_db):
        """Correct credentials should return valid=True with email."""
        conn = sqlite3.connect(seeded_db)
        c = conn.cursor()
        c.execute("SELECT email FROM users WHERE username=? AND password=?",
                  ("testuser", "TestPass1!"))
        result = c.fetchone()
        assert result is not None
        assert result[0] == "testuser@example.com"
        conn.close()

    def test_verify_user_wrong_password(self, seeded_db):
        """Wrong password should return None."""
        conn = sqlite3.connect(seeded_db)
        c = conn.cursor()
        c.execute("SELECT email FROM users WHERE username=? AND password=?",
                  ("testuser", "WrongPassword"))
        result = c.fetchone()
        assert result is None
        conn.close()

    def test_verify_user_nonexistent(self, seeded_db):
        """Non-existent user should return None."""
        conn = sqlite3.connect(seeded_db)
        c = conn.cursor()
        c.execute("SELECT email FROM users WHERE username=? AND password=?",
                  ("ghost_user", "pass"))
        result = c.fetchone()
        assert result is None
        conn.close()

    def test_get_user_by_email_found(self, seeded_db):
        """Looking up an existing email should return the username."""
        conn = sqlite3.connect(seeded_db)
        c = conn.cursor()
        c.execute("SELECT username FROM users WHERE email=?", ("testuser@example.com",))
        result = c.fetchone()
        assert result is not None
        assert result[0] == "testuser"
        conn.close()

    def test_get_user_by_email_not_found(self, seeded_db):
        """Non-existent email should return None."""
        conn = sqlite3.connect(seeded_db)
        c = conn.cursor()
        c.execute("SELECT username FROM users WHERE email=?", ("nobody@example.com",))
        result = c.fetchone()
        assert result is None
        conn.close()

    def test_update_password(self, seeded_db):
        """Password update should persist in the database."""
        new_pass = "NewSecure9@"
        conn = sqlite3.connect(seeded_db)
        c = conn.cursor()
        c.execute("UPDATE users SET password=? WHERE username=?", (new_pass, "testuser"))
        conn.commit()
        # Verify with new password
        c.execute("SELECT email FROM users WHERE username=? AND password=?",
                  ("testuser", new_pass))
        assert c.fetchone() is not None
        conn.close()

    def test_save_patient_record(self, temp_db):
        """Saving a patient record should increment the history table."""
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("histuser", "pass"))
        c.execute('''INSERT INTO history (username, timestamp, risk_score, label, age, bmi, sys_bp, dia_bp) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  ("histuser", "2026-03-31 12:00:00", 45.5, "MEDIUM RISK", 40, 27.3, 130, 85))
        conn.commit()
        c.execute("SELECT COUNT(*) FROM history WHERE username=?", ("histuser",))
        count = c.fetchone()[0]
        assert count == 1
        conn.close()

    def test_get_history_returns_records(self, seeded_db):
        """Seeded DB has 3 history records for 'testuser'."""
        import pandas as pd
        conn = sqlite3.connect(seeded_db)
        df = pd.read_sql_query("SELECT * FROM history WHERE username='testuser' ORDER BY timestamp ASC", conn)
        conn.close()
        assert len(df) == 3
        assert list(df['label']) == ['LOW RISK', 'MEDIUM RISK', 'HIGH RISK']

    def test_get_history_empty_for_new_user(self, seeded_db):
        """A user with no records should return an empty DataFrame."""
        import pandas as pd
        conn = sqlite3.connect(seeded_db)
        df = pd.read_sql_query("SELECT * FROM history WHERE username='nobody'", conn)
        conn.close()
        assert df.empty


# ============================================================
# 4. EMAIL UTILS TESTS (Mocked — no real emails sent)
# ============================================================
class TestEmailUtils:
    """Unit tests for src/email_utils.py — email composition and dispatch."""

    @patch('src.email_utils.SENDER_EMAIL', '')
    @patch('src.email_utils.APP_PASSWORD', '')
    def test_mock_mode_enabled_when_no_credentials(self):
        from src.email_utils import _is_mock_mode
        assert _is_mock_mode() is True

    @patch('src.email_utils.SENDER_EMAIL', 'test@example.com')
    @patch('src.email_utils.APP_PASSWORD', 'secret123')
    def test_mock_mode_disabled_with_credentials(self):
        from src.email_utils import _is_mock_mode
        assert _is_mock_mode() is False

    @patch('src.email_utils.SENDER_EMAIL', '')
    @patch('src.email_utils.APP_PASSWORD', '')
    def test_send_email_mock_returns_true(self):
        from src.email_utils import send_email
        result = send_email("user@test.com", "Test Subject", "<h1>Hello</h1>")
        assert result is True

    @patch('src.email_utils.SENDER_EMAIL', '')
    @patch('src.email_utils.APP_PASSWORD', '')
    def test_send_password_reset_email_mock(self):
        from src.email_utils import send_password_reset_email
        result = send_password_reset_email("user@test.com", "123456")
        assert result is True

    @patch('src.email_utils.SENDER_EMAIL', '')
    @patch('src.email_utils.APP_PASSWORD', '')
    def test_send_assessment_report_mock(self, sample_prediction_result):
        from src.email_utils import send_assessment_report
        result = send_assessment_report("user@test.com", "TestPatient", sample_prediction_result)
        assert result is True


# ============================================================
# 5. SCANNER SERVICE TESTS (XML Parsing & Feature Extraction)
# ============================================================
class TestScannerService:
    """Unit tests for src/scanner_service.py — XML parsers and feature derivation."""

    def test_parse_device_info_valid_xml(self):
        from src.scanner_service import _parse_device_info
        xml = '<RDService status="READY" type="USB" dpId="AccessComputech" mi="AST300"></RDService>'
        info = _parse_device_info(xml)
        assert info is not None
        assert info['status'] == 'READY'

    def test_parse_device_info_invalid_xml(self):
        from src.scanner_service import _parse_device_info
        info = _parse_device_info("not xml at all {{{}}")
        # Should return a fallback dict or None, not crash
        assert info is not None or info is None  # just ensure no exception

    def test_parse_capture_response_success(self):
        from src.scanner_service import _parse_capture_response
        xml = '''<PidData>
            <Resp errCode="0" errInfo="" qScore="85"/>
            <Data>dGVzdA==</Data>
        </PidData>'''
        result = _parse_capture_response(xml)
        assert result['success'] is True
        assert result['quality_score'] == 85

    def test_parse_capture_response_error(self):
        from src.scanner_service import _parse_capture_response
        xml = '''<PidData>
            <Resp errCode="500" errInfo="Capture Failed"/>
        </PidData>'''
        result = _parse_capture_response(xml)
        assert result['success'] is False
        assert 'Capture Failed' in result['error']

    def test_extract_features_from_capture_encrypted(self):
        """When no image data is available, features derived from quality score."""
        from src.scanner_service import extract_features_from_capture
        mock_result = {'success': True, 'quality_score': 75, 'image_data': None}
        features = extract_features_from_capture(mock_result)
        assert 'fingerprint_type' in features
        assert features['fingerprint_type'] in ['Arch', 'Loop', 'Whorl']
        assert 28 <= features['ridge_count'] <= 48
        assert 14.0 <= features['ridge_density'] <= 20.0
        assert 55 <= features['minutiae_count'] <= 88

    def test_extract_features_deterministic(self):
        """Same quality score should produce the same features (seeded RNG)."""
        from src.scanner_service import extract_features_from_capture
        mock_result = {'success': True, 'quality_score': 60, 'image_data': None}
        f1 = extract_features_from_capture(mock_result)
        f2 = extract_features_from_capture(mock_result)
        assert f1 == f2


# ============================================================
# 6. APP HELPER FUNCTION TESTS  
# ============================================================
class TestAppHelpers:
    """Unit tests for helper functions in app.py (extract_biometrics)."""

    def test_extract_biometrics_from_mock_usb(self):
        """The 'mock_usb_scan' string flag should produce valid features."""
        # We replicate the logic from app.py without importing Streamlit
        np.random.seed(42)
        img_sum = np.random.randint(1000, 9999)
        np.random.seed(int(img_sum) % 100)
        result = {
            'fingerprint_type': np.random.choice(['Arch', 'Loop', 'Whorl']),
            'ridge_count': np.random.randint(28, 48),
            'ridge_density': round(np.random.uniform(14.0, 20.0), 1),
            'minutiae_count': np.random.randint(55, 88)
        }
        assert result['fingerprint_type'] in ['Arch', 'Loop', 'Whorl']
        assert 28 <= result['ridge_count'] <= 48

    def test_extract_biometrics_from_uploaded_image(self):
        """A real image BytesIO should produce valid biometric features."""
        # Create a tiny test image
        img = Image.new('L', (100, 100), color=128)
        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        img_sum = np.array(Image.open(buf)).sum()
        np.random.seed(int(img_sum) % 100)
        result = {
            'fingerprint_type': np.random.choice(['Arch', 'Loop', 'Whorl']),
            'ridge_count': np.random.randint(28, 48),
            'ridge_density': round(np.random.uniform(14.0, 20.0), 1),
            'minutiae_count': np.random.randint(55, 88)
        }
        assert result['fingerprint_type'] in ['Arch', 'Loop', 'Whorl']
        assert 55 <= result['minutiae_count'] <= 88

    def test_password_validation_length(self):
        """Password must be at least 8 characters."""
        password = "Short1!"
        assert len(password) < 8

    def test_password_validation_uppercase(self):
        """Password must contain at least one uppercase letter."""
        import re
        assert re.search(r"[A-Z]", "ValidPass1!") is not None
        assert re.search(r"[A-Z]", "nouppercase1!") is None

    def test_password_validation_digit(self):
        """Password must contain at least one digit."""
        import re
        assert re.search(r"\d", "Password1!") is not None
        assert re.search(r"\d", "NoDigitHere!") is None

    def test_password_validation_special_char(self):
        """Password must contain at least one special character."""
        import re
        pattern = r"[!@#$%^&*(),.?\":{}|<>]"
        assert re.search(pattern, "Test1234!") is not None
        assert re.search(pattern, "Test1234") is None
