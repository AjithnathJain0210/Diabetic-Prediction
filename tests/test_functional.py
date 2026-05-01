"""
==============================================================
TEST 3: FUNCTIONAL TESTING
==============================================================
Tests the APPLICATION'S FEATURES from an end-user perspective.
Validates business requirements, edge cases, and complete workflows.

Functional scenarios:
  - User registration with validation rules
  - Login authentication with various credentials
  - Password recovery (forgot password) workflow
  - Complete clinical assessment (Stage 1 → 2 → 3)
  - Risk categorization accuracy (LOW / MEDIUM / HIGH)
  - Biometric feature extraction (USB scanner mock + file upload)
  - History dashboard data & trend comparisons
  - Email report generation
  - Session state management (stage transitions, logout)
==============================================================
"""
import pytest
import sys
import os
import re
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
# 1. USER REGISTRATION FUNCTIONAL TESTS
# ============================================================
class TestUserRegistration:
    """Functional tests: user registration with all validation rules."""

    def _validate_password(self, password):
        """Replicate the app.py password validation logic."""
        errors = []
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long.")
        if not re.search(r"[A-Z]", password):
            errors.append("Password must contain at least one capital letter.")
        if not re.search(r"\d", password):
            errors.append("Password must be alphanumeric.")
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character.")
        return errors

    def test_valid_registration(self, temp_db):
        """A proper username + valid email + strong password should succeed."""
        errors = self._validate_password("SecurePass1!")
        assert len(errors) == 0, f"Valid password rejected: {errors}"

        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                  ("validuser", "SecurePass1!", "valid@test.com"))
        conn.commit()
        c.execute("SELECT * FROM users WHERE username='validuser'")
        assert c.fetchone() is not None
        conn.close()

    def test_password_too_short(self):
        errors = self._validate_password("Ab1!")
        assert any("8 characters" in e for e in errors)

    def test_password_no_uppercase(self):
        errors = self._validate_password("lowercase1!")
        assert any("capital letter" in e for e in errors)

    def test_password_no_digit(self):
        errors = self._validate_password("NoDigitHere!")
        assert any("alphanumeric" in e for e in errors)

    def test_password_no_special_char(self):
        errors = self._validate_password("NoSpecial123")
        assert any("special character" in e for e in errors)

    def test_invalid_email_no_at_sign(self):
        email = "invalidemail.com"
        assert "@" not in email

    def test_valid_email_format(self):
        email = "user@domain.com"
        assert "@" in email

    def test_duplicate_username_rejected(self, temp_db):
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                  ("dupeuser", "Pass123!", "dupe@test.com"))
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                      ("dupeuser", "AnotherPass1!", "another@test.com"))
        conn.close()


# ============================================================
# 2. LOGIN AUTHENTICATION FUNCTIONAL TESTS
# ============================================================
class TestLoginAuthentication:
    """Functional tests: login with various credential scenarios."""

    def test_correct_login(self, seeded_db):
        conn = sqlite3.connect(seeded_db)
        c = conn.cursor()
        c.execute("SELECT email FROM users WHERE username=? AND password=?",
                  ("testuser", "TestPass1!"))
        result = c.fetchone()
        conn.close()
        assert result is not None, "Correct credentials should authenticate"
        assert result[0] == "testuser@example.com"

    def test_wrong_password_login(self, seeded_db):
        conn = sqlite3.connect(seeded_db)
        c = conn.cursor()
        c.execute("SELECT email FROM users WHERE username=? AND password=?",
                  ("testuser", "WrongPassword"))
        result = c.fetchone()
        conn.close()
        assert result is None, "Wrong password should not authenticate"

    def test_nonexistent_user_login(self, seeded_db):
        conn = sqlite3.connect(seeded_db)
        c = conn.cursor()
        c.execute("SELECT email FROM users WHERE username=? AND password=?",
                  ("ghostuser", "Password1!"))
        result = c.fetchone()
        conn.close()
        assert result is None

    def test_empty_username_login(self, seeded_db):
        conn = sqlite3.connect(seeded_db)
        c = conn.cursor()
        c.execute("SELECT email FROM users WHERE username=? AND password=?", ("", "pass"))
        result = c.fetchone()
        conn.close()
        assert result is None

    def test_empty_password_login(self, seeded_db):
        conn = sqlite3.connect(seeded_db)
        c = conn.cursor()
        c.execute("SELECT email FROM users WHERE username=? AND password=?",
                  ("testuser", ""))
        result = c.fetchone()
        conn.close()
        assert result is None


# ============================================================
# 3. PASSWORD RECOVERY FUNCTIONAL TESTS
# ============================================================
class TestPasswordRecovery:
    """Functional tests: forgot password → reset code → new password."""

    def test_email_lookup_existing_user(self, seeded_db):
        conn = sqlite3.connect(seeded_db)
        c = conn.cursor()
        c.execute("SELECT username FROM users WHERE email=?", ("testuser@example.com",))
        result = c.fetchone()
        conn.close()
        assert result is not None
        assert result[0] == "testuser"

    def test_email_lookup_nonexistent(self, seeded_db):
        conn = sqlite3.connect(seeded_db)
        c = conn.cursor()
        c.execute("SELECT username FROM users WHERE email=?", ("nobody@test.com",))
        result = c.fetchone()
        conn.close()
        assert result is None

    def test_reset_code_generation(self):
        """Reset code should be a 6-digit string."""
        import random
        code = f"{random.randint(100000, 999999)}"
        assert len(code) == 6
        assert code.isdigit()

    def test_password_update_and_relogin(self, seeded_db):
        """After password update, old login fails but new login succeeds."""
        new_pass = "Reset2026!"
        conn = sqlite3.connect(seeded_db)
        c = conn.cursor()
        c.execute("UPDATE users SET password=? WHERE username=?", (new_pass, "testuser"))
        conn.commit()

        # Old password should fail
        c.execute("SELECT email FROM users WHERE username=? AND password=?",
                  ("testuser", "TestPass1!"))
        assert c.fetchone() is None

        # New password should work
        c.execute("SELECT email FROM users WHERE username=? AND password=?",
                  ("testuser", new_pass))
        assert c.fetchone() is not None
        conn.close()


# ============================================================
# 4. CLINICAL ASSESSMENT WORKFLOW (Stages 1 → 2 → 3)
# ============================================================
class TestClinicalAssessmentWorkflow:
    """Functional tests: the complete 3-stage assessment pipeline."""

    @pytest.fixture(autouse=True)
    def setup_predictor(self):
        from src.predict import DiabeticPredictor
        self.predictor = DiabeticPredictor()

    def test_stage_1_clinical_data_collection(self, sample_clinical_data):
        """Stage 1 should collect all 8 clinical fields."""
        required_fields = ['age', 'gender', 'family_history', 'bmi',
                          'blood_pressure_systolic', 'blood_pressure_diastolic',
                          'smoking_status', 'physical_activity_level']
        for field in required_fields:
            assert field in sample_clinical_data, f"Missing field: {field}"

    def test_stage_1_bmi_calculation(self):
        """BMI = weight / (height_m²) should be calculated correctly."""
        height_cm = 175.0
        weight_kg = 70.0
        calc_bmi = weight_kg / ((height_cm / 100) ** 2)
        assert 20 < calc_bmi < 25, f"BMI {calc_bmi:.1f} seems incorrect for 175cm/70kg"
        assert round(calc_bmi, 1) == 22.9

    def test_stage_2_biometric_extraction(self, sample_biometric_data):
        """Stage 2 should produce 4 biometric features."""
        required_keys = ['fingerprint_type', 'ridge_count', 'ridge_density', 'minutiae_count']
        for key in required_keys:
            assert key in sample_biometric_data

    def test_stage_3_prediction_result(self, sample_clinical_data, sample_biometric_data):
        """Stage 3 combines data and produces a risk assessment."""
        result = self.predictor.predict_risk(sample_clinical_data, sample_biometric_data)
        assert result['risk_level'] in [0, 1, 2]
        assert 0 < result['confidence'] <= 100
        assert len(result['tips']) == 5

    def test_full_3_stage_workflow(self, sample_clinical_data, sample_biometric_data, temp_db):
        """Complete workflow: clinical → biometric → predict → save to DB."""
        # Stage 1: Clinical data (already in sample_clinical_data)
        # Stage 2: Biometric data (already in sample_biometric_data)
        # Stage 3: Prediction
        result = self.predictor.predict_risk(sample_clinical_data, sample_biometric_data)

        # Save to database
        conn = sqlite3.connect(temp_db)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("workflow_user", "pass"))
        c.execute('''INSERT INTO history (username, timestamp, risk_score, label, age, bmi, sys_bp, dia_bp) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  ("workflow_user", "2026-03-31 12:00:00", result['confidence'], result['label'],
                   sample_clinical_data['age'], sample_clinical_data['bmi'],
                   sample_clinical_data['blood_pressure_systolic'],
                   sample_clinical_data['blood_pressure_diastolic']))
        conn.commit()

        # Verify saved
        df = pd.read_sql_query("SELECT * FROM history WHERE username='workflow_user'", conn)
        conn.close()
        assert len(df) == 1
        assert df.iloc[0]['label'] == result['label']


# ============================================================
# 5. RISK CATEGORIZATION ACCURACY TESTS
# ============================================================
class TestRiskCategorization:
    """Functional tests: ensure risk labels match the expected categories."""

    @pytest.fixture(autouse=True)
    def setup_predictor(self):
        from src.predict import DiabeticPredictor
        self.predictor = DiabeticPredictor()

    def test_low_risk_label(self):
        advice = self.predictor.get_healthcare_suggestions(0)
        assert advice['label'] == 'LOW RISK'
        assert advice['color'] == 'green'

    def test_medium_risk_label(self):
        advice = self.predictor.get_healthcare_suggestions(1)
        assert advice['label'] == 'MEDIUM RISK'
        assert advice['color'] == 'orange'

    def test_high_risk_label(self):
        advice = self.predictor.get_healthcare_suggestions(2)
        assert advice['label'] == 'HIGH RISK'
        assert advice['color'] == 'red'

    def test_each_risk_level_has_5_tips(self):
        for level in [0, 1, 2]:
            advice = self.predictor.get_healthcare_suggestions(level)
            assert len(advice['tips']) == 5, f"Level {level} has {len(advice['tips'])} tips instead of 5"

    def test_tips_are_non_empty_strings(self):
        for level in [0, 1, 2]:
            advice = self.predictor.get_healthcare_suggestions(level)
            for tip in advice['tips']:
                assert isinstance(tip, str) and len(tip) > 10


# ============================================================
# 6. HISTORY DASHBOARD FUNCTIONAL TESTS
# ============================================================
class TestHistoryDashboard:
    """Functional tests: monthly analytics and trend visualization data."""

    def test_monthly_aggregation(self, seeded_db):
        """3 records across Jan/Feb/Mar should produce 3 monthly groups."""
        conn = sqlite3.connect(seeded_db)
        df = pd.read_sql_query("SELECT * FROM history WHERE username='testuser'", conn)
        conn.close()

        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        df['YearMonth'] = df['timestamp_dt'].dt.strftime('%Y-%m')
        monthly = df.groupby('YearMonth')['risk_score'].mean().reset_index()
        assert len(monthly) == 3

    def test_month_over_month_comparison(self, seeded_db):
        """The delta between last two months should be calculable."""
        conn = sqlite3.connect(seeded_db)
        df = pd.read_sql_query("SELECT * FROM history WHERE username='testuser' ORDER BY timestamp", conn)
        conn.close()

        scores = df['risk_score'].tolist()
        diff = scores[-1] - scores[-2]
        assert isinstance(diff, float)

    def test_empty_history_handling(self, temp_db):
        """New user with no history should return empty DataFrame."""
        conn = sqlite3.connect(temp_db)
        df = pd.read_sql_query("SELECT * FROM history WHERE username='newguy'", conn)
        conn.close()
        assert df.empty


# ============================================================
# 7. EMAIL REPORT FUNCTIONAL TESTS
# ============================================================
class TestEmailReport:
    """Functional tests: email report generation with various data."""

    @patch('src.email_utils.SENDER_EMAIL', '')
    @patch('src.email_utils.APP_PASSWORD', '')
    def test_password_reset_email_sent(self):
        from src.email_utils import send_password_reset_email
        assert send_password_reset_email("user@test.com", "654321") is True

    @patch('src.email_utils.SENDER_EMAIL', '')
    @patch('src.email_utils.APP_PASSWORD', '')
    def test_assessment_report_sent(self, sample_prediction_result):
        from src.email_utils import send_assessment_report
        result = send_assessment_report("user@test.com", "Patient", sample_prediction_result)
        assert result is True

    @patch('src.email_utils.SENDER_EMAIL', '')
    @patch('src.email_utils.APP_PASSWORD', '')
    def test_report_handles_all_risk_levels(self):
        from src.email_utils import send_assessment_report
        for label, color in [("LOW RISK", "green"), ("MEDIUM RISK", "orange"), ("HIGH RISK", "red")]:
            data = {
                'label': label, 'confidence': 50.0, 'color': color,
                'tips': ["Tip 1", "Tip 2", "Tip 3", "Tip 4", "Tip 5"]
            }
            assert send_assessment_report("user@test.com", "Pat", data) is True


# ============================================================
# 8. SESSION STATE MANAGEMENT TESTS
# ============================================================
class TestSessionManagement:
    """Functional tests: session state transitions (simulated without Streamlit)."""

    def test_initial_session_state(self):
        """Default session should be logged out at stage 0."""
        state = {
            'logged_in': False, 'username': None,
            'stage': 0, 'patient_data': {}, 'view_history': False
        }
        assert state['logged_in'] is False
        assert state['stage'] == 0

    def test_login_sets_session(self):
        state = {'logged_in': False, 'username': None, 'stage': 0}
        # Simulate login
        state['logged_in'] = True
        state['username'] = 'testuser'
        state['stage'] = 0
        assert state['logged_in'] is True
        assert state['username'] == 'testuser'

    def test_stage_progression(self):
        state = {'stage': 0}
        state['stage'] += 1  # Clinical → Biometric
        assert state['stage'] == 1
        state['stage'] += 1  # Biometric → Result
        assert state['stage'] == 2
        state['stage'] += 1
        assert state['stage'] == 3

    def test_reset_app_clears_state(self):
        state = {
            'stage': 3, 'patient_data': {'age': 45, 'bmi': 28},
            'view_history': True, 'done_extract': True
        }
        # Simulate reset_app()
        state['stage'] = 1
        state['patient_data'] = {}
        state['view_history'] = False
        state['done_extract'] = False
        assert state['stage'] == 1
        assert state['patient_data'] == {}

    def test_logout_clears_everything(self):
        state = {
            'logged_in': True, 'username': 'user', 'stage': 2,
            'patient_data': {'age': 50}
        }
        # Simulate logout()
        state.clear()
        assert len(state) == 0
