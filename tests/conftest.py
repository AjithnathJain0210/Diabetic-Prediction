"""
Shared pytest fixtures and configuration for the entire test suite.
All tests share these fixtures to avoid code duplication.
"""
import pytest
import sys
import os
import sqlite3
import tempfile
import shutil

# Add project root to the Python path so all modules can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# DATABASE FIXTURES (Isolated per-test temp database)
# ============================================================

@pytest.fixture
def temp_db(tmp_path):
    """Creates a fresh, isolated SQLite database for each test."""
    db_path = str(tmp_path / "test_patient_records.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, email TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT, timestamp DATETIME, risk_score REAL, 
                  label TEXT, age INTEGER, bmi REAL, 
                  sys_bp INTEGER, dia_bp INTEGER,
                  FOREIGN KEY(username) REFERENCES users(username))''')
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def seeded_db(temp_db):
    """Database pre-populated with a test user and some history records."""
    conn = sqlite3.connect(temp_db)
    c = conn.cursor()
    # Insert a test user
    c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
              ("testuser", "TestPass1!", "testuser@example.com"))
    # Insert 3 history records
    records = [
        ("testuser", "2026-01-15 10:00:00", 25.5, "LOW RISK", 45, 24.5, 120, 80),
        ("testuser", "2026-02-20 14:30:00", 55.2, "MEDIUM RISK", 45, 26.1, 135, 88),
        ("testuser", "2026-03-10 09:15:00", 72.8, "HIGH RISK", 46, 28.3, 148, 95),
    ]
    for r in records:
        c.execute('''INSERT INTO history (username, timestamp, risk_score, label, age, bmi, sys_bp, dia_bp) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', r)
    conn.commit()
    conn.close()
    return temp_db


# ============================================================
# SAMPLE DATA FIXTURES
# ============================================================

@pytest.fixture
def sample_clinical_data():
    """Standard clinical input matching the 8 clinical features."""
    return {
        'gender': 1,                       # Male
        'age': 45,
        'bmi': 28.5,
        'blood_pressure_systolic': 140,
        'blood_pressure_diastolic': 90,
        'smoking_status': 1,               # Occasional
        'physical_activity_level': 0,       # Low
        'family_history': 1                 # Yes
    }


@pytest.fixture
def sample_biometric_data():
    """Standard biometric features from fingerprint analysis."""
    return {
        'fingerprint_type': 'Loop',
        'ridge_count': 35,
        'ridge_density': 17.5,
        'minutiae_count': 68
    }


@pytest.fixture
def sample_low_risk_clinical():
    """Clinical data that should yield a LOW risk prediction."""
    return {
        'gender': 0,
        'age': 25,
        'bmi': 22.0,
        'blood_pressure_systolic': 110,
        'blood_pressure_diastolic': 70,
        'smoking_status': 0,
        'physical_activity_level': 2,
        'family_history': 0
    }


@pytest.fixture
def sample_high_risk_clinical():
    """Clinical data that should yield a HIGH risk prediction."""
    return {
        'gender': 1,
        'age': 65,
        'bmi': 38.0,
        'blood_pressure_systolic': 180,
        'blood_pressure_diastolic': 110,
        'smoking_status': 2,
        'physical_activity_level': 0,
        'family_history': 1
    }


@pytest.fixture
def sample_prediction_result():
    """A mock prediction result object for testing downstream features."""
    return {
        'risk_level': 1,
        'label': 'MEDIUM RISK',
        'confidence': 55.2,
        'color': 'orange',
        'tips': [
            "Sugar Audit: Reduce refined sugars.",
            "Post-Meal Movement: 15-minute brisk walk after meals.",
            "Weight Target: Aim for 5% weight reduction.",
            "Active Monitoring: Check blood sugar every 6 months.",
            "Stress Management: Use yoga to manage cortisol."
        ]
    }
