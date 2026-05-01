import sqlite3
import pandas as pd
from datetime import datetime
import sys
import os

# Ensure we can import from the root config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_PATH

def init_db():
    """Creates the database and tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. Users Table for Login
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (username TEXT PRIMARY KEY, password TEXT, email TEXT)''')
    
    # Check if 'email' column exists (for existing databases)
    c.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in c.fetchall()]
    if 'email' not in columns:
        print("Migrating schema: Adding 'email' column to 'users' table...")
        c.execute("ALTER TABLE users ADD COLUMN email TEXT")

    
    # 2. History Table to track health over time
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT, 
                  timestamp DATETIME, 
                  risk_score REAL, 
                  label TEXT, 
                  age INTEGER, 
                  bmi REAL, 
                  sys_bp INTEGER,
                  dia_bp INTEGER,
                  FOREIGN KEY(username) REFERENCES users(username))''')
    
    conn.commit()
    conn.close()
    print(f"[OK] Database initialized at: {DB_PATH}")

def add_user(username, password, email=None):
    """Registers a new user/patient."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)", (username, password, email))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False # Username taken

def get_user_by_email(email):
    """Retrieves a user by their email for password recovery."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE email=?", (email,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def update_password(username, new_password):
    """Updates a user's password using their username."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE users SET password=? WHERE username=?", (new_password, username))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def verify_user(username, password):
    """Checks if credentials are correct and returns the user's email if valid."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT email FROM users WHERE username=? AND password=?", (username, password))
    result = c.fetchone()
    conn.close()
    if result is not None:
        return {"valid": True, "email": result[0]}
    return {"valid": False}

def save_patient_record(username, risk_score, label, age, bmi, sys_bp, dia_bp):
    """Saves the current diagnosis to history."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''INSERT INTO history (username, timestamp, risk_score, label, age, bmi, sys_bp, dia_bp) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (username, now, risk_score, label, age, bmi, sys_bp, dia_bp))
    conn.commit()
    conn.close()

def get_history(username):
    """Fetches all past records for a patient to show trends."""
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT * FROM history WHERE username='{username}' ORDER BY timestamp ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Initialize database schema automatically
init_db()

if __name__ == "__main__":
    init_db()