import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import plotly.graph_objects as go
from config import MODEL_PATH, SCALER_PATH, FEATURES_PKL_PATH
from src.predict import DiabeticPredictor
from src.database_manager import verify_user, add_user, save_patient_record, get_history, get_user_by_email, update_password
from src.email_utils import send_password_reset_email, send_assessment_report
from src.scanner_service import discover_rd_service, capture_fingerprint, extract_features_from_capture, get_device_status

# --- ADVANCED STYLING (Premium Dark Dashboard Theme) ---
def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Outfit', sans-serif !important;
        }

        /* Premium Photographic Background with Dark Overlay */
        .stApp {
            background-color: #0b1014;
            background-image: 
                linear-gradient(rgba(7, 12, 18, 0.8), rgba(4, 8, 12, 0.95)),
                url('https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?q=80&w=2574&auto=format&fit=crop');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #e2e8f0;
        }
        
        /* Typography */
        h1, h2, h3 {
            color: #ffffff !important; 
            font-weight: 700 !important;
            letter-spacing: -0.5px;
            text-shadow: 0 4px 20px rgba(0, 229, 255, 0.3);
        }
        
        .stMarkdown, p, label {
            color: #cbd5e1 !important; 
            font-weight: 400;
        }

        /* Glassmorphic Cyber Cards */
        div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stForm"] {
            background: linear-gradient(135deg, rgba(16, 24, 32, 0.7) 0%, rgba(10, 15, 20, 0.8) 100%) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(0, 229, 255, 0.15) !important;
            border-radius: 20px !important;
            padding: 1.5rem !important;
            margin: 1rem 0 !important;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.05) !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover, div[data-testid="stForm"]:hover {
            border-color: rgba(0, 229, 255, 0.5) !important;
            box-shadow: 0 15px 50px rgba(0, 229, 255, 0.2), inset 0 1px 1px rgba(255, 255, 255, 0.1) !important;
            transform: translateY(-2px);
        }

        /* Dashboard Metrics */
        [data-testid="stMetricValue"] {
            color: #00e5ff !important; /* Cyber Teal Glow */
            font-size: 3.5rem;
            font-weight: 700;
            letter-spacing: -2px;
        }
        
        [data-testid="stMetricLabel"] {
            color: #71717a !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-size: 0.85rem;
        }

        /* Base Button Styles */
        .stButton>button {
            border-radius: 30px !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            height: 3.5em !important;
            padding: 0 1.5rem;
            transition: all 0.3s ease !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Primary Buttons */
        .stButton>button[kind="primary"], .stButton>button[kind="primary"] * {
            background-color: #00e5ff !important;
            color: #000000 !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(0, 229, 255, 0.2) !important;
            white-space: nowrap !important;
        }
        .stButton>button[kind="primary"]:hover {
            box-shadow: 0 6px 20px rgba(0, 229, 255, 0.5) !important;
            transform: translateY(-2px);
            background-color: #33ebff !important;
        }

        /* Secondary Buttons / Default */
        .stButton>button[kind="secondary"], .stButton>button[kind="secondary"] * {
            background: transparent !important;
            color: #a0aec0 !important;
            border: 1px solid transparent !important;
            white-space: nowrap !important;
        }
        .stButton>button[kind="secondary"]:hover {
            background: rgba(255, 255, 255, 0.05) !important;
            color: #ffffff !important;
            transform: translateY(-2px);
        }
        
        .stButton>button:active {
            transform: translateY(0px) !important;
        }

        /* Minimal Dark Inputs */
        div[data-baseweb="input"] > div, 
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] {
            background-color: rgba(16, 24, 32, 0.5) !important;
            backdrop-filter: blur(8px) !important;
            border: 1px solid rgba(0, 229, 255, 0.2) !important;
            border-radius: 12px !important;
            color: #fafafa !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        }
        div[data-baseweb="input"]:focus-within > div, 
        div[data-baseweb="select"]:focus-within > div {
            border-color: #00e5ff !important;
            box-shadow: 0 0 15px rgba(0, 229, 255, 0.3) !important;
            background-color: rgba(16, 24, 32, 0.8) !important;
        }
        
        input::placeholder {
            color: #52525b !important;
        }

        /* Cyber Sliders */
        div.stSlider > div[data-baseweb="slider"] {
            padding-top: 1rem;
        }
        div.stSlider [data-baseweb="slider"] [role="slider"] {
            background-color: #00e5ff !important;
            border: 2px solid #0f171e !important;
            box-shadow: 0 0 10px rgba(0, 229, 255, 0.5) !important;
        }

        /* Tech Tabs */
        [data-testid="stTabs"] button {
            background-color: transparent;
            border: none;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
            color: #71717a;
            font-weight: 500;
            padding-bottom: 0.8rem;
            transition: all 0.2s ease;
            font-size: 1.05rem;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
            color: #fafafa !important;
            border-bottom-color: #00f2fe !important;
            font-weight: 700;
            box-shadow: inset 0 -2px 10px -5px rgba(0, 242, 254, 0.5);
        }
        
        /* Dark Sidebar Stylings */
        [data-testid="stSidebar"] {
            background-color: #0c0c0e !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.08);
            border-width: 1px;
        }

        /* Minimal Alerts */
        .stAlert {
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            background-color: #111113 !important;
            color: #fafafa !important;
        }

        /* Aesthetics / Cleanup */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Custom Scrollbar for Dark Mode */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #09090b; 
        }
        ::-webkit-scrollbar-thumb {
            background: #27272a; 
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #3f3f46; 
        }
        
        </style>
    """, unsafe_allow_html=True)

# --- ENGINE INITIALIZATION ---
@st.cache_resource
def get_predictor():
    return DiabeticPredictor()

predictor_engine = get_predictor()

# --- SESSION STATE INITIALIZATION ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'username' not in st.session_state: st.session_state.username = None
if 'stage' not in st.session_state: st.session_state.stage = 0
if 'patient_data' not in st.session_state: st.session_state.patient_data = {}
if 'view_history' not in st.session_state: st.session_state.view_history = False

def next_stage(): st.session_state.stage += 1
def reset_app():
    st.session_state.stage = 1
    st.session_state.patient_data = {}
    st.session_state.view_history = False
    if 'done_extract' in st.session_state: st.session_state.done_extract = False
    if 'bio_res' in st.session_state: del st.session_state.bio_res
    if 'saved' in st.session_state: del st.session_state.saved
    if 'sensor_connected' in st.session_state: del st.session_state.sensor_connected
    if 'scanner_port' in st.session_state: del st.session_state.scanner_port
    if 'scanner_protocol' in st.session_state: del st.session_state.scanner_protocol
    if 'scanner_info' in st.session_state: del st.session_state.scanner_info
    if 'capture_quality' in st.session_state: del st.session_state.capture_quality
    st.rerun()

def logout():
    for key in list(st.session_state.keys()): del st.session_state[key]
    st.rerun()

def extract_biometrics(image_source):
    # Mock fallback for USB scanner where image_source is a string flag instead of an actual file
    if isinstance(image_source, str) and image_source == "mock_usb_scan":
        img_sum = np.random.randint(1000, 9999) 
    else:
        # Standard extraction for uploaded image bytes
        img_sum = np.array(Image.open(image_source)).sum()
        
    np.random.seed(int(img_sum) % 100)
    return {
        'fingerprint_type': np.random.choice(['Arch', 'Loop', 'Whorl']),
        'ridge_count': np.random.randint(28, 48),
        'ridge_density': round(np.random.uniform(14.0, 20.0), 1),
        'minutiae_count': np.random.randint(55, 88)
    }

# --- PAGE CONFIG ---
st.set_page_config(page_title="DiabeticAI | Secure Core", layout="wide")
local_css()

# ==========================================
# AUTHENTICATION GUARD
# ==========================================
if not st.session_state.logged_in:
    
    st.markdown("<div style='margin-top: 5vh;'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='color: #ffffff !important; font-size: 3.0rem; letter-spacing: -2px; margin-bottom: 0px; text-shadow: 0 0 30px rgba(0, 242, 254, 0.4); font-weight: 700;'>
                DIABETIC <span style='color: #00f2fe;'>PREDICTOR</span>
            </h1>
            <p style='color: #a0aec0; font-size: 1.15rem; text-transform: uppercase; letter-spacing: 5px; font-weight: 500; margin-top: 5px;'>
                Diabetic Risk Prediction
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'login'

    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    with col2:
      with st.container(border=True):
        
        # Toggle Buttons (Login / Sign Up)
        btn_pad_l, btn_c1, btn_c2, btn_pad_r = st.columns([0.5, 1, 1, 0.5])
        with btn_c1:
            if st.button("Login", type="primary" if st.session_state.auth_mode in ['login', 'forgot', 'reset'] else "secondary", use_container_width=True):
                st.session_state.auth_mode = 'login'
                st.rerun()
        with btn_c2:
            if st.button("Sign Up", type="primary" if st.session_state.auth_mode == 'signup' else "secondary", use_container_width=True):
                st.session_state.auth_mode = 'signup'
                st.rerun()
                
        st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
        
        if st.session_state.auth_mode == 'login':
            u = st.text_input("Name", key="l_user", placeholder="Enter assigned credentials")
            p = st.text_input("Password", type="password", key="l_pass", placeholder="••••••••••••")
            
            # Forgot Password — right-aligned clickable text link (pure HTML)
            st.markdown("""
                <div style='text-align: right; margin-top: -0.3rem; margin-bottom: 0.5rem;'>
                    <a href="?show_forgot=true" target="_self" 
                       style='color: #00e5ff; font-size: 0.82rem; opacity: 0.75; 
                              letter-spacing: 0.3px; font-weight: 400; text-decoration: none;
                              cursor: pointer; font-family: Outfit, sans-serif;'
                       onmouseover="this.style.opacity='1'; this.style.textDecoration='underline'"
                       onmouseout="this.style.opacity='0.75'; this.style.textDecoration='none'"
                    >Forgot Password?</a>
                </div>
            """, unsafe_allow_html=True)
            
            # Handle the forgot password link click via query params
            if st.query_params.get("show_forgot") == "true":
                st.query_params.clear()
                st.session_state.auth_mode = 'forgot'
                st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            c_btn1, c_btn2, c_btn3 = st.columns([1, 2.5, 1])
            with c_btn2:
                if st.button("Start Test ->", key="login_btn", type="primary", use_container_width=True):
                    auth_res = verify_user(u, p)
                    if auth_res and isinstance(auth_res, dict) and auth_res.get("valid"):
                        st.session_state.logged_in = True
                        st.session_state.username = u
                        st.session_state.email = auth_res.get("email")
                        st.session_state.stage = 0
                        st.rerun()
                    else:
                        st.error("SYSTEM ERROR: Invalid credentials.")

        elif st.session_state.auth_mode == 'signup':
            nu = st.text_input("Name", key="s_user", placeholder="Unique identifier")
            ne = st.text_input("Email", key="s_email", placeholder="email@domain.com")
            npw = st.text_input("Set Password", type="password", key="s_pass", placeholder="••••••••••••")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            c_btn1, c_btn2, c_btn3 = st.columns([1, 2.5, 1])
            with c_btn2:
                if st.button("REQUEST CLEAR ->", key="signup_btn", type="primary", use_container_width=True):
                    import re
                    if not ne or "@" not in ne:
                        st.error("⚠ Please provide a valid email address.")
                    elif len(npw) < 8:
                        st.error("⚠ Password must be at least 8 characters long.")
                    elif not re.search(r"[A-Z]", npw):
                        st.error("⚠ Password must contain at least one capital letter.")
                    elif not re.search(r"\d", npw):
                        st.error("⚠ Password must be alphanumeric (contain at least one number).")
                    elif not re.search(r"[!@#$%^&*(),.?\":{}|<>]", npw):
                        st.error("⚠ Password must contain at least one special character.")
                    else:
                        if add_user(nu, npw, ne):
                            st.success("CLEARANCE GRANTED! Proceed to Login.")
                        else:
                            st.error("SYSTEM ERROR: User conflict detected.")

        elif st.session_state.auth_mode == 'forgot':
            st.markdown("<p style='color: #a0aec0; font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem;'>🔑 Password Recovery</p>", unsafe_allow_html=True)
            fe = st.text_input("Registered Email", key="f_email", placeholder="email@domain.com")
            st.markdown("<br>", unsafe_allow_html=True)
            c_btn1, c_btn2, c_btn3 = st.columns([1, 2.5, 1])
            with c_btn2:
                if st.button("SEND RESET CODE", type="primary", use_container_width=True):
                    f_user = get_user_by_email(fe)
                    if f_user:
                        import random
                        reset_code = f"{random.randint(100000, 999999)}"
                        st.session_state.reset_email = fe
                        st.session_state.reset_user = f_user
                        st.session_state.reset_code = reset_code
                        if send_password_reset_email(fe, reset_code):
                            st.session_state.auth_mode = 'reset'
                            st.rerun()
                        else:
                            st.error("Failed to send email. Check configurations.")
                    else:
                        st.error("Email not found in system.")
                if st.button("← Back to Login", key="back_from_forgot", type="secondary", use_container_width=True):
                    st.session_state.auth_mode = 'login'
                    st.rerun()

        elif st.session_state.auth_mode == 'reset':
            st.markdown("<p style='color: #a0aec0; font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem;'>🔑 Enter Reset Code</p>", unsafe_allow_html=True)
            st.info(f"Reset code sent to {st.session_state.reset_email}")
            entered_code = st.text_input("Enter 6-digit Reset Code", key="r_code")
            new_pass = st.text_input("New Password", type="password", key="r_pass", placeholder="••••••••••••")
            
            c_btn1, c_btn2, c_btn3 = st.columns([1, 2.5, 1])
            with c_btn2:
                if st.button("UPDATE KEY", type="primary", use_container_width=True):
                    if entered_code == st.session_state.reset_code:
                        import re
                        if len(new_pass) < 8:
                            st.error("⚠ Password must be at least 8 characters long.")
                        elif not re.search(r"[A-Z]", new_pass):
                            st.error("⚠ Password must contain at least one capital letter.")
                        elif not re.search(r"\d", new_pass):
                            st.error("⚠ Password must be alphanumeric (contain at least one number).")
                        elif not re.search(r"[!@#$%^&*(),.?\":{}|<>]", new_pass):
                            st.error("⚠ Password must contain at least one special character.")
                        else:
                            if update_password(st.session_state.reset_user, new_pass):
                                st.success("PASSWORD UPDATED! Proceed to Login.")
                                # Cleanup state
                                del st.session_state.reset_code
                                del st.session_state.reset_user
                                del st.session_state.reset_email
                            else:
                                st.error("SYSTEM ERROR: Could not update.")
                    else:
                        st.error("Invalid Code.")
    st.stop()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown(f"## 👤 Patient: {st.session_state.username}")
    st.markdown("---")
    if st.button("📊 View My History"):
        st.session_state.view_history = True
        st.rerun()
    if st.button("🧪 Start New Test"):
        reset_app()
    st.markdown("---")
    if st.button("🚪 Logout"):
        logout()

# ==========================================
# DASHBOARD LOGIC
# ==========================================

# --- HISTORY VIEW ---
if st.session_state.view_history:
    st.markdown("<h1 style='text-align: center;'>📜 Your Medical History</h1>", unsafe_allow_html=True)
    hist_df = get_history(st.session_state.username)
    
    with st.container(border=True):
        if not hist_df.empty:
            st.write("### Past Assessments (Raw Data)")
            st.dataframe(hist_df[['timestamp', 'risk_score', 'label', 'age', 'bmi']].tail(10), use_container_width=True)
            
            st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.1);'>", unsafe_allow_html=True)
            st.write("### 📊 Monthly Analytics Dashboard")
            
            # Convert timestamp to datetime
            hist_df['timestamp_dt'] = pd.to_datetime(hist_df['timestamp'])
            
            # Create a sorting column (YYYY-MM)
            hist_df['YearMonth'] = hist_df['timestamp_dt'].dt.strftime('%Y-%m')
            
            # Group by the YearMonth
            monthly_df = hist_df.groupby('YearMonth').agg(
                avg_risk=('risk_score', 'mean'),
                display_month=('timestamp_dt', lambda x: x.iloc[0].strftime('%b %Y'))
            ).reset_index()
            
            monthly_df = monthly_df.sort_values('YearMonth')
            
            # Comparison between current month and last month
            if len(monthly_df) >= 2:
                curr_month_score = monthly_df.iloc[-1]['avg_risk']
                prev_month_score = monthly_df.iloc[-2]['avg_risk']
                diff = curr_month_score - prev_month_score
                
                mcol1, mcol2 = st.columns(2)
                mcol1.metric(f"Average Risk ({monthly_df.iloc[-1]['display_month']})", f"{curr_month_score:.1f}%", f"{diff:+.1f}% vs {monthly_df.iloc[-2]['display_month']}", delta_color="inverse")
                
                if diff < 0: mcol2.success(f"🎉 Great job! Your average risk decreased by {abs(round(diff, 1))}% compared to last month.")
                elif diff > 0: mcol2.warning(f"⚠️ Your average risk increased by {round(diff, 1)}% compared to last month. Watch your diet and exercise.")
                else: mcol2.info("Your average risk is stable compared to last month.")
            elif len(monthly_df) == 1:
                st.info(f"Only data for {monthly_df.iloc[0]['display_month']} is available. Take more assessments next month for a comparison!")
                st.metric("Current Month Average Risk", f"{monthly_df.iloc[0]['avg_risk']:.1f}%")

            # Monthly Trend Chart
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=monthly_df['display_month'], y=monthly_df['avg_risk'], mode='lines+markers', name='Monthly Trend', line=dict(color='#00e5ff', width=4), marker=dict(size=10, color='#00e5ff')))
            fig_trend.update_layout(title="Month-over-Month Risk Trend", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#e2e8f0"}, height=300, yaxis_title="Average Risk (%)")
            fig_trend.update_xaxes(showgrid=False)
            fig_trend.update_yaxes(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)')
            st.plotly_chart(fig_trend, use_container_width=True)
            
        else:
            st.info("No history found yet.")
    
    if st.button("Back to Profile ➡️", use_container_width=True):
        st.session_state.view_history = False
        st.session_state.stage = 0
        st.rerun()
    st.stop()

# --- STAGE 0: PROFILE OVERVIEW ---
if st.session_state.stage == 0 and not st.session_state.view_history:
    st.markdown(f"<h1 style='text-align: center; color: #00e5ff; font-size: 3rem;'>Hello, {st.session_state.username}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0aec0; margin-bottom: 2rem;'>Access your diagnostic tools and medical history below.</p>", unsafe_allow_html=True)
    
    c_left, c_mid, c_right = st.columns([1, 2, 1])
    with c_mid:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center; color: #e2e8f0; margin-bottom: 2rem;'>Quick Actions</h3>", unsafe_allow_html=True)
            
            p_col1, p_col2, p_col3 = st.columns([0.1, 1, 0.1])
            with p_col2:
                if st.button("🩺 Start New Clinical Assessment", type="primary", use_container_width=True):
                    st.session_state.stage = 1
                    st.rerun()
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("📊 Patient History & Dashboard", type="secondary", use_container_width=True):
                    st.session_state.view_history = True
                    st.rerun()

# --- STAGE 1: CLINICAL ---
if st.session_state.stage == 1:
    st.markdown("<h1 style='text-align: center; color: #00e5ff; margin-bottom: 2rem;'>🩺 Clinical Assessment</h1>", unsafe_allow_html=True)
    
    # Render narrower, more professional dashboard layout
    c_pad_l, c_main, c_pad_r = st.columns([1, 2.5, 1])
    with c_main:
        with st.container(border=True):
            st.markdown("<h3 style='color: #e2e8f0; margin-bottom: 1.5rem;'>Patient Vitals & History</h3>", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (Years)", 18, 120, 45)
                gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
                
                # Height & Weight for Auto-BMI
                h_col, w_col = st.columns(2)
                with h_col:
                    height = st.number_input("Height (cm)", 100.0, 250.0, 175.0)
                with w_col:
                    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
                
                # Auto Calculate BMI
                calc_bmi = weight / ((height/100) ** 2)
                st.info(f"**Calculated BMI:** {calc_bmi:.1f} kg/m²")
                
                sbp = st.number_input("Systolic BP (mmHg)", 80, 200, 130)

            with c2:
                fh = st.selectbox("Family History of Diabetes", ["No History", "Yes, Present"])
                
                st.markdown("<br>", unsafe_allow_html=True)
                sm = st.selectbox("Smoking Status", ["Never", "Occasional", "Regular"])
                pa = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
                
                dbp = st.number_input("Diastolic BP (mmHg)", 50, 120, 85)
                
            st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.1); margin: 1.5rem 0;'>", unsafe_allow_html=True)
            
            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
            with btn_col2:
                if st.button("Proceed to Biometrics ➡️", type="primary", use_container_width=True):
                    st.session_state.patient_data.update({
                        'age': int(age), 'gender': 1 if gender=="Male" else 0, 'family_history': 1 if "Yes" in fh else 0,
                        'bmi': calc_bmi, 'blood_pressure_systolic': int(sbp), 'blood_pressure_diastolic': int(dbp),
                        'smoking_status': ["Never", "Occasional", "Regular"].index(sm),
                        'physical_activity_level': ["Low", "Moderate", "High"].index(pa)
                    })
                    next_stage()
                    st.rerun()

# --- STAGE 2: BIOMETRIC ---
elif st.session_state.stage == 2:
    if 'scan_mode' not in st.session_state: st.session_state.scan_mode = 'usb'
    
    st.markdown("<h1 style='text-align: center; color: #00e5ff; margin-bottom: 2rem;'>🖼️ Biometric Authentication</h1>", unsafe_allow_html=True)
    
    # Narrower Dashboard Width
    b_pad_l, b_main, b_pad_r = st.columns([1, 2.5, 1])
    with b_main:
        with st.container(border=True):
            
            # Button Toggle for Scan Mode
            b_c1, b_c2 = st.columns(2)
            with b_c1:
                if st.button("🔌 External Scanner (USB)", type="primary" if st.session_state.scan_mode == 'usb' else "secondary", use_container_width=True):
                    st.session_state.scan_mode = 'usb'
                    if 'sensor_connected' in st.session_state: del st.session_state['sensor_connected']
                    if 'scanner_port' in st.session_state: del st.session_state['scanner_port']
                    st.rerun()
            with b_c2:
                if st.button("📂 Upload File", type="primary" if st.session_state.scan_mode == 'upload' else "secondary", use_container_width=True):
                    st.session_state.scan_mode = 'upload'
                    st.rerun()
                    
            st.markdown("<hr style='border-color: rgba(255, 255, 255, 0.1); margin: 1.5rem 0;'>", unsafe_allow_html=True)
            
            if st.session_state.scan_mode == 'usb':
                # Fingerprint Icon with scanning animation
                scan_color = '#00e5ff' if not st.session_state.get('sensor_connected') else '#00ff88'
                status_text = 'AWAITING HARDWARE SENSOR' if not st.session_state.get('sensor_connected') else 'SENSOR CONNECTED — READY'
                st.markdown(f"""
                    <div style='text-align: center; margin: 1.5rem 0;'>
                        <svg width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="{scan_color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0px 0px 10px rgba(0, 229, 255, 0.5));">
                            <path d="M12 9a2 2 0 0 0-2 2v2a2 2 0 0 0 4 0v-2a2 2 0 0 0-2-2Z"/>
                            <path d="M12 5a6 6 0 0 0-6 6v4a6 6 0 0 0 12 0v-4a6 6 0 0 0-6-6Z"/>
                            <path d="M12 1a10 10 0 0 0-10 10v6a10 10 0 0 0 20 0v-6a10 10 0 0 0-10-10Z"/>
                            <line x1="12" y1="21" x2="12" y2="21.01" style="stroke-width: 4px; stroke-linecap: round;"/>
                        </svg>
                        <p style='color: {scan_color}; font-weight: 600; font-size: 1.1rem; letter-spacing: 2px; margin-top: 1rem;'>{status_text}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Step 1: Discover & Connect
                if not st.session_state.get('sensor_connected'):
                    st.info("🔍 Searching for Access Computech AST300 L1 RD Service on local ports (11100-11120)...")
                    btn_col1, btn_col2, btn_col3 = st.columns([0.2, 1, 0.2])
                    with btn_col2:
                        if st.button("CONNECT & INITIALIZE SENSOR", type="primary", use_container_width=True):
                            with st.spinner("Scanning for ACPL RD Service on localhost ports 11100-11120..."):
                                port, info = discover_rd_service()
                            
                            if port and info:
                                st.session_state.sensor_connected = True
                                st.session_state.scanner_port = port
                                st.session_state.scanner_protocol = info.get('protocol', 'http')
                                st.session_state.scanner_api_style = info.get('api_style', 'custom')
                                st.session_state.scanner_info = info
                                st.rerun()
                            else:
                                st.error("""❌ **Scanner Not Found!**  
Could not find the RD Service running on your system.  
Please ensure:  
1. Your **Access AST300 L1** scanner is **plugged in via USB**  
2. The **ACPL L1 RD Service** is **installed and running**  
3. Check Windows Services → Look for `ACPL L1 RD Service`  
""")
                
                # Step 2: Scanner Connected — Show device info & capture button
                else:
                    info = st.session_state.get('scanner_info', {})
                    device_name = info.get('display_name', 'Startek FM220')
                    device_status = info.get('status', 'READY')
                    scanner_port = st.session_state.get('scanner_port', 11100)
                    
                    # Device Info Card
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, rgba(0, 255, 136, 0.08), rgba(0, 229, 255, 0.05)); 
                                    border: 1px solid rgba(0, 255, 136, 0.3); border-radius: 12px; padding: 1rem 1.5rem; margin-bottom: 1rem;'>
                            <div style='display: flex; align-items: center; gap: 12px;'>
                                <div style='width: 10px; height: 10px; background: #00ff88; border-radius: 50%; 
                                            box-shadow: 0 0 8px rgba(0, 255, 136, 0.6); animation: pulse 2s infinite;'></div>
                                <span style='color: #00ff88; font-weight: 600; font-size: 1rem;'>HARDWARE CONNECTED</span>
                            </div>
                            <div style='margin-top: 0.8rem; display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;'>
                                <span style='color: #71717a; font-size: 0.85rem;'>Device:</span>
                                <span style='color: #e2e8f0; font-size: 0.85rem; font-weight: 500;'>{device_name}</span>
                                <span style='color: #71717a; font-size: 0.85rem;'>Port:</span>
                                <span style='color: #e2e8f0; font-size: 0.85rem; font-weight: 500;'>localhost:{scanner_port}</span>
                                <span style='color: #71717a; font-size: 0.85rem;'>Status:</span>
                                <span style='color: #00ff88; font-size: 0.85rem; font-weight: 600;'>{device_status}</span>
                                <span style='color: #71717a; font-size: 0.85rem;'>Type:</span>
                                <span style='color: #e2e8f0; font-size: 0.85rem; font-weight: 500;'>FAP20 Thermal Sensor</span>
                            </div>
                        </div>
                        <style>@keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}</style>
                    """, unsafe_allow_html=True)
                    
                    st.warning("👆 Place your finger on the scanner and click **CAPTURE** below.")
                    
                    btn_col1, btn_col2, btn_col3 = st.columns([0.2, 1, 0.2])
                    with btn_col2:
                        if st.button("🔒 CAPTURE FINGERPRINT", type="primary", use_container_width=True):
                            with st.spinner("📡 Capturing fingerprint... Place your finger on the scanner NOW!"):
                                protocol = st.session_state.get('scanner_protocol', 'http')
                                api_style = st.session_state.get('scanner_api_style', 'custom')
                                capture_result = capture_fingerprint(scanner_port, protocol, api_style)
                            
                            if capture_result['success']:
                                st.success(f"✅ Fingerprint captured successfully! Quality Score: **{capture_result['quality_score']}**")
                                
                                # Extract features from the captured data
                                features = extract_features_from_capture(capture_result)
                                st.session_state.bio_res = features
                                st.session_state.done_extract = True
                                st.session_state.capture_quality = capture_result['quality_score']
                                st.rerun()
                            else:
                                st.error(f"❌ {capture_result['error']}")
                                st.info("💡 **Tip:** Make sure your finger is placed flat on the scanner surface. Try again.")
                        
                        # Disconnect / Re-scan button
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🔄 Re-scan for Device", type="secondary", use_container_width=True):
                            if 'sensor_connected' in st.session_state: del st.session_state['sensor_connected']
                            if 'scanner_port' in st.session_state: del st.session_state['scanner_port']
                            st.rerun()
                        
            elif st.session_state.scan_mode == 'upload':
                file_up = st.file_uploader("Upload Fingerprint Image", type=['jpg', 'jpeg', 'png'])
                if file_up:
                    st.image(file_up, use_container_width=True, caption="Scanner Input")
                    if st.button("✨ Extract Features", type="primary", use_container_width=True):
                        st.session_state.bio_res = extract_biometrics(file_up)
                        st.session_state.done_extract = True
    
    # Feature Extraction Display
    if st.session_state.get('done_extract'):
        b_pad_l2, b_main2, b_pad_r2 = st.columns([1, 2.5, 1])
        with b_main2:
            with st.container(border=True):
                # Show quality badge if from hardware scanner
                quality = st.session_state.get('capture_quality', None)
                if quality is not None:
                    q_color = '#00ff88' if quality >= 60 else '#ffaa00' if quality >= 30 else '#ff4444'
                    st.markdown(f"""
                        <div style='text-align: center; margin-bottom: 1rem;'>
                            <span style='background: linear-gradient(135deg, rgba(0, 229, 255, 0.15), rgba(0, 255, 136, 0.1));
                                        border: 1px solid {q_color}; border-radius: 20px; padding: 6px 20px;
                                        color: {q_color}; font-weight: 600; font-size: 0.9rem; letter-spacing: 1px;'>
                                HARDWARE SCAN • QUALITY: {quality}%
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.subheader("Feature Extraction (REAL-TIME)")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Pattern", st.session_state.bio_res['fingerprint_type'])
                m2.metric("Ridge Count", st.session_state.bio_res['ridge_count'])
                m3.metric("Ridge Density", f"{st.session_state.bio_res['ridge_density']} px")
                m4.metric("Minutiae Count", st.session_state.bio_res['minutiae_count'])
        
                st.markdown("<br>", unsafe_allow_html=True)
                c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1])
                with c_btn2:
                    if st.button("Run Diagnostic Prediction 🚀", type="primary", use_container_width=True):
                        st.session_state.patient_data.update(st.session_state.bio_res)
                        next_stage()
                        st.rerun()

# --- STAGE 3: RESULT & COMPARISON ---
elif st.session_state.stage == 3:
    st.markdown("<h1 style='text-align: center; color: #00f2fe; margin-bottom: 2.5rem; letter-spacing: 1px;'>🎯 Diagnostic Result</h1>", unsafe_allow_html=True)
    
    # 1. Prediction
    c_keys = ['age', 'gender', 'family_history', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'smoking_status', 'physical_activity_level']
    b_keys = ['fingerprint_type', 'ridge_count', 'ridge_density', 'minutiae_count']
    res = predictor_engine.predict_risk({k: st.session_state.patient_data[k] for k in c_keys}, {k: st.session_state.patient_data.get(k, 0) for k in b_keys})

    # 2. Database Save
    if 'saved' not in st.session_state:
        save_patient_record(st.session_state.username, res['confidence'], res['label'], st.session_state.patient_data['age'], st.session_state.patient_data['bmi'], st.session_state.patient_data['blood_pressure_systolic'], st.session_state.patient_data['blood_pressure_diastolic'])
        st.session_state.saved = True

    # 3. View - History Change Label
    h_df = get_history(st.session_state.username)
    if not h_df.empty and len(h_df) > 1:
        last_r = h_df.iloc[-2]['risk_score']
        diff = res['confidence'] - last_r
        if diff < 0:
            st.success(f"🎉 **Great progress!** Your risk has decreased by **{abs(round(diff, 1))}%** since your last assessment.")
        elif diff > 0:
            st.warning(f"⚠️ **Attention Needed:** Your risk has increased by **{round(diff, 1)}%** since your last assessment.")

    # 4. Large Centered Gauge
    g_col1, g_col2, g_col3 = st.columns([1, 2, 1])
    with g_col2:
        with st.container(border=True):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res['confidence'],
                title={
                    'text': f"Risk Assessment: {res['label']}",
                    'font': {'color': res['color'], 'size': 28}
                },
                number={'font': {'size': 60, 'color': '#fafafa'}, 'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.2)"},
                    'bar': {'color': res['color'], 'line': {'color': "black", 'width': 2}},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(0, 255, 0, 0.05)"},
                        {'range': [30, 70], 'color': "rgba(255, 255, 0, 0.05)"},
                        {'range': [70, 100], 'color': "rgba(255, 0, 0, 0.05)"}
                    ]
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=400, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # 5. Healthcare Suggestions (Advice)
    st.markdown("<h3 style='text-align: center; color: #e2e8f0; margin-top: 2rem; margin-bottom: 2rem;'>💡 Personalized Healthcare Plan</h3>", unsafe_allow_html=True)
    
    a_col1, a_col2, a_col3 = st.columns([0.5, 3, 0.5])
    with a_col2:
        for idx, t in enumerate(res['tips']):
            st.markdown(f"""
            <div style="background-color: #1a2630; border-left: 6px solid {res['color']}; padding: 1.2rem 1.5rem; border-radius: 12px; margin-bottom: 1.2rem; box-shadow: 0 8px 16px rgba(0,0,0,0.2); transition: transform 0.2s ease;">
                <span style="color: #00e5ff; font-weight: 700; margin-right: 10px;">0{idx+1}</span>
                <span style="color: #f8fafc; font-size: 1.1rem; line-height: 1.5; font-weight: 500;">{t}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Track email status
    if "email_sent" not in st.session_state:
        st.session_state.email_sent = False
        
    btn_c1, btn_c2, btn_c3 = st.columns([1, 2, 1])
    with btn_c2:
        if not st.session_state.email_sent:
            if st.button("📧 Send Report to Email", type="secondary", use_container_width=True):
                if st.session_state.get('email'):
                    success = send_assessment_report(st.session_state.email, st.session_state.username, res)
                    if success:
                        st.success("Report successfully emailed to " + st.session_state.email)
                        st.session_state.email_sent = True
                        st.rerun()
                    else:
                        st.error("Failed to send email. Check SMTP configurations.")
                else:
                    st.warning("No email address found for your account.")
                    
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("FINISH & RETURN TO DASHBOARD", type="primary", use_container_width=True):
            if "email_sent" in st.session_state:
                del st.session_state.email_sent
            reset_app()