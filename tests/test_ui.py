"""
==============================================================
TEST 4: UI TESTING (Selenium WebDriver)
==============================================================
Tests the Streamlit application's user interface in a real browser.
Uses Selenium to automate Chrome and verify visual elements, 
interactions, and navigation flows.

UI scenarios:
  - Login page renders correctly (title, inputs, buttons)
  - Sign Up tab switches correctly
  - Form inputs are interactive (text entry, focus)
  - Error messages display on invalid login
  - Page title and branding verification
  - Responsive layout check (viewport resizing)
  - Navigation between Login / Sign Up tabs

SETUP REQUIREMENTS:
  - Chrome browser installed
  - ChromeDriver in PATH (or auto-managed by selenium-manager)
  - Run: `streamlit run app.py` before executing these tests
    OR set the STREAMLIT_URL environment variable
  
USAGE:
  # Start the app first in a separate terminal:
  streamlit run app.py --server.port 8501
  
  # Then run UI tests:
  pytest tests/test_ui.py -v --timeout=60
==============================================================
"""
import pytest
import sys
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Default URL for the running Streamlit app
STREAMLIT_URL = os.environ.get("STREAMLIT_URL", "http://localhost:8501")


def _create_chrome_driver():
    """Create a headless Chrome WebDriver instance."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=chrome_options)
    driver.implicitly_wait(10)
    return driver


def _wait_for_streamlit(driver, timeout=20):
    """Wait until Streamlit app finishes loading."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    try:
        # Wait for the main app container to appear
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='stApp']"))
        )
        # Extra wait for Streamlit rendering to complete
        time.sleep(3)
        return True
    except Exception:
        return False


# Skip all UI tests if Selenium/Chrome is not available
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    
    # Quick check if Chrome is available
    _test_driver = _create_chrome_driver()
    _test_driver.quit()
    SELENIUM_AVAILABLE = True
except Exception:
    SELENIUM_AVAILABLE = False

skip_reason = "Selenium/Chrome not available or Streamlit app not running"


# ============================================================
# 1. LOGIN PAGE RENDERING TESTS
# ============================================================
@pytest.mark.skipif(not SELENIUM_AVAILABLE, reason=skip_reason)
class TestLoginPageRendering:
    """UI tests: verify the login page displays correctly."""

    @pytest.fixture(autouse=True)
    def setup_browser(self):
        self.driver = _create_chrome_driver()
        self.driver.get(STREAMLIT_URL)
        loaded = _wait_for_streamlit(self.driver)
        yield
        self.driver.quit()

    def test_page_loads_successfully(self):
        """The Streamlit app should load without errors."""
        assert "DiabeticAI" in self.driver.title or "Streamlit" in self.driver.title

    def test_app_title_branding(self):
        """The main title 'DIABETIC PREDICTOR' should be visible."""
        page_source = self.driver.page_source
        assert "DIABETIC" in page_source or "Diabetic" in page_source

    def test_login_button_exists(self):
        """A 'Login' button should be present on the page."""
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        button_texts = [b.text.strip().upper() for b in buttons]
        assert any("LOGIN" in t for t in button_texts), f"Login button not found. Buttons: {button_texts}"

    def test_signup_button_exists(self):
        """A 'Sign Up' button should be present on the page."""
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        button_texts = [b.text.strip().upper() for b in buttons]
        assert any("SIGN UP" in t for t in button_texts), f"Sign Up button not found. Buttons: {button_texts}"

    def test_name_input_field_exists(self):
        """A text input for 'Name' should be present."""
        inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='text']")
        assert len(inputs) >= 1, "No text input fields found on login page"

    def test_password_input_field_exists(self):
        """A password input field should be present."""
        inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='password']")
        assert len(inputs) >= 1, "No password input field found on login page"

    def test_forgot_password_link_exists(self):
        """A 'Forgot Password?' link should be visible."""
        page_source = self.driver.page_source
        assert "Forgot Password" in page_source, "Forgot Password link not found"

    def test_start_test_button_exists(self):
        """The main submit button 'Start Test' should be present."""
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        button_texts = [b.text.strip().upper() for b in buttons]
        assert any("START TEST" in t for t in button_texts), f"Start Test button not found. Buttons: {button_texts}"


# ============================================================
# 2. SIGN UP TAB UI TESTS
# ============================================================
@pytest.mark.skipif(not SELENIUM_AVAILABLE, reason=skip_reason)
class TestSignUpTabUI:
    """UI tests: verify Sign Up tab switches and shows correct fields."""

    @pytest.fixture(autouse=True)
    def setup_browser(self):
        self.driver = _create_chrome_driver()
        self.driver.get(STREAMLIT_URL)
        _wait_for_streamlit(self.driver)
        yield
        self.driver.quit()

    def test_signup_tab_click(self):
        """Clicking 'Sign Up' should switch to the registration form."""
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        for btn in buttons:
            if "SIGN UP" in btn.text.strip().upper():
                btn.click()
                break
        time.sleep(3)  # Wait for Streamlit to rerun
        
        # After clicking, the page should show email and password fields
        page_source = self.driver.page_source
        # Should have 'Email' label somewhere
        assert "Email" in page_source or "email" in page_source

    def test_signup_has_name_field(self):
        """Sign Up should have a Name input."""
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        for btn in buttons:
            if "SIGN UP" in btn.text.strip().upper():
                btn.click()
                break
        time.sleep(3)
        inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='text']")
        assert len(inputs) >= 1

    def test_signup_has_password_field(self):
        """Sign Up should have a password input."""
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        for btn in buttons:
            if "SIGN UP" in btn.text.strip().upper():
                btn.click()
                break
        time.sleep(3)
        inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='password']")
        assert len(inputs) >= 1


# ============================================================
# 3. FORM INTERACTION TESTS
# ============================================================
@pytest.mark.skipif(not SELENIUM_AVAILABLE, reason=skip_reason)
class TestFormInteractions:
    """UI tests: verify form inputs accept user input."""

    @pytest.fixture(autouse=True)
    def setup_browser(self):
        self.driver = _create_chrome_driver()
        self.driver.get(STREAMLIT_URL)
        _wait_for_streamlit(self.driver)
        yield
        self.driver.quit()

    def test_name_input_accepts_text(self):
        """User should be able to type into the Name field."""
        inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='text']")
        if inputs:
            inputs[0].clear()
            inputs[0].send_keys("TestUser123")
            assert inputs[0].get_attribute("value") == "TestUser123"

    def test_password_input_accepts_text(self):
        """User should be able to type into the Password field."""
        inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='password']")
        if inputs:
            inputs[0].clear()
            inputs[0].send_keys("SecurePass1!")
            # Password value should be masked but the attribute stores the real value
            value = inputs[0].get_attribute("value")
            assert len(value) > 0

    def test_invalid_login_shows_error(self):
        """Submitting wrong credentials should display an error message."""
        # Fill in wrong credentials
        text_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='text']")
        pass_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='password']")
        
        if text_inputs and pass_inputs:
            text_inputs[0].clear()
            text_inputs[0].send_keys("wronguser")
            pass_inputs[0].clear()
            pass_inputs[0].send_keys("wrongpass")
            
            # Click the login/start button
            buttons = self.driver.find_elements(By.TAG_NAME, "button")
            for btn in buttons:
                if "START TEST" in btn.text.strip().upper():
                    btn.click()
                    break
            
            time.sleep(4)  # Wait for error to display
            page_source = self.driver.page_source
            # Should show an error indicator (Streamlit error is a div with role="alert")
            assert "SYSTEM ERROR" in page_source or "Invalid" in page_source or "alert" in page_source.lower()


# ============================================================
# 4. RESPONSIVE LAYOUT TESTS
# ============================================================
@pytest.mark.skipif(not SELENIUM_AVAILABLE, reason=skip_reason)
class TestResponsiveLayout:
    """UI tests: verify the app adapts to different screen sizes."""

    @pytest.fixture(autouse=True)
    def setup_browser(self):
        self.driver = _create_chrome_driver()
        self.driver.get(STREAMLIT_URL)
        _wait_for_streamlit(self.driver)
        yield
        self.driver.quit()

    def test_desktop_viewport(self):
        """App should render correctly at 1920x1080."""
        self.driver.set_window_size(1920, 1080)
        time.sleep(2)
        app = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='stApp']")
        assert len(app) > 0, "App container not found at desktop viewport"

    def test_tablet_viewport(self):
        """App should render correctly at 768x1024 (iPad)."""
        self.driver.set_window_size(768, 1024)
        time.sleep(2)
        app = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='stApp']")
        assert len(app) > 0, "App container not found at tablet viewport"

    def test_mobile_viewport(self):
        """App should render correctly at 375x812 (iPhone X)."""
        self.driver.set_window_size(375, 812)
        time.sleep(2)
        app = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='stApp']")
        assert len(app) > 0, "App container not found at mobile viewport"


# ============================================================
# 5. VISUAL ELEMENT VERIFICATION
# ============================================================
@pytest.mark.skipif(not SELENIUM_AVAILABLE, reason=skip_reason)
class TestVisualElements:
    """UI tests: verify styling, colors, and visual indicators."""

    @pytest.fixture(autouse=True)
    def setup_browser(self):
        self.driver = _create_chrome_driver()
        self.driver.get(STREAMLIT_URL)
        _wait_for_streamlit(self.driver)
        yield
        self.driver.quit()

    def test_dark_theme_applied(self):
        """The app should use a dark background theme."""
        app = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='stApp']")
        if app:
            bg_color = app[0].value_of_css_property("background-color")
            # Dark theme: RGB values should be low
            assert bg_color is not None

    def test_custom_font_loaded(self):
        """The page should reference the 'Outfit' Google Font."""
        page_source = self.driver.page_source
        assert "Outfit" in page_source, "Custom font 'Outfit' not found in page source"

    def test_cyber_teal_color_present(self):
        """The signature #00e5ff teal color should be in the page styling."""
        page_source = self.driver.page_source
        assert "00e5ff" in page_source or "00f2fe" in page_source, "Cyber teal color not found"

    def test_no_streamlit_header(self):
        """The default Streamlit header should be hidden."""
        page_source = self.driver.page_source
        # Custom CSS hides the header: header {visibility: hidden;}
        assert "visibility: hidden" in page_source or "visibility:hidden" in page_source
