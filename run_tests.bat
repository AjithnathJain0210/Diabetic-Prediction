@echo off
echo ============================================================
echo    DIABETIC PREDICTOR - COMPLETE TEST SUITE
echo ============================================================
echo.

echo [1/4] Running UNIT TESTS...
echo ------------------------------------------------------------
python -m pytest tests/test_unit.py -v --tb=short
echo.

echo [2/4] Running INTEGRATION TESTS...
echo ------------------------------------------------------------
python -m pytest tests/test_integration.py -v --tb=short
echo.

echo [3/4] Running FUNCTIONAL TESTS...
echo ------------------------------------------------------------
python -m pytest tests/test_functional.py -v --tb=short
echo.

echo [4/4] Running UI TESTS (requires Streamlit app running)...
echo ------------------------------------------------------------
echo NOTE: Start the app first with: streamlit run app.py
python -m pytest tests/test_ui.py -v --tb=short
echo.

echo ============================================================
echo    ALL TESTS COMPLETED
echo ============================================================
pause
