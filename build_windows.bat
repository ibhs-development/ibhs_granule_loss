@echo off
setlocal

rem TESSERACT_DIR may be set before running this script to use a custom install.
pyinstaller --clean --noconfirm GranuleLoss.spec
if errorlevel 1 exit /b %errorlevel%

echo.
echo Built dist\GranuleLoss.exe with Tesseract OCR included.
