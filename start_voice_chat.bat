@echo off
echo 🎙️ Starting Cerebrus AI Voice Chat...
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo ⚡ Activating virtual environment...
    call ".venv\Scripts\activate.bat"
)

REM Start the launcher
python launch_voice_chat.py

pause