@echo off
echo Starting Sentiment Analysis Console Application...

:: Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found!
    echo Creating new virtual environment...
    python -m venv venv
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Check if requirements are installed
if not exist "venv\Scripts\pytest.exe" (
    echo Installing requirements...
    pip install -r requirements.txt
)

:: Run the console application
echo Starting application...
python web/console-app.py

:: Deactivate virtual environment
deactivate

:: Pause to see any error messages
pause