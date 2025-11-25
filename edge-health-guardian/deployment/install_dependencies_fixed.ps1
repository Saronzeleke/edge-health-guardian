# deployment/install_dependencies_fixed.ps1

Write-Host "🚀 Edge Health Guardian - Windows Dependency Installation" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Function to print colored output
function Print-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Print-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Print-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Print-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check Python
Print-Status "Checking Python installation..."
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Print-Error "Python not found. Please install Python 3.8+ from python.org"
    Start-Process "https://www.python.org/downloads/"
    exit 1
}

# Upgrade pip
Print-Status "Upgrading pip..."
python -m pip install --upgrade pip

# Create virtual environment
Print-Status "Creating Python virtual environment..."
python -m venv health-guardian-env

# Install packages
Print-Status "Installing Python dependencies..."

$packages = @(
    "numpy", "scipy", "pandas", "matplotlib", "seaborn", 
    "scikit-learn", "pillow", "psutil", "pyyaml", "requests",
    "tensorflow", "tflite-runtime", "tensorflow-model-optimization",
    "opencv-python", "mediapipe",
    "pygame", "pyaudio", "flask", "flask-socketio",
    "jupyter", "ipython", "pytest",
    "pyserial"
)

foreach ($package in $packages) {
    Print-Status "Installing $package..."
    & "health-guardian-env\Scripts\python" -m pip install $package
}

# Create directories
Print-Status "Creating project directory structure..."
$directories = @(
    "models\trained_models",
    "models\optimized_models", 
    "models\training_history",
    "data\raw",
    "data\processed", 
    "logs",
    "tests"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Create startup script
Print-Status "Creating startup script..."
@'
@echo off
echo 🚀 Starting Edge Health Guardian...
echo 📍 Privacy-First ^| 🔒 On-Device AI ^| 🏥 Health Monitoring

call health-guardian-env\Scripts\activate
set PYTHONPATH=%PYTHONPATH%;%CD%\src
python main.py %*
pause
'@ | Out-File -FilePath "start_health_guardian.bat" -Encoding ASCII

# Test installation
Print-Status "Testing installation..."
& "health-guardian-env\Scripts\python" -c "import tensorflow as tf; import cv2; import numpy as np; print('✅ All imports successful')"

Print-Success "Installation complete!"
Write-Host "`nNext steps:"
Write-Host "1. Run: .\start_health_guardian.bat"
Write-Host "2. Or manually: health-guardian-env\Scripts\activate then python main.py"