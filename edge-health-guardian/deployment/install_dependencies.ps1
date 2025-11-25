# deployment/install_dependencies.ps1

Write-Host "🚀 Edge Health Guardian - Windows Dependency Installation" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Colors for output
$ErrorColor = 'Red'
$SuccessColor = 'Green'
$WarningColor = 'Yellow'
$InfoColor = 'Cyan'

# Function to print colored output
function Print-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $InfoColor
}

function Print-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $SuccessColor
}

function Print-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $WarningColor
}

function Print-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $ErrorColor
}

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
if ($isAdmin) {
    Print-Warning "Running as administrator. It's recommended to run as regular user."
}

# Detect architecture
$architecture = $env:PROCESSOR_ARCHITECTURE
Print-Status "Detected architecture: $architecture"

# Check for Python installation
Print-Status "Checking Python installation..."
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Print-Error "Python not found or not in PATH"
    Print-Status "Please install Python 3.8+ from https://python.org"
    Start-Process "https://www.python.org/downloads/"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check for pip
Print-Status "Checking pip installation..."
try {
    $pipVersion = python -m pip --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ pip installed" -ForegroundColor Green
    } else {
        throw "pip not found"
    }
} catch {
    Print-Error "pip not found. Please ensure pip is installed with Python"
    exit 1
}

# Upgrade pip
Print-Status "Upgrading pip..."
python -m pip install --upgrade pip

# Create Python virtual environment
Print-Status "Creating Python virtual environment..."
python -m venv health-guardian-env

# Install Python dependencies
Print-Status "Installing Python dependencies..."

$packages = @(
    # Base packages
    "numpy", "scipy", "pandas", "matplotlib", "seaborn", 
    "scikit-learn", "pillow", "psutil", "pyyaml", "requests",
    
    # TensorFlow and ML packages
    "tensorflow", "tflite-runtime", "tensorflow-model-optimization",
    
    # Computer vision
    "opencv-python", "mediapipe",
    
    # Audio and UI
    "pygame", "pyaudio", "flask", "flask-socketio",
    
    # Development
    "jupyter", "ipython", "black", "flake8", "pytest", "pytest-cov",
    
    # Hardware (Windows-compatible)
    "pyserial"
)

foreach ($package in $packages) {
    Print-Status "Installing $package..."
    & "health-guardian-env\Scripts\python" -m pip install $package
}

# Create directory structure
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

# Check for pre-trained models
Print-Status "Checking for pre-trained models..."
if (!(Test-Path "models\optimized_models\face_analyzer_int8.tflite")) {
    Print-Warning "Pre-trained models not found. Please run training scripts after installation."
}

# Set up environment variables
Print-Status "Setting up environment variables..."
@"
# Edge Health Guardian Environment Variables
HEALTH_GUARDIAN_ENV=development
LOG_LEVEL=INFO
CAMERA_DEVICE=0
SAMPLING_RATE=30
ENABLE_ALERTS=true
ALERT_THRESHOLD=0.8
"@ | Out-File -FilePath ".env" -Encoding UTF8

# Create Windows startup script
Print-Status "Creating Windows startup script..."
@"
@echo off
echo 🚀 Starting Edge Health Guardian...
echo 📍 Privacy-First ^| 🔒 On-Device AI ^| 🏥 Health Monitoring

REM Activate virtual environment
call health-guardian-env\Scripts\activate

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%CD%\src

REM Start the main application
python main.py %*

pause
"@ | Out-File -FilePath "start_health_guardian.bat" -Encoding ASCII

# Create PowerShell startup script
Print-Status "Creating PowerShell startup script..."
@"
Write-Host "🚀 Starting Edge Health Guardian..." -ForegroundColor Cyan
Write-Host "📍 Privacy-First | 🔒 On-Device AI | 🏥 Health Monitoring" -ForegroundColor Cyan

# Activate virtual environment
& ".\health-guardian-env\Scripts\Activate.ps1"

# Set environment variables
`$env:PYTHONPATH = "`$env:PYTHONPATH;`$PWD\src"

# Start the main application
python main.py `$args

Read-Host "`nPress Enter to exit"
"@ | Out-File -FilePath "start_health_guardian.ps1" -Encoding UTF8

# Run basic tests
Print-Status "Running basic system tests..."
& "health-guardian-env\Scripts\python" -c "
import sys
try:
    import tensorflow as tf
    import cv2
    import numpy as np
    print('✅ TensorFlow version:', tf.__version__)
    print('✅ OpenCV version:', cv2.__version__)
    print('✅ NumPy version:', np.__version__)
    print('✅ Basic imports successful')
except ImportError as e:
    print('❌ Import error:', e)
    sys.exit(1)
"

# Final instructions
Write-Host "`n🎉 Installation Complete!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "`n📝 Next steps:" -ForegroundColor Yellow
Write-Host "1. Activate the virtual environment:"
Write-Host "   .\health-guardian-env\Scripts\activate" -ForegroundColor White
Write-Host "2. Train the models (if pre-trained models not available):"
Write-Host "   python models\training\train_face_model.py" -ForegroundColor White
Write-Host "   python models\training\train_movement_model.py" -ForegroundColor White
Write-Host "   python models\training\fusion_trainer.py" -ForegroundColor White
Write-Host "3. Convert models to TFLite:"
Write-Host "   python models\conversion\convert_to_tflite.py" -ForegroundColor White
Write-Host "4. Start the health monitoring system:"
Write-Host "   .\start_health_guardian.bat  (or .ps1)" -ForegroundColor White
Write-Host "`n🔧 Windows-specific notes:" -ForegroundColor Yellow
Write-Host "   - Camera access might require administrator privileges"
Write-Host "   - For IMU sensors, use compatible USB devices"
Write-Host "   - Consider using Windows Subsystem for Linux (WSL) for better compatibility"
Write-Host "`n📚 Documentation:"
Write-Host "   Check docs\ for detailed setup and usage instructions" -ForegroundColor White
Write-Host "`n🐛 Troubleshooting:"
Write-Host "   If you encounter issues, check logs\ directory for error logs" -ForegroundColor White

Print-Success "Edge Health Guardian is ready to use on Windows!"
Read-Host "nPress Enter to exit"