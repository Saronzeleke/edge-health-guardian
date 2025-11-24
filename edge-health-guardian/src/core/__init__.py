# src/core/__init__.py
"""
Edge Health Guardian Core Module
On-Device AI Health Monitoring System
"""

__version__ = "1.0.0"
__author__ = "Edge Health Guardian Team"
__description__ = "Privacy-first on-device health anomaly detection"

from .inference_engine import HealthInferenceEngine, ArmOptimizedInference
from .sensor_fusion import AdaptiveSensorFusion
from .anomaly_detector import HealthAnomalyDetector, RealTimeAnomalyDetector

__all__ = [
    'HealthInferenceEngine',
    'ArmOptimizedInference', 
    'AdaptiveSensorFusion',
    'HealthAnomalyDetector',
    'RealTimeAnomalyDetector'
]