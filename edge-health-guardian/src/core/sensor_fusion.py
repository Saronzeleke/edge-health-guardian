# src/core/sensor_fusion.py
import numpy as np
from scipy import signal
from typing import Dict, List, Tuple
import threading
from collections import deque

class AdaptiveSensorFusion:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.face_data_buffer = deque(maxlen=window_size)
        self.movement_data_buffer = deque(maxlen=window_size)
        self.hr_data_buffer = deque(maxlen=window_size)
        
        self.face_confidence = 1.0
        self.movement_confidence = 1.0
        self.hr_confidence = 1.0
        
        self.lock = threading.Lock()
        
    def calculate_signal_quality(self, data: np.ndarray, sensor_type: str) -> float:
        """Calculate signal quality for adaptive weighting"""
        if sensor_type == 'face':
            # Calculate brightness consistency, motion blur, etc.
            if len(data.shape) == 3:  # Image data
                brightness_std = np.std(data)
                # Normalize to 0-1 range (lower is better for std in this context)
                quality = 1.0 / (1.0 + brightness_std)
                return max(0.1, min(1.0, quality))
                
        elif sensor_type == 'movement':
            # Calculate signal-to-noise ratio for IMU data
            if len(data) > 10:
                noise_floor = np.std(data[:10])  # First 10 samples as noise reference
                signal_power = np.std(data)
                snr = signal_power / (noise_floor + 1e-8)
                quality = 1.0 - 1.0 / (1.0 + snr)
                return max(0.1, min(1.0, quality))
                
        elif sensor_type == 'hr':
            # Calculate heart rate signal quality
            if len(data) > 5:
                hr_std = np.std(data)
                quality = 1.0 / (1.0 + hr_std)
                return max(0.1, min(1.0, quality))
                
        return 0.5  # Default medium confidence
    
    def update_confidence_scores(self, face_data: np.ndarray, 
                               movement_data: np.ndarray,
                               hr_data: Optional[np.ndarray] = None):
        """Update confidence scores based on recent data quality"""
        with self.lock:
            self.face_confidence = self.calculate_signal_quality(face_data, 'face')
            self.movement_confidence = self.calculate_signal_quality(movement_data, 'movement')
            
            if hr_data is not None:
                self.hr_confidence = self.calculate_signal_quality(hr_data, 'hr')
            else:
                self.hr_confidence = 0.0
    
    def temporal_fusion(self, current_features: Dict) -> np.ndarray:
        """Fuse data across time domain with confidence weighting"""
        with self.lock:
            # Add current data to buffers
            self.face_data_buffer.append(current_features['face'])
            self.movement_data_buffer.append(current_features['movement'])
            if current_features.get('hr') is not None:
                self.hr_data_buffer.append(current_features['hr'])
            
            # Apply confidence-weighted moving average
            if len(self.face_data_buffer) > 0:
                face_weights = np.linspace(0.5, 1.0, len(self.face_data_buffer)) * self.face_confidence
                face_weights /= np.sum(face_weights)
                fused_face = np.average(list(self.face_data_buffer), axis=0, weights=face_weights)
            else:
                fused_face = current_features['face']
            
            if len(self.movement_data_buffer) > 0:
                movement_weights = np.linspace(0.5, 1.0, len(self.movement_data_buffer)) * self.movement_confidence
                movement_weights /= np.sum(movement_weights)
                fused_movement = np.average(list(self.movement_data_buffer), axis=0, weights=movement_weights)
            else:
                fused_movement = current_features['movement']
            
            fused_hr = None
            if len(self.hr_data_buffer) > 0 and self.hr_confidence > 0.1:
                hr_weights = np.linspace(0.5, 1.0, len(self.hr_data_buffer)) * self.hr_confidence
                hr_weights /= np.sum(hr_weights)
                fused_hr = np.average(list(self.hr_data_buffer), axis=0, weights=hr_weights)
            
            return fused_face, fused_movement, fused_hr
    
    def get_fusion_weights(self) -> Tuple[float, float, float]:
        """Get current confidence weights for sensor fusion"""
        total_confidence = self.face_confidence + self.movement_confidence + max(0.1, self.hr_confidence)
        
        face_weight = self.face_confidence / total_confidence
        movement_weight = self.movement_confidence / total_confidence
        hr_weight = max(0.1, self.hr_confidence) / total_confidence
        
        return face_weight, movement_weight, hr_weight