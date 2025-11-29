import numpy as np
import threading
from collections import deque
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AdaptiveSensorFusion:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.face_buffer = deque(maxlen=window_size)
        self.movement_buffer = deque(maxlen=window_size)
        self.hr_buffer = deque(maxlen=window_size)

        self.face_confidence = 1.0
        self.movement_confidence = 1.0
        self.hr_confidence = 1.0

        self.lock = threading.Lock()

    def calculate_signal_quality(self, data, sensor_type: str) -> float:
        """Robustly calculate quality of incoming data"""
        try:
            if data is None:
                return 0.1  
                
            if sensor_type == 'face':
                if isinstance(data, dict):
                    # Use face detection success and feature quality
                    success = data.get('success', 0.0)
                    face_detected = data.get('face_detected', 0.0)
                    # Combine success and face detection for confidence
                    return float(max(0.1, min(1.0, success * face_detected)))
                else:
                    return 0.5  
                    
            elif sensor_type == 'movement':
                # For movement data, check if it's a valid dictionary with data
                if isinstance(data, dict) and data:
                    return 0.7  
                else:
                    return 0.3 
            elif sensor_type == 'hr':
                if data is not None and hasattr(data, '__len__'):
                    data_array = np.array(data)
                    if len(data_array) > 5:
                        return float(max(0.1, min(1.0, 1.0 / (1.0 + np.std(data_array)))))
                return 0.1  
                
        except Exception as e:
            logger.warning(f"Signal quality calculation failed for {sensor_type}: {e}")
            return 0.1  

    def update_confidence_scores(self, face_data, movement_data, hr_data: Optional[np.ndarray] = None):
        """Update confidence scores with robust error handling"""
        try:
            with self.lock:
                self.face_confidence = self.calculate_signal_quality(face_data, 'face')
                self.movement_confidence = self.calculate_signal_quality(movement_data, 'movement')
                self.hr_confidence = self.calculate_signal_quality(hr_data, 'hr') if hr_data is not None else 0.0
        except Exception as e:
            logger.error(f"Confidence score update failed: {e}")
            # Set default confidences on error
            self.face_confidence = 0.5
            self.movement_confidence = 0.5
            self.hr_confidence = 0.0

    def temporal_fusion(self, current_features: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Perform temporal fusion with robust error handling"""
        try:
            with self.lock:
                # Safely add features to buffers
                face_features = current_features.get('face', {})
                movement_features = current_features.get('movement', {})
                hr_features = current_features.get('hr')
                
                if isinstance(face_features, dict):
                    self.face_buffer.append(face_features)
                if isinstance(movement_features, dict):
                    self.movement_buffer.append(movement_features)
                if hr_features is not None:
                    self.hr_buffer.append(hr_features)

                def weighted_avg(buffer, confidence):
                    """Calculate weighted average for a buffer"""
                    if not buffer:
                        return None
                    
                    try:
                        # Extract numeric values from dictionaries
                        values = []
                        for item in buffer:
                            if isinstance(item, dict):
                                # Try to extract a meaningful numeric value
                                if 'success' in item:
                                    values.append([item['success']])
                                elif 'face_detected' in item:
                                    values.append([item['face_detected']])
                                elif any(isinstance(v, (int, float)) for v in item.values()):
                                    # Use the first numeric value found
                                    for v in item.values():
                                        if isinstance(v, (int, float)):
                                            values.append([v])
                                            break
                            elif isinstance(item, (int, float)):
                                values.append([item])
                            elif isinstance(item, np.ndarray):
                                values.append(item.flatten()[:1])  
                        
                        if not values:
                            return None
                            
                        # Convert to numpy array for processing
                        values_array = np.array(values)
                        if len(values_array) == 0:
                            return None
                            
                        # Calculate weighted average
                        weights = np.linspace(0.5, 1.0, len(values_array)) * confidence
                        weights /= np.sum(weights)
                        return np.average(values_array, axis=0, weights=weights)
                        
                    except Exception as e:
                        logger.warning(f"Weighted average calculation failed: {e}")
                        return None

                # Calculate fused features
                fused_face = weighted_avg(self.face_buffer, self.face_confidence)
                fused_movement = weighted_avg(self.movement_buffer, self.movement_confidence)
                fused_hr = weighted_avg(self.hr_buffer, self.hr_confidence) if self.hr_confidence > 0.1 else None

                return fused_face, fused_movement, fused_hr
                
        except Exception as e:
            logger.error(f"Temporal fusion failed: {e}")
            return None, None, None

    def get_fusion_weights(self) -> Tuple[float, float, float]:
        """Get normalized fusion weights"""
        try:
            total = self.face_confidence + self.movement_confidence + max(0.1, self.hr_confidence)
            if total == 0:
                return (0.33, 0.33, 0.34)  
            return (
                self.face_confidence / total,
                self.movement_confidence / total,
                max(0.1, self.hr_confidence) / total
            )
        except Exception as e:
            logger.error(f"Fusion weights calculation failed: {e}")
            return (0.33, 0.33, 0.34)  

    def get_system_status(self) -> Dict:
        """Get current system status for debugging"""
        return {
            'face_confidence': self.face_confidence,
            'movement_confidence': self.movement_confidence,
            'hr_confidence': self.hr_confidence,
            'face_buffer_size': len(self.face_buffer),
            'movement_buffer_size': len(self.movement_buffer),
            'hr_buffer_size': len(self.hr_buffer)
        }