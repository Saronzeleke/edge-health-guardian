import argparse
import time
import json 
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from src.core.inference_engine import HealthInferenceEngine
from src.core.sensor_fusion import AdaptiveSensorFusion
from src.sensors.camera_processor import CameraProcessor
from src.sensors.imu_processor import MovementPatternAnalyzer
from src.ui.cli_interface import HealthMonitorCLI
from src.utils.alert_system import AlertSystem

class EdgeHealthGuardian:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Initialize components
        self.inference_engine = HealthInferenceEngine()
        self.sensor_fusion = AdaptiveSensorFusion()
        self.movement_analyzer = MovementPatternAnalyzer()
        self.alert_system = AlertSystem()
        self.ui = HealthMonitorCLI()
        
        self.is_running = False
        self.health_history = []
        self.current_frame = None
        self.callback_count = 0
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        default_config = {
            'camera_id': 0,
            'sampling_rate': 30,
            'alert_thresholds': {
                'stress': 0.7,
                'fatigue': 0.8,
                'anomaly': 0.9
            },
            'ui_refresh_rate': 1.0,
            'data_logging': True
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logging.warning(f"Could not load config file: {e}")
                
        return default_config
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('health_guardian.log'),
                logging.StreamHandler()
            ]
        )
        
    def test_camera_feed(self):
        """Test method to check if camera is working"""
        try:
            cap = cv2.VideoCapture(self.config['camera_id'])
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    logging.info(f"Camera test SUCCESSFUL - frame shape: {frame.shape}")
                    # Show frame briefly
                    cv2.imshow('Camera Test - Press any key', frame)
                    cv2.waitKey(2000)  # Show for 2 seconds
                    cv2.destroyAllWindows()
                else:
                    logging.error("Camera opened but failed to read frame")
                cap.release()
            else:
                logging.error("Camera failed to open")
        except Exception as e:
            logging.error(f"Camera test failed: {e}")
        
    def initialize_system(self):
        """Initialize all system components"""
        logging.info("Initializing Edge Health Guardian...")
        
        try:
            # Initialize models
            self.inference_engine.initialize_models()
            logging.info("AI models loaded successfully")
            
            # TEST: Check camera feed first
            logging.info("Testing camera feed...")
            self.test_camera_feed()
            
            # Initialize camera
            self.camera_processor = CameraProcessor(self.config['camera_id'])
            logging.info("Camera initialized")
            
            # Setup UI
            self.ui.initialize()
            logging.info("UI initialized")
            
            return True
            
        except Exception as e:
            logging.error(f"Initialization failed: {e}")
            return False

    def camera_frame_callback(self, *args):
        """
        FLEXIBLE callback that handles both old and new camera processor formats
        """
        self.callback_count += 1
        
        # Only log every 30th callback to avoid spam
        if self.callback_count % 30 == 1:
            logging.info(f"Callback #{self.callback_count} received {len(args)} arguments")
        
        # Determine which callback format we're receiving
        if len(args) == 1 and isinstance(args[0], dict):
            # NEW format: single dictionary
            data = args[0]
            frame_tensor = data.get('frame_tensor')
            expression_features = data.get('features', {})
            metadata = data.get('metadata', {})
            if self.callback_count == 1:
                logging.info("Using NEW callback format (1 argument)")
        elif len(args) == 2:
            # OLD format: frame, expression_features
            frame, expression_features = args
            frame_tensor = self._frame_to_tensor(frame)
            metadata = {}
            if self.callback_count == 1:
                logging.info("Using OLD callback format (2 arguments)")
        else:
            logging.error(f"Unexpected callback arguments: {args}")
            return

        if not self.is_running:
            return

        try:
            # DEBUG: Log feature extraction status
            face_detected = expression_features.get('face_detected', 0)
            success = expression_features.get('success', 0)
            
            if self.callback_count % 30 == 1:  
                logging.info(f"Frame #{self.callback_count}: Face detected: {face_detected}, Success: {success}")
            
            # Convert tensor back to frame for display if needed
            display_frame = self._tensor_to_display_frame(frame_tensor)
            if display_frame is not None:
                self.current_frame = display_frame
                if self.callback_count == 1:
                    logging.info("Frame converted for display")
            
            # Validate features before processing
            if not self._validate_expression_features(expression_features):
                if self.callback_count % 30 == 1:  # Reduce spam
                    logging.warning("Invalid expression features, skipping processing")
                return
            
            if self.callback_count % 30 == 1:
                logging.info("Valid features detected, processing health data...")
            
            # Get movement features 
            movement_features = self.movement_analyzer.extract_movement_features()
            
            # Update sensor fusion with validated data
            self.sensor_fusion.update_confidence_scores(
                expression_features, movement_features
            )
            
            # Perform temporal fusion with error handling
            fused_face, fused_movement, _ = self.sensor_fusion.temporal_fusion({
                'face': expression_features,
                'movement': movement_features
            })
            
            # Check if fusion returned valid data
            if fused_face is None or fused_movement is None:
                if self.callback_count % 30 == 1:
                    logging.debug("Sensor fusion returned None, skipping inference")
                return
            
            # Run health inference with fused features
            health_results = self.inference_engine.process_health_data(
                fused_face, fused_movement
            )
            
            # LOG HEALTH RESULTS - THIS IS THE KEY ADDITION!
            if self.callback_count % 10 == 1:  # Log every 10th health result
                stress = health_results.get('stress_score', 0)
                fatigue = health_results.get('fatigue_level', 0)
                anomaly = health_results.get('anomaly_confidence', 0)
                mock_flag = " [MOCK]" if health_results.get('mock_data', False) else ""
                logging.info(f"HEALTH RESULTS{mock_flag} - Stress: {stress:.3f}, Fatigue: {fatigue:.3f}, Anomaly: {anomaly:.3f}")
            
            # Add metadata to health results
            health_results.update({
                'timestamp': time.time(),
                'frame_count': metadata.get('frame_count', 0),
                'fps': metadata.get('fps', 0)
            })
            
            # Update health history
            self.health_history.append({
                'timestamp': time.time(),
                'results': health_results,
                'sensor_confidence': self.sensor_fusion.get_fusion_weights(),
                'expression_features': self._extract_key_features(expression_features)
            })
            
            # Keep only recent history (last 5 minutes at ~1Hz)
            max_history_size = 300
            if len(self.health_history) > max_history_size:
                self.health_history = self.health_history[-max_history_size:]
            
            # Check for health alerts
            self._check_health_alerts(health_results)
            
            # Update UI with current frame and results
            self.ui.update_display(health_results, self.health_history, self.current_frame)
            
        except Exception as e:
            logging.error(f"Camera frame callback processing failed: {e}")

    def _frame_to_tensor(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to 4D tensor format for ML models"""
        try:
            if frame is None:
                logging.warning("Received None frame in tensor conversion")
                return np.zeros((1, 480, 640, 3), dtype=np.float32)
            
            # Normalize pixel values to [0, 1]
            frame_normalized = frame.astype(np.float32) / 255.0
            
            # Add batch dimension: (1, height, width, channels)
            tensor = np.expand_dims(frame_normalized, axis=0)
            return tensor
        except Exception as e:
            logging.error(f"Frame to tensor conversion failed: {e}")
            return np.zeros((1, 480, 640, 3), dtype=np.float32)

    def _tensor_to_display_frame(self, tensor: np.ndarray) -> Optional[np.ndarray]:
        """Convert 4D tensor back to displayable frame"""
        try:
            if tensor is None:
                return None
                
            if len(tensor.shape) != 4:
                if tensor.shape == (480, 640, 3):
                    tensor = np.expand_dims(tensor, axis=0)
                else:
                    return None
                    
            # Remove batch dimension and convert back to uint8
            frame = tensor[0]
            
            if frame.dtype == np.float32 and frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            elif frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
                
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
            return frame
            
        except Exception as e:
            logging.error(f"Tensor to frame conversion failed: {e}")
            return None

    def _validate_expression_features(self, features: Dict) -> bool:
        """Validate that expression features are usable"""
        try:
            if not features or features.get('success', 0) < 0.5:
                return False
            
            required_keys = ['face_detected', 'eyebrow_tension', 'eye_aperture']
            if not all(key in features for key in required_keys):
                return False
            
            for key in required_keys:
                value = features.get(key)
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    return False
            
            return True
            
        except Exception:
            return False

    def _extract_key_features(self, features: Dict) -> Dict:
        """Extract key features for health monitoring"""
        try:
            return {
                'eyebrow_tension': features.get('eyebrow_tension', 0),
                'eye_aperture': features.get('eye_aperture', 0),
                'mouth_tension': features.get('mouth_tension', 0),
                'facial_symmetry': features.get('facial_symmetry', 0),
                'micro_expression_intensity': features.get('micro_expression_intensity', 0),
                'face_detected': features.get('face_detected', 0)
            }
        except Exception as e:
            logging.warning(f"Key feature extraction failed: {e}")
            return {}

    def _check_health_alerts(self, health_results: Dict):
        """Check if health metrics exceed alert thresholds"""
        thresholds = self.config['alert_thresholds']
        
        alerts = []
        
        try:
            stress_score = health_results.get('stress_score', 0)
            fatigue_level = health_results.get('fatigue_level', 0)
            anomaly_confidence = health_results.get('anomaly_confidence', 0)
            
            if stress_score > thresholds['stress']:
                alerts.append({
                    'type': 'stress',
                    'level': 'high',
                    'value': stress_score,
                    'message': 'High stress level detected. Consider taking a break.'
                })
                logging.warning(f"STRESS ALERT: {stress_score:.3f} (threshold: {thresholds['stress']})")
                
            if fatigue_level > thresholds['fatigue']:
                alerts.append({
                    'type': 'fatigue', 
                    'level': 'high',
                    'value': fatigue_level,
                    'message': 'High fatigue detected. Rest recommended.'
                })
                logging.warning(f"FATIGUE ALERT: {fatigue_level:.3f} (threshold: {thresholds['fatigue']})")
                
            if anomaly_confidence > thresholds['anomaly']:
                alerts.append({
                    'type': 'anomaly',
                    'level': 'critical', 
                    'value': anomaly_confidence,
                    'message': 'Health anomaly detected! Please seek medical attention.'
                })
                logging.warning(f"ANOMALY ALERT: {anomaly_confidence:.3f} (threshold: {thresholds['anomaly']})")
            
            # Trigger alerts
            for alert in alerts:
                self.alert_system.trigger_alert(alert)
                
        except Exception as e:
            logging.error(f"Alert checking failed: {e}")
    
    def start_monitoring(self):
        """Start the health monitoring system"""
        if not self.initialize_system():
            return False
            
        self.is_running = True
        logging.info("Starting health monitoring...")
        
        try:
            # Start camera processing with the updated callback
            self.camera_processor.start_capture(self.camera_frame_callback)
            
            logging.info("Health monitoring system active")
            logging.info("Make sure your face is visible to the camera for detection...")
            logging.info("You should see HEALTH RESULTS in the logs every few seconds!")
            
            # Main monitoring loop
            while self.is_running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received")
            self.stop_monitoring()
        except Exception as e:
            logging.error(f"Monitoring loop error: {e}")
            self.stop_monitoring()
            
        return True
    
    def stop_monitoring(self):
        """Stop the health monitoring system gracefully"""
        self.is_running = False
        
        try:
            if hasattr(self, 'camera_processor'):
                self.camera_processor.stop_capture()
                
            # Show final summary
            logging.info(f"Health monitoring stopped. Processed {self.callback_count} frames.")
            if self.health_history:
                avg_stress = np.mean([h['results'].get('stress_score', 0) for h in self.health_history])
                avg_fatigue = np.mean([h['results'].get('fatigue_level', 0) for h in self.health_history])
                logging.info(f"Final averages - Stress: {avg_stress:.3f}, Fatigue: {avg_fatigue:.3f}")
                
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
    
    def get_health_summary(self) -> Dict:
        """Get summary of recent health metrics"""
        if not self.health_history:
            return {}
            
        try:
            recent_results = [entry['results'] for entry in self.health_history[-30:] if 'results' in entry]
            
            if not recent_results:
                return {}
            
            summary = {
                'average_stress': np.mean([r.get('stress_score', 0) for r in recent_results]),
                'average_fatigue': np.mean([r.get('fatigue_level', 0) for r in recent_results]),
                'trend_stress': self._calculate_trend([r.get('stress_score', 0) for r in recent_results]),
                'trend_fatigue': self._calculate_trend([r.get('fatigue_level', 0) for r in recent_results]),
                'anomaly_count': sum(1 for r in recent_results if r.get('anomaly_confidence', 0) > 0.7),
                'total_samples': len(recent_results)
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"Health summary calculation failed: {e}")
            return {}

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from recent values"""
        try:
            if len(values) < 2:
                return 'stable'
                
            clean_values = [v for v in values if np.isfinite(v)]
            if len(clean_values) < 2:
                return 'stable'
                
            x = np.arange(len(clean_values))
            slope = np.polyfit(x, clean_values, 1)[0]
            
            if slope > 0.01:
                return 'increasing'
            elif slope < -0.01:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception:
            return 'stable'

def main():
    parser = argparse.ArgumentParser(description='Edge Health Guardian - On-Device Health Monitoring')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with simulated data')
    
    args = parser.parse_args()
    
    # Initialize and start the system
    health_guardian = EdgeHealthGuardian(args.config)
    
    print("🚀 Starting Edge Health Guardian...")
    print(" Privacy-First |  On-Device AI |  Health Monitoring")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        health_guardian.start_monitoring()
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        print(f"System crashed: {e}")

if __name__ == "__main__":
    main()