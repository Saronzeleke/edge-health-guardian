# main.py
import argparse
import time
import json
import logging
from typing import Dict, Any

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
        
    def initialize_system(self):
        """Initialize all system components"""
        logging.info("Initializing Edge Health Guardian...")
        
        try:
            # Initialize models
            self.inference_engine.initialize_models()
            logging.info("AI models loaded successfully")
            
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
    
    def camera_frame_callback(self, frame, expression_features):
        """Callback for processed camera frames"""
        if not self.is_running:
            return
            
        # Get movement features (in real implementation, this would come from IMU)
        movement_features = self.movement_analyzer.extract_movement_features()
        
        # Update sensor fusion confidence
        self.sensor_fusion.update_confidence_scores(
            expression_features, movement_features
        )
        
        # Perform temporal fusion
        fused_face, fused_movement, _ = self.sensor_fusion.temporal_fusion({
            'face': expression_features,
            'movement': movement_features
        })
        
        # Run health inference
        health_results = self.inference_engine.process_health_data(
            fused_face, fused_movement
        )
        
        # Update health history
        self.health_history.append({
            'timestamp': time.time(),
            'results': health_results,
            'sensor_confidence': self.sensor_fusion.get_fusion_weights()
        })
        
        # Keep only recent history
        if len(self.health_history) > 300:  # 5 minutes at 1Hz
            self.health_history.pop(0)
        
        # Check alerts
        self._check_health_alerts(health_results)
        
        # Update UI
        self.ui.update_display(health_results, self.health_history)
    
    def _check_health_alerts(self, health_results: Dict):
        """Check if health metrics exceed alert thresholds"""
        thresholds = self.config['alert_thresholds']
        
        alerts = []
        
        if health_results['stress_score'] > thresholds['stress']:
            alerts.append({
                'type': 'stress',
                'level': 'high',
                'value': health_results['stress_score'],
                'message': 'High stress level detected. Consider taking a break.'
            })
            
        if health_results['fatigue_level'] > thresholds['fatigue']:
            alerts.append({
                'type': 'fatigue', 
                'level': 'high',
                'value': health_results['fatigue_level'],
                'message': 'High fatigue detected. Rest recommended.'
            })
            
        if health_results['anomaly_confidence'] > thresholds['anomaly']:
            alerts.append({
                'type': 'anomaly',
                'level': 'critical', 
                'value': health_results['anomaly_confidence'],
                'message': 'Health anomaly detected! Please seek medical attention.'
            })
        
        # Trigger alerts
        for alert in alerts:
            self.alert_system.trigger_alert(alert)
            logging.warning(f"ALERT: {alert['type']} - {alert['message']}")
    
    def start_monitoring(self):
        """Start the health monitoring system"""
        if not self.initialize_system():
            return False
            
        self.is_running = True
        logging.info("Starting health monitoring...")
        
        # Start camera processing
        self.camera_processor.start_capture(self.camera_frame_callback)
        
        # Main loop
        try:
            while self.is_running:
                time.sleep(0.1)  # Small sleep to prevent CPU overload
                
        except KeyboardInterrupt:
            self.stop_monitoring()
            
        return True
    
    def stop_monitoring(self):
        """Stop the health monitoring system"""
        self.is_running = False
        self.camera_processor.stop_capture()
        logging.info("Health monitoring stopped")
    
    def get_health_summary(self) -> Dict:
        """Get summary of recent health metrics"""
        if not self.health_history:
            return {}
            
        recent_results = [entry['results'] for entry in self.health_history[-30:]]  # Last 30 seconds
        
        summary = {
            'average_stress': np.mean([r['stress_score'] for r in recent_results]),
            'average_fatigue': np.mean([r['fatigue_level'] for r in recent_results]),
            'trend_stress': self._calculate_trend([r['stress_score'] for r in recent_results]),
            'trend_fatigue': self._calculate_trend([r['fatigue_level'] for r in recent_results]),
            'anomaly_count': sum(1 for r in recent_results if r['anomaly_confidence'] > 0.7)
        }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from recent values"""
        if len(values) < 2:
            return 'stable'
            
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'

def main():
    parser = argparse.ArgumentParser(description='Edge Health Guardian - On-Device Health Monitoring')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with simulated data')
    
    args = parser.parse_args()
    
    # Initialize and start the system
    health_guardian = EdgeHealthGuardian(args.config)
    
    print("🚀 Starting Edge Health Guardian...")
    print("📍 Privacy-First | 🔒 On-Device AI | 🏥 Health Monitoring")
    print("Press Ctrl+C to stop monitoring\n")
    
    health_guardian.start_monitoring()

if __name__ == "__main__":
    main()