# src/core/anomaly_detector.py
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple
from collections import deque
import json

class HealthAnomalyDetector:
    """Advanced anomaly detection for health monitoring"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.health_history = deque(maxlen=window_size)
        self.baseline_established = False
        self.baseline_stats = {}
        self.anomaly_threshold = 0.85
        
        # Autoencoder for novelty detection
        self.autoencoder = None
        self.reconstruction_errors = deque(maxlen=window_size)
        
    def build_autoencoder(self, input_dim: int):
        """Build autoencoder model for anomaly detection"""
        # Encoder
        encoder_input = tf.keras.Input(shape=(input_dim,))
        encoded = tf.keras.layers.Dense(64, activation='relu')(encoder_input)
        encoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
        encoded = tf.keras.layers.Dense(16, activation='relu')(encoded)
        
        # Decoder
        decoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(64, activation='relu')(decoded)
        decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        self.autoencoder = tf.keras.Model(encoder_input, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        return self.autoencoder
    
    def update_health_history(self, health_data: Dict):
        """Update health history with new data point"""
        features = self._extract_features(health_data)
        self.health_history.append(features)
        
        # Update baseline after collecting enough data
        if len(self.health_history) >= 50 and not self.baseline_established:
            self._establish_baseline()
        
        # Update reconstruction errors if autoencoder is available
        if self.autoencoder and len(self.health_history) >= 10:
            self._update_reconstruction_errors()
    
    def _extract_features(self, health_data: Dict) -> np.ndarray:
        """Extract relevant features for anomaly detection"""
        features = []
        
        # Stress-related features
        if 'stress_score' in health_data:
            features.append(health_data['stress_score'])
        
        # Fatigue-related features
        if 'fatigue_level' in health_data:
            features.append(health_data['fatigue_level'])
        
        # Sensor confidence features
        if 'sensor_confidence' in health_data:
            conf = health_data['sensor_confidence']
            features.extend([conf.get('face', 0.5), conf.get('movement', 0.5), conf.get('hr', 0.0)])
        
        # Inference time features (potential system issues)
        if 'inference_times' in health_data:
            times = health_data['inference_times']
            total_time = sum(times.values())
            features.append(total_time)
        
        return np.array(features)
    
    def _establish_baseline(self):
        """Establish baseline statistics from normal operation"""
        if len(self.health_history) < 50:
            return
        
        history_array = np.array(list(self.health_history))
        
        self.baseline_stats = {
            'mean': np.mean(history_array, axis=0),
            'std': np.std(history_array, axis=0),
            'min': np.min(history_array, axis=0),
            'max': np.max(history_array, axis=0),
            'covariance': np.cov(history_array.T)
        }
        
        self.baseline_established = True
        print("✅ Health baseline established")
    
    def _update_reconstruction_errors(self):
        """Update autoencoder reconstruction errors"""
        if not self.autoencoder or len(self.health_history) < 10:
            return
        
        # Use recent data for reconstruction
        recent_data = np.array(list(self.health_history)[-10:])
        
        # Normalize data
        if self.baseline_established:
            normalized_data = (recent_data - self.baseline_stats['mean']) / (self.baseline_stats['std'] + 1e-8)
        else:
            normalized_data = recent_data
        
        # Get reconstructions and calculate errors
        reconstructions = self.autoencoder.predict(normalized_data, verbose=0)
        errors = np.mean(np.square(normalized_data - reconstructions), axis=1)
        
        self.reconstruction_errors.extend(errors)
    
    def detect_anomalies(self, current_health: Dict) -> Dict:
        """Detect anomalies in current health data"""
        if not self.baseline_established:
            return {'anomaly_confidence': 0.0, 'anomaly_type': 'baseline_not_established'}
        
        current_features = self._extract_features(current_health)
        
        # Multiple anomaly detection methods
        statistical_anomaly = self._statistical_anomaly_detection(current_features)
        trend_anomaly = self._trend_anomaly_detection()
        reconstruction_anomaly = self._reconstruction_anomaly_detection(current_features)
        
        # Combine anomaly scores
        anomaly_scores = [
            statistical_anomaly,
            trend_anomaly,
            reconstruction_anomaly
        ]
        
        # Remove None values
        valid_scores = [score for score in anomaly_scores if score is not None]
        
        if not valid_scores:
            final_confidence = 0.0
        else:
            final_confidence = np.mean(valid_scores)
        
        # Determine anomaly type
        anomaly_type = self._classify_anomaly_type(
            statistical_anomaly, trend_anomaly, reconstruction_anomaly
        )
        
        return {
            'anomaly_confidence': float(final_confidence),
            'anomaly_type': anomaly_type,
            'statistical_anomaly': statistical_anomaly,
            'trend_anomaly': trend_anomaly,
            'reconstruction_anomaly': reconstruction_anomaly
        }
    
    def _statistical_anomaly_detection(self, current_features: np.ndarray) -> float:
        """Statistical anomaly detection using Mahalanobis distance"""
        try:
            # Calculate Mahalanobis distance
            diff = current_features - self.baseline_stats['mean']
            cov_inv = np.linalg.pinv(self.baseline_stats['covariance'])
            mahalanobis_dist = np.sqrt(diff.T @ cov_inv @ diff)
            
            # Convert to anomaly score (0-1)
            # Assuming normal distribution, distance > 3 is anomalous
            anomaly_score = min(1.0, mahalanobis_dist / 3.0)
            return anomaly_score
            
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to simple z-score method
            z_scores = np.abs((current_features - self.baseline_stats['mean']) / (self.baseline_stats['std'] + 1e-8))
            max_z_score = np.max(z_scores)
            return min(1.0, max_z_score / 3.0)
    
    def _trend_anomaly_detection(self) -> float:
        """Detect anomalies in health trends"""
        if len(self.health_history) < 20:
            return 0.0
        
        recent_history = list(self.health_history)[-20:]
        
        # Calculate trends for each feature
        trend_scores = []
        for feature_idx in range(len(recent_history[0])):
            feature_values = [point[feature_idx] for point in recent_history]
            
            # Calculate trend using linear regression
            x = np.arange(len(feature_values))
            slope = np.polyfit(x, feature_values, 1)[0]
            
            # Normalize slope to anomaly score
            trend_magnitude = abs(slope) * len(feature_values)
            trend_score = min(1.0, trend_magnitude / 2.0)  # Adjust divisor based on expected variation
            trend_scores.append(trend_score)
        
        return np.mean(trend_scores)
    
    def _reconstruction_anomaly_detection(self, current_features: np.ndarray) -> float:
        """Anomaly detection using autoencoder reconstruction error"""
        if not self.autoencoder or len(self.reconstruction_errors) < 10:
            return 0.0
        
        # Normalize current features
        if self.baseline_established:
            normalized_current = (current_features - self.baseline_stats['mean']) / (self.baseline_stats['std'] + 1e-8)
        else:
            normalized_current = current_features
        
        # Get reconstruction error for current point
        reconstruction = self.autoencoder.predict(
            normalized_current.reshape(1, -1), verbose=0
        )
        current_error = np.mean(np.square(normalized_current - reconstruction.flatten()))
        
        # Compare with historical errors
        if len(self.reconstruction_errors) > 0:
            mean_error = np.mean(self.reconstruction_errors)
            std_error = np.std(self.reconstruction_errors)
            
            if std_error > 0:
                z_score = (current_error - mean_error) / std_error
                anomaly_score = min(1.0, max(0.0, z_score / 3.0))
                return anomaly_score
        
        return 0.0
    
    def _classify_anomaly_type(self, statistical_score: float, trend_score: float, reconstruction_score: float) -> str:
        """Classify the type of anomaly based on detection scores"""
        scores = {
            'statistical': statistical_score or 0.0,
            'trend': trend_score or 0.0,
            'reconstruction': reconstruction_score or 0.0
        }
        
        # Find the dominant anomaly type
        dominant_type = max(scores.items(), key=lambda x: x[1])
        
        if dominant_type[1] < 0.3:
            return 'normal'
        
        anomaly_types = {
            'statistical': 'sudden_change',
            'trend': 'gradual_deterioration',
            'reconstruction': 'novel_pattern'
        }
        
        return anomaly_types.get(dominant_type[0], 'unknown')
    
    def train_autoencoder(self, training_data: List[Dict], epochs: int = 50):
        """Train the autoencoder on normal health data"""
        if not training_data:
            return
        
        # Extract features from training data
        features_list = [self._extract_features(data) for data in training_data]
        features_array = np.array(features_list)
        
        # Build autoencoder if not already built
        if self.autoencoder is None:
            self.build_autoencoder(features_array.shape[1])
        
        # Normalize training data
        if self.baseline_established:
            normalized_data = (features_array - self.baseline_stats['mean']) / (self.baseline_stats['std'] + 1e-8)
        else:
            # Calculate normalization from training data
            self.baseline_stats['mean'] = np.mean(features_array, axis=0)
            self.baseline_stats['std'] = np.std(features_array, axis=0)
            normalized_data = (features_array - self.baseline_stats['mean']) / (self.baseline_stats['std'] + 1e-8)
            self.baseline_established = True
        
        # Train autoencoder
        self.autoencoder.fit(
            normalized_data, normalized_data,
            epochs=epochs,
            batch_size=32,
            shuffle=True,
            verbose=0
        )
        
        print(f"✅ Autoencoder trained on {len(training_data)} samples")
    
    def get_health_summary(self) -> Dict:
        """Get summary of current health monitoring state"""
        return {
            'baseline_established': self.baseline_established,
            'history_size': len(self.health_history),
            'reconstruction_errors_size': len(self.reconstruction_errors),
            'autoencoder_trained': self.autoencoder is not None,
            'recent_anomaly_scores': list(self.reconstruction_errors)[-5:] if self.reconstruction_errors else []
        }
    
    def save_state(self, filepath: str):
        """Save anomaly detector state"""
        if not self.baseline_established:
            return
        
        state = {
            'baseline_stats': {
                'mean': self.baseline_stats['mean'].tolist(),
                'std': self.baseline_stats['std'].tolist(),
                'covariance': self.baseline_stats['covariance'].tolist()
            },
            'window_size': self.window_size,
            'health_history': [arr.tolist() for arr in self.health_history],
            'reconstruction_errors': list(self.reconstruction_errors)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f)
    
    def load_state(self, filepath: str):
        """Load anomaly detector state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.baseline_stats = {
                'mean': np.array(state['baseline_stats']['mean']),
                'std': np.array(state['baseline_stats']['std']),
                'covariance': np.array(state['baseline_stats']['covariance'])
            }
            
            self.health_history = deque([np.array(arr) for arr in state['health_history']], maxlen=self.window_size)
            self.reconstruction_errors = deque(state['reconstruction_errors'], maxlen=self.window_size)
            self.baseline_established = True
            
            print("✅ Anomaly detector state loaded")
            
        except Exception as e:
            print(f"❌ Failed to load anomaly detector state: {e}")

class RealTimeAnomalyDetector:
    """Real-time anomaly detection with adaptive thresholds"""
    
    def __init__(self):
        self.health_detector = HealthAnomalyDetector()
        self.alert_history = deque(maxlen=100)
        self.adaptive_threshold = 0.7
        self.false_positive_count = 0
        self.true_positive_count = 0
        
    def process_health_update(self, health_data: Dict) -> Dict:
        """Process new health data and return anomaly results"""
        # Update health history
        self.health_detector.update_health_history(health_data)
        
        # Detect anomalies
        anomaly_result = self.health_detector.detect_anomalies(health_data)
        
        # Update adaptive threshold
        self._update_adaptive_threshold(anomaly_result)
        
        # Record alert
        if anomaly_result['anomaly_confidence'] > self.adaptive_threshold:
            alert_record = {
                'timestamp': health_data.get('timestamp', 'unknown'),
                'anomaly_confidence': anomaly_result['anomaly_confidence'],
                'anomaly_type': anomaly_result['anomaly_type'],
                'health_data': health_data
            }
            self.alert_history.append(alert_record)
        
        return {
            **anomaly_result,
            'adaptive_threshold': self.adaptive_threshold,
            'alert_triggered': anomaly_result['anomaly_confidence'] > self.adaptive_threshold
        }
    
    def _update_adaptive_threshold(self, anomaly_result: Dict):
        """Adaptively update anomaly threshold based on recent performance"""
        # Simple adaptive logic - in real implementation, use more sophisticated approach
        recent_alerts = list(self.alert_history)[-10:]  # Last 10 alerts
        
        if len(recent_alerts) >= 5:
            # Calculate false positive rate (simplified)
            # In real implementation, you'd have ground truth data
            avg_confidence = np.mean([alert['anomaly_confidence'] for alert in recent_alerts])
            
            # Adjust threshold based on recent confidence levels
            if avg_confidence > 0.9:
                # Too many high-confidence alerts, might be too sensitive
                self.adaptive_threshold = min(0.9, self.adaptive_threshold + 0.05)
            elif avg_confidence < 0.5:
                # Too few alerts, might be missing anomalies
                self.adaptive_threshold = max(0.5, self.adaptive_threshold - 0.05)
    
    def get_alert_summary(self) -> Dict:
        """Get summary of recent alerts"""
        recent_alerts = list(self.alert_history)[-20:]  # Last 20 alerts
        
        return {
            'total_alerts': len(self.alert_history),
            'recent_alerts': len(recent_alerts),
            'average_confidence': np.mean([alert['anomaly_confidence'] for alert in recent_alerts]) if recent_alerts else 0,
            'most_common_type': self._get_most_common_anomaly_type(recent_alerts),
            'adaptive_threshold': self.adaptive_threshold
        }
    
    def _get_most_common_anomaly_type(self, alerts: List[Dict]) -> str:
        """Get the most common anomaly type from alerts"""
        if not alerts:
            return 'none'
        
        types = [alert['anomaly_type'] for alert in alerts]
        return max(set(types), key=types.count)