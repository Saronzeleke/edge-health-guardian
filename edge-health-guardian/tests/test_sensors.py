# tests/test_sensors.py
import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sensors.imu_processor import MovementPatternAnalyzer
from sensors.hr_processor import HeartRateAnalyzer
from sensors.camera_processor import FaceMicroExpressionAnalyzer

class TestIMUProcessor(unittest.TestCase):
    
    def setUp(self):
        self.imu_processor = MovementPatternAnalyzer(sampling_rate=100)
    
    def test_movement_feature_extraction(self):
        """Test movement feature extraction"""
        # Generate synthetic IMU data
        time_steps = 500
        t = np.linspace(0, 5, time_steps)
        
        # Simulate normal movement
        accel_data = np.column_stack([
            np.sin(2 * np.pi * 1 * t) + 0.1 * np.random.normal(size=time_steps),  # X
            np.cos(2 * np.pi * 1 * t) + 0.1 * np.random.normal(size=time_steps),  # Y
            9.8 + 0.05 * np.random.normal(size=time_steps)  # Z (gravity)
        ])
        
        gyro_data = np.column_stack([
            0.1 * np.sin(2 * np.pi * 2 * t) + 0.05 * np.random.normal(size=time_steps),
            0.1 * np.cos(2 * np.pi * 2 * t) + 0.05 * np.random.normal(size=time_steps),
            0.02 * np.random.normal(size=time_steps)
        ])
        
        # Add data to processor
        for i in range(time_steps):
            self.imu_processor.add_imu_data(accel_data[i], gyro_data[i])
        
        # Extract features
        features = self.imu_processor.extract_movement_features()
        
        # Check that features are extracted
        self.assertGreater(len(features), 0)
        self.assertIn('accel_sma', features)
        self.assertIn('gyro_sma', features)
        
        print("✅ Movement feature extraction test passed")
    
    def test_tremor_detection(self):
        """Test tremor detection in movement data"""
        # Generate data with simulated tremor
        time_steps = 1000
        t = np.linspace(0, 10, time_steps)
        
        # Add high-frequency tremor (8Hz)
        tremor_freq = 8
        tremor_component = 0.2 * np.sin(2 * np.pi * tremor_freq * t)
        
        accel_data = np.column_stack([
            np.sin(2 * np.pi * 1 * t) + tremor_component,
            np.cos(2 * np.pi * 1 * t) + tremor_component,
            9.8 + 0.1 * np.random.normal(size=time_steps)
        ])
        
        gyro_data = np.column_stack([
            tremor_component,
            tremor_component,
            0.05 * np.random.normal(size=time_steps)
        ])
        
        for i in range(time_steps):
            self.imu_processor.add_imu_data(accel_data[i], gyro_data[i])
        
        features = self.imu_processor.extract_movement_features()
        
        # Should detect increased tremor intensity
        self.assertIn('accel_axis0_tremor_intensity', features)
        tremor_intensity = features['accel_axis0_tremor_intensity']
        
        self.assertGreater(tremor_intensity, 0.05)  # Should detect tremor
        print("✅ Tremor detection test passed")

class TestHeartRateProcessor(unittest.TestCase):
    
    def setUp(self):
        self.hr_processor = HeartRateAnalyzer(sampling_rate=100)
    
    def test_hrv_calculation(self):
        """Test heart rate variability calculation"""
        # Generate synthetic R-peak data
        sampling_rate = 100
        duration = 60  # 60 seconds
        total_samples = duration * sampling_rate
        
        # Simulate normal heart rate with some variability
        base_rr = 0.8  # 750ms RR interval (80 BPM)
        rr_variability = 0.05  # 5% variability
        
        r_peaks = []
        current_time = 0
        
        while current_time < total_samples:
            # Add some random variability to RR intervals
            rr_interval = base_rr * (1 + rr_variability * np.random.normal())
            current_time += int(rr_interval * sampling_rate)
            if current_time < total_samples:
                r_peaks.append(current_time)
        
        self.hr_processor.last_r_peaks = r_peaks
        hrv_features = self.hr_processor._calculate_hrv_metrics(r_peaks)
        
        # Check HRV metrics
        self.assertIn('hrv_mean_rr', hrv_features)
        self.assertIn('hrv_rmssd', hrv_features)
        self.assertIn('hrv_std_rr', hrv_features)
        
        # RMSSD should be positive
        self.assertGreater(hrv_features['hrv_rmssd'], 0)
        
        print("✅ HRV calculation test passed")

class TestCameraProcessor(unittest.TestCase):
    
    def test_face_landmark_detection(self):
        """Test face landmark detection (simplified)"""
        # This would normally test actual face detection
        # For unit testing, we'll test the feature calculation logic
        
        analyzer = FaceMicroExpressionAnalyzer()
        
        # Create dummy landmark data
        dummy_landmarks = np.random.rand(478, 3)  # MediaPipe has 478 landmarks
        
        # Test feature calculations
        eyebrow_tension = analyzer.calculate_facial_displacement(
            dummy_landmarks, analyzer.eyebrow_indices
        )
        eye_aperture = analyzer.calculate_eye_aperture(dummy_landmarks)
        mouth_tension = analyzer.calculate_mouth_tension(dummy_landmarks)
        symmetry = analyzer.calculate_facial_symmetry(dummy_landmarks)
        
        # Check that features are calculated
        self.assertIsInstance(eyebrow_tension, float)
        self.assertIsInstance(eye_aperture, float)
        self.assertIsInstance(mouth_tension, float)
        self.assertIsInstance(symmetry, float)
        
        # Features should be in reasonable ranges
        self.assertGreaterEqual(eyebrow_tension, 0)
        self.assertGreaterEqual(eye_aperture, 0)
        self.assertGreaterEqual(mouth_tension, 0)
        self.assertGreaterEqual(symmetry, 0)
        self.assertLessEqual(symmetry, 1)
        
        print("✅ Face feature calculation test passed")

if __name__ == '__main__':
    unittest.main(verbosity=2)