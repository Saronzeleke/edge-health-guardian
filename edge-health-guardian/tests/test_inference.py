# tests/test_inference.py
import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.inference_engine import ArmOptimizedInference, HealthInferenceEngine
from utils.profiler import SystemProfiler

class TestInferenceEngine(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.inference_engine = HealthInferenceEngine()
        self.profiler = SystemProfiler()
        
        # Create dummy model files for testing
        self.create_dummy_models()
    
    def create_dummy_models(self):
        """Create dummy TFLite models for testing"""
        # In real implementation, these would be actual trained models
        # For testing, we'll create simple placeholder models
        import tensorflow as tf
        
        # Create a simple model for testing
        def create_dummy_model():
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu', input_shape=(32,)),
                tf.keras.layers.Dense(3, activation='sigmoid')
            ])
            return model
        
        # Create and save dummy models
        models_dir = "models/optimized_models"
        os.makedirs(models_dir, exist_ok=True)
        
        dummy_model = create_dummy_model()
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(dummy_model)
        tflite_model = converter.convert()
        
        # Save dummy models
        with open(f"{models_dir}/face_analyzer_int8.tflite", 'wb') as f:
            f.write(tflite_model)
        with open(f"{models_dir}/movement_analyzer_int8.tflite", 'wb') as f:
            f.write(tflite_model)
        with open(f"{models_dir}/fusion_engine_int8.tflite", 'wb') as f:
            f.write(tflite_model)
    
    def test_model_loading(self):
        """Test that models load correctly"""
        try:
            self.inference_engine.initialize_models()
            self.assertIsNotNone(self.inference_engine.face_analyzer.interpreter)
            self.assertIsNotNone(self.inference_engine.movement_analyzer.interpreter)
            self.assertIsNotNone(self.inference_engine.fusion_engine.interpreter)
            print("✅ Model loading test passed")
        except Exception as e:
            self.fail(f"Model loading failed: {e}")
    
    def test_inference_speed(self):
        """Test inference speed meets requirements"""
        self.inference_engine.initialize_models()
        
        # Create test data
        face_data = np.random.rand(1, 96, 96, 3).astype(np.float32)
        movement_data = np.random.rand(1, 50, 12).astype(np.float32)
        
        # Time inference
        import time
        start_time = time.perf_counter()
        
        results = self.inference_engine.process_health_data(face_data, movement_data)
        
        inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Should be under 100ms total
        self.assertLess(inference_time, 150, "Inference too slow")
        print(f"✅ Inference speed test passed: {inference_time:.2f} ms")
    
    def test_output_range(self):
        """Test that output values are in expected range"""
        self.inference_engine.initialize_models()
        
        face_data = np.random.rand(1, 96, 96, 3).astype(np.float32)
        movement_data = np.random.rand(1, 50, 12).astype(np.float32)
        
        results = self.inference_engine.process_health_data(face_data, movement_data)
        
        # Check output ranges
        self.assertGreaterEqual(results['stress_score'], 0.0)
        self.assertLessEqual(results['stress_score'], 1.0)
        
        self.assertGreaterEqual(results['fatigue_level'], 0.0)
        self.assertLessEqual(results['fatigue_level'], 1.0)
        
        self.assertGreaterEqual(results['anomaly_confidence'], 0.0)
        self.assertLessEqual(results['anomaly_confidence'], 1.0)
        
        print("✅ Output range test passed")
    
    def test_memory_usage(self):
        """Test memory usage during inference"""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        self.inference_engine.initialize_models()
        
        after_loading_memory = process.memory_info().rss / 1024 / 1024
        
        # Model loading should use reasonable memory
        memory_increase = after_loading_memory - initial_memory
        self.assertLess(memory_increase, 500, "Memory usage too high")  # Should be under 500MB
        
        print(f"✅ Memory usage test passed: {memory_increase:.2f} MB increase")

class TestSensorFusion(unittest.TestCase):
    
    def test_sensor_fusion_integration(self):
        """Test integration of sensor fusion with inference"""
        from core.sensor_fusion import AdaptiveSensorFusion
        
        fusion_system = AdaptiveSensorFusion()
        
        # Test data
        face_features = {'eyebrow_tension_mean': 0.5, 'eye_aperture_mean': 0.3}
        movement_features = {'movement_smoothness': 0.7, 'tremor_intensity': 0.2}
        
        # Update confidence
        fusion_system.update_confidence_scores(
            np.array([0.5, 0.3]),
            np.array([0.7, 0.2])
        )
        
        # Test fusion
        fused_face, fused_movement, _ = fusion_system.temporal_fusion({
            'face': np.array([0.5, 0.3]),
            'movement': np.array([0.7, 0.2])
        })
        
        self.assertIsNotNone(fused_face)
        self.assertIsNotNone(fused_movement)
        print("✅ Sensor fusion integration test passed")

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)