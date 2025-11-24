# tests/integration_test.py
import unittest
import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import EdgeHealthGuardian
from core.inference_engine import HealthInferenceEngine
from core.sensor_fusion import AdaptiveSensorFusion
from utils.profiler import SystemProfiler

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.health_guardian = EdgeHealthGuardian()
        self.system_profiler = SystemProfiler()
        
    def test_end_to_end_processing(self):
        """Test complete end-to-end processing pipeline"""
        print("🚀 Testing end-to-end processing pipeline...")
        
        # Start system profiling
        self.system_profiler.start_profiling(duration=30)
        
        try:
            # Initialize system
            self.assertTrue(
                self.health_guardian.initialize_system(),
                "System initialization failed"
            )
            
            # Simulate sensor data processing
            test_results = []
            for i in range(10):  # Process 10 frames
                # Simulate face data (96x96x3 image)
                face_data = np.random.rand(1, 96, 96, 3).astype(np.float32) * 255
                
                # Simulate movement data (50 timesteps, 12 features)
                movement_data = np.random.rand(1, 50, 12).astype(np.float32)
                
                # Process through inference engine
                results = self.health_guardian.inference_engine.process_health_data(
                    face_data, movement_data
                )
                
                test_results.append(results)
                
                # Record inference time
                total_time = sum(results['inference_times'].values())
                self.system_profiler.record_inference_time(total_time)
                
                time.sleep(0.1)  # Simulate real-time processing
            
            # Verify results
            self.assertEqual(len(test_results), 10)
            
            for result in test_results:
                self.assertIn('stress_score', result)
                self.assertIn('fatigue_level', result)
                self.assertIn('anomaly_confidence', result)
                self.assertIn('inference_times', result)
                
                # Check value ranges
                self.assertGreaterEqual(result['stress_score'], 0.0)
                self.assertLessEqual(result['stress_score'], 1.0)
            
            print("✅ End-to-end processing test passed")
            
        except Exception as e:
            self.fail(f"End-to-end test failed: {e}")
        
        finally:
            # Stop profiling and print report
            profile_summary = self.system_profiler.stop_profiling()
            self.system_profiler.print_profile_report()
    
    def test_system_resources(self):
        """Test system resource usage under load"""
        print("💻 Testing system resource usage...")
        
        # Start profiling
        self.system_profiler.start_profiling(duration=10)
        
        # Simulate load
        inference_engine = HealthInferenceEngine()
        inference_engine.initialize_models()
        
        load_results = []
        for i in range(50):  # Heavy load test
            face_data = np.random.rand(1, 96, 96, 3).astype(np.float32)
            movement_data = np.random.rand(1, 50, 12).astype(np.float32)
            
            start_time = time.perf_counter()
            results = inference_engine.process_health_data(face_data, movement_data)
            inference_time = (time.perf_counter() - start_time) * 1000
            
            self.system_profiler.record_inference_time(inference_time)
            load_results.append(results)
            
            time.sleep(0.05)  # 20Hz processing
        
        # Check performance
        profile_summary = self.system_profiler.stop_profiling()
        
        # Average inference time should be under 100ms
        avg_inference_time = profile_summary['inference_times']['mean']
        self.assertLess(avg_inference_time, 100, 
                       f"Average inference time too high: {avg_inference_time:.2f}ms")
        
        # CPU usage should be reasonable
        avg_cpu = profile_summary['cpu_usage']['mean']
        self.assertLess(avg_cpu, 80, f"CPU usage too high: {avg_cpu:.1f}%")
        
        print(f"✅ System resource test passed - Avg inference: {avg_inference_time:.2f}ms, CPU: {avg_cpu:.1f}%")
    
    def test_alert_system_integration(self):
        """Test alert system integration"""
        print("🚨 Testing alert system integration...")
        
        from utils.alert_system import AlertSystem
        
        alert_system = AlertSystem()
        
        # Test different alert levels
        test_alerts = [
            {
                'type': 'high_stress',
                'level': 'warning',
                'message': 'Stress level above threshold'
            },
            {
                'type': 'health_anomaly', 
                'level': 'critical',
                'message': 'Potential health anomaly detected'
            }
        ]
        
        for alert_data in test_alerts:
            alert_system.trigger_alert(alert_data)
        
        # Check that alerts are active
        active_alerts = alert_system.get_active_alerts()
        self.assertEqual(len(active_alerts), 2)
        
        # Acknowledge one alert
        if active_alerts:
            alert_system.acknowledge_alert(active_alerts[0]['id'])
            remaining_alerts = alert_system.get_active_alerts()
            self.assertEqual(len(remaining_alerts), 1)
        
        print("✅ Alert system integration test passed")

    def test_sensor_fusion_consistency(self):
        """Test sensor fusion consistency across multiple inputs"""
        print("🔄 Testing sensor fusion consistency...")
        
        fusion_system = AdaptiveSensorFusion()
        
        # Test with consistent data
        consistent_face_data = np.array([0.5, 0.3, 0.2])
        consistent_movement_data = np.array([0.6, 0.4, 0.1])
        
        for i in range(20):
            fusion_system.update_confidence_scores(
                consistent_face_data, consistent_movement_data
            )
            
            fused_face, fused_movement, _ = fusion_system.temporal_fusion({
                'face': consistent_face_data,
                'movement': consistent_movement_data
            })
            
            # Check that fused data is consistent
            self.assertIsNotNone(fused_face)
            self.assertIsNotNone(fused_movement)
        
        # Get final confidence weights
        face_weight, movement_weight, hr_weight = fusion_system.get_fusion_weights()
        
        # Weights should be reasonable
        self.assertGreater(face_weight, 0.1)
        self.assertGreater(movement_weight, 0.1)
        self.assertLessEqual(hr_weight, 1.0)
        
        print("✅ Sensor fusion consistency test passed")

def run_all_integration_tests():
    """Run all integration tests and provide summary"""
    print("🧪 RUNNING INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("=" * 60)
    print("📊 INTEGRATION TEST SUMMARY")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("🎉 ALL INTEGRATION TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
        for test, traceback in result.failures + result.errors:
            print(f"\nFailed: {test}")
            print(f"Error: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    run_all_integration_tests()