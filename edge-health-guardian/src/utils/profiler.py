# src/utils/profiler.py
import time
import psutil
import numpy as np
from typing import Dict, List
import threading

class SystemProfiler:
    """System performance profiler for resource monitoring"""
    
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.inference_times = []
        self.sampling_interval = 1.0  # seconds
        self.is_profiling = False
        self.profile_thread = None
        
    def start_profiling(self, duration: int = 300):
        """Start system profiling"""
        self.is_profiling = True
        self.profile_thread = threading.Thread(target=self._profile_loop, args=(duration,))
        self.profile_thread.daemon = True
        self.profile_thread.start()
        print(f"📊 Started system profiling for {duration} seconds")
    
    def _profile_loop(self, duration: int):
        """Main profiling loop"""
        start_time = time.time()
        
        while self.is_profiling and (time.time() - start_time) < duration:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_usage.append(cpu_percent)
            
            # Memory usage
            memory_info = psutil.virtual_memory()
            self.memory_usage.append(memory_info.percent)
            
            time.sleep(self.sampling_interval)
    
    def record_inference_time(self, inference_time: float):
        """Record inference time"""
        self.inference_times.append(inference_time)
    
    def stop_profiling(self) -> Dict:
        """Stop profiling and return results"""
        self.is_profiling = False
        if self.profile_thread:
            self.profile_thread.join()
        
        return self.get_profile_summary()
    
    def get_profile_summary(self) -> Dict:
        """Get profiling summary"""
        if not self.cpu_usage:
            return {}
        
        summary = {
            'cpu_usage': {
                'mean': np.mean(self.cpu_usage),
                'std': np.std(self.cpu_usage),
                'max': np.max(self.cpu_usage),
                'min': np.min(self.cpu_usage)
            },
            'memory_usage': {
                'mean': np.mean(self.memory_usage),
                'std': np.std(self.memory_usage),
                'max': np.max(self.memory_usage),
                'min': np.min(self.memory_usage)
            },
            'inference_times': {
                'mean': np.mean(self.inference_times) if self.inference_times else 0,
                'std': np.std(self.inference_times) if self.inference_times else 0,
                'max': np.max(self.inference_times) if self.inference_times else 0,
                'min': np.min(self.inference_times) if self.inference_times else 0
            },
            'samples_collected': len(self.cpu_usage)
        }
        
        return summary
    
    def print_profile_report(self):
        """Print detailed profiling report"""
        summary = self.get_profile_summary()
        
        if not summary:
            print("No profiling data available")
            return
        
        print("\n" + "="*50)
        print("📊 SYSTEM PROFILING REPORT")
        print("="*50)
        
        print(f"CPU Usage:")
        print(f"  Average: {summary['cpu_usage']['mean']:.1f}%")
        print(f"  Range: {summary['cpu_usage']['min']:.1f}% - {summary['cpu_usage']['max']:.1f}%")
        
        print(f"Memory Usage:")
        print(f"  Average: {summary['memory_usage']['mean']:.1f}%")
        print(f"  Range: {summary['memory_usage']['min']:.1f}% - {summary['memory_usage']['max']:.1f}%")
        
        if summary['inference_times']['mean'] > 0:
            print(f"Inference Times:")
            print(f"  Average: {summary['inference_times']['mean']:.2f} ms")
            print(f"  Range: {summary['inference_times']['min']:.2f} - {summary['inference_times']['max']:.2f} ms")
        
        print(f"Samples Collected: {summary['samples_collected']}")
        print("="*50)

class ModelProfiler:
    """Model-specific performance profiler"""
    
    @staticmethod
    def profile_model_memory(model_path: str) -> Dict:
        """Profile model memory usage"""
        import os
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        return {
            'model_size_mb': model_size,
            'estimated_ram_mb': model_size * 2,  # Rough estimate
            'tensor_arena_size': model_size * 3   # TFLite tensor arena
        }
    
    @staticmethod
    def benchmark_model(model_path: str, input_shape: tuple, iterations: int = 100) -> Dict:
        """Benchmark model inference performance"""
        import tensorflow as tf
        
        # Load model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare input data
        input_data = np.random.random(input_shape).astype(np.float32)
        
        # Warm-up
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'average_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'throughput_fps': 1000 / np.mean(times)
        }