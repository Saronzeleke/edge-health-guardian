# src/core/model_manager.py
import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

class ModelManager:
    """Manager for loading and managing TFLite models"""
    
    def __init__(self, models_dir: str = "models/optimized_models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.model_info = {}
        
        self.logger = logging.getLogger("ModelManager")
        
    def load_model(self, model_name: str, model_path: Optional[str] = None):
        """Load a TFLite model with optimization"""
        if model_path is None:
            model_path = self.models_dir / f"{model_name}.tflite"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.logger.info(f"📦 Loading model: {model_name}")
        
        try:
            # Try to load with XNNPACK delegate for Arm optimization
            try:
                delegate = tf.lite.load_delegate('XNNPACK')
                interpreter = tf.lite.Interpreter(
                    model_path=str(model_path),
                    experimental_delegates=[delegate]
                )
                self.logger.info("✅ Model loaded with XNNPACK delegate")
            except Exception as e:
                self.logger.warning(f"XNNPACK not available, using CPU: {e}")
                interpreter = tf.lite.Interpreter(model_path=str(model_path))
            
            interpreter.allocate_tensors()
            
            # Store model and its details
            self.models[model_name] = {
                'interpreter': interpreter,
                'input_details': interpreter.get_input_details(),
                'output_details': interpreter.get_output_details(),
                'model_path': model_path
            }
            
            # Load model info if available
            info_path = self.models_dir / "model_info" / f"{model_name}_info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    self.model_info[model_name] = json.load(f)
            
            self.logger.info(f"✅ Model {model_name} loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model {model_name}: {e}")
            raise
    
    def unload_model(self, model_name: str):
        """Unload a model from memory"""
        if model_name in self.models:
            del self.models[model_name]
            self.logger.info(f"✅ Model {model_name} unloaded")
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a loaded model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model_data = self.models[model_name]
        interpreter = model_data['interpreter']
        
        info = {
            'model_name': model_name,
            'input_details': model_data['input_details'],
            'output_details': model_data['output_details'],
            'tensor_count': len(interpreter.get_tensor_details()),
            'model_size_mb': model_data['model_path'].stat().st_size / (1024 * 1024)
        }
        
        # Add saved model info if available
        if model_name in self.model_info:
            info.update(self.model_info[model_name])
        
        return info
    
    def run_inference(self, model_name: str, input_data: np.ndarray) -> np.ndarray:
        """Run inference on a loaded model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model_data = self.models[model_name]
        interpreter = model_data['interpreter']
        input_details = model_data['input_details']
        output_details = model_data['output_details']
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        return output_data
    
    def benchmark_model(self, model_name: str, input_shape: tuple, iterations: int = 100) -> Dict:
        """Benchmark model performance"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        # Generate test data
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # Warm-up
        for _ in range(10):
            self.run_inference(model_name, input_data)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start_time = tf.timestamp()
            self.run_inference(model_name, input_data)
            end_time = tf.timestamp()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'average_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'throughput_fps': float(1000 / np.mean(times))
        }
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names"""
        return list(self.models.keys())
    
    def preload_all_models(self):
        """Preload all models in the models directory"""
        model_files = list(self.models_dir.glob("*.tflite"))
        
        for model_file in model_files:
            model_name = model_file.stem
            try:
                self.load_model(model_name, model_file)
            except Exception as e:
                self.logger.error(f"Failed to load {model_name}: {e}")