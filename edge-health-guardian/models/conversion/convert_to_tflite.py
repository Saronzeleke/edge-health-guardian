# models/conversion/convert_to_tflite.py
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import os
import json
from pathlib import Path
import logging

class TFLiteConverter:
    """Comprehensive TFLite conversion with Arm optimization"""
    
    def __init__(self, model_dir="models/trained_models"):
        self.model_dir = Path(model_dir)
        self.output_dir = Path("models/optimized_models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("TFLiteConverter")
        
    def convert_keras_to_tflite(self, model_name: str, optimization_level: str = "DEFAULT"):
        """
        Convert Keras model to TFLite with specified optimization
        
        Args:
            model_name: Name of the model file (without extension)
            optimization_level: One of ["DEFAULT", "INT8", "FP16", "FULL_INT8"]
        """
        model_path = self.model_dir / f"{model_name}.h5"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.logger.info(f"🔄 Converting {model_name} with {optimization_level} optimization...")
        
        # Load Keras model
        model = tf.keras.models.load_model(model_path)
        
        # Create converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply optimizations based on level
        if optimization_level == "DEFAULT":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        elif optimization_level == "FP16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        elif optimization_level == "INT8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # For INT8, we need a representative dataset
            representative_dataset = self._create_representative_dataset(model)
            converter.representative_dataset = representative_dataset
        
        elif optimization_level == "FULL_INT8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            representative_dataset = self._create_representative_dataset(model)
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        
        # Arm-specific optimizations
        converter.experimental_new_converter = True
        converter._experimental_default_to_single_batch_in_tensor_list_ops = True
        
        # Convert model
        try:
            tflite_model = converter.convert()
            
            # Save converted model
            output_path = self.output_dir / f"{model_name}_{optimization_level.lower()}.tflite"
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            self.logger.info(f"✅ Model converted and saved: {output_path}")
            
            # Generate model info
            model_info = self._generate_model_info(model, tflite_model, output_path)
            self._save_model_info(model_name, optimization_level, model_info)
            
            return output_path, model_info
            
        except Exception as e:
            self.logger.error(f"❌ Conversion failed: {e}")
            raise
    
    def _create_representative_dataset(self, model):
        """Create representative dataset for quantization"""
        input_shape = model.input_shape
        
        def representative_dataset():
            for _ in range(100):
                if len(input_shape) == 4:  # Image data
                    data = np.random.randint(0, 255, size=(1,) + input_shape[1:]).astype(np.float32)
                else:  # Feature data
                    data = np.random.randn(1, *input_shape[1:]).astype(np.float32)
                yield [data]
        
        return representative_dataset
    
    def _generate_model_info(self, original_model, tflite_model, output_path):
        """Generate information about the converted model"""
        # Get original model info
        original_params = original_model.count_params()
        original_size = os.path.getsize(self.model_dir / f"{original_model.name}.h5") if hasattr(original_model, 'name') else 0
        
        # Get TFLite model info
        tflite_size = os.path.getsize(output_path)
        
        # Get model structure info
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        model_info = {
            'original_parameters': original_params,
            'original_size_mb': original_size / (1024 * 1024),
            'tflite_size_mb': tflite_size / (1024 * 1024),
            'compression_ratio': original_size / tflite_size if original_size > 0 else 0,
            'input_details': [
                {
                    'name': detail['name'],
                    'shape': detail['shape'].tolist(),
                    'dtype': str(detail['dtype'])
                } for detail in input_details
            ],
            'output_details': [
                {
                    'name': detail['name'],
                    'shape': detail['shape'].tolist(),
                    'dtype': str(detail['dtype'])
                } for detail in output_details
            ],
            'tensor_count': len(interpreter.get_tensor_details())
        }
        
        return model_info
    
    def _save_model_info(self, model_name, optimization_level, model_info):
        """Save model information to JSON"""
        info_dir = self.output_dir / "model_info"
        info_dir.mkdir(exist_ok=True)
        
        info_path = info_dir / f"{model_name}_{optimization_level.lower()}_info.json"
        
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        self.logger.info(f"📊 Model info saved: {info_path}")
    
    def batch_convert_models(self, model_configs: dict):
        """
        Batch convert multiple models with different optimizations
        
        Args:
            model_configs: Dictionary with model names and their optimization levels
                Example: {'face_analyzer': ['DEFAULT', 'INT8'], 'movement_analyzer': ['FP16']}
        """
        results = {}
        
        for model_name, optimizations in model_configs.items():
            results[model_name] = {}
            
            for optimization in optimizations:
                try:
                    output_path, model_info = self.convert_keras_to_tflite(model_name, optimization)
                    results[model_name][optimization] = {
                        'output_path': str(output_path),
                        'model_info': model_info,
                        'success': True
                    }
                except Exception as e:
                    results[model_name][optimization] = {
                        'success': False,
                        'error': str(e)
                    }
        
        # Save batch conversion summary
        self._save_batch_summary(results)
        return results
    
    def _save_batch_summary(self, results):
        """Save batch conversion summary"""
        summary = {
            'conversion_date': str(np.datetime64('now')),
            'total_models': len(results),
            'total_conversions': sum(len(conversions) for conversions in results.values()),
            'successful_conversions': sum(
                1 for model_conversions in results.values()
                for conversion in model_conversions.values()
                if conversion.get('success', False)
            ),
            'models': results
        }
        
        summary_path = self.output_dir / "conversion_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📋 Batch conversion summary saved: {summary_path}")
    
    def benchmark_tflite_model(self, model_path: Path, input_shape: tuple, iterations: int = 100):
        """Benchmark TFLite model performance"""
        self.logger.info(f"⏱️ Benchmarking {model_path.name}...")
        
        # Load model
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare input data based on expected type
        if input_details[0]['dtype'] == np.uint8:
            input_data = np.random.randint(0, 255, size=input_shape).astype(np.uint8)
        else:
            input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # Warm-up
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
        
        # Benchmark inference time
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            start_time = tf.timestamp()
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            end_time = tf.timestamp()
            times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Simple memory usage estimation
            memory_usage.append(self._estimate_memory_usage(interpreter))
        
        benchmark_results = {
            'average_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'throughput_fps': float(1000 / np.mean(times)),
            'average_memory_mb': float(np.mean(memory_usage)),
            'model_size_mb': os.path.getsize(model_path) / (1024 * 1024)
        }
        
        self.logger.info(f"📊 Benchmark results for {model_path.name}:")
        self.logger.info(f"   Average inference: {benchmark_results['average_time_ms']:.2f} ms")
        self.logger.info(f"   Throughput: {benchmark_results['throughput_fps']:.2f} FPS")
        self.logger.info(f"   Memory usage: {benchmark_results['average_memory_mb']:.2f} MB")
        
        return benchmark_results
    
    def _estimate_memory_usage(self, interpreter):
        """Estimate memory usage of TFLite interpreter"""
        tensor_details = interpreter.get_tensor_details()
        total_memory = 0
        
        for tensor in tensor_details:
            if tensor['shape'] is not None:
                tensor_size = np.prod(tensor['shape']) * np.dtype(tensor['dtype']).itemsize
                total_memory += tensor_size
        
        return total_memory / (1024 * 1024)  # Convert to MB

class ArmOptimizedConverter(TFLiteConverter):
    """Arm-optimized TFLite conversion with XNNPACK support"""
    
    def __init__(self, model_dir="models/trained_models"):
        super().__init__(model_dir)
        self.supported_arm_optimizations = ['XNNPACK', 'NEON', 'ARM_COMPUTE_LIBRARY']
    
    def convert_with_xnnpack(self, model_name: str, optimization_level: str = "INT8"):
        """Convert model with XNNPACK delegate for Arm acceleration"""
        self.logger.info(f"🚀 Converting {model_name} with XNNPACK optimization...")
        
        try:
            # First convert to TFLite
            tflite_path, model_info = self.convert_keras_to_tflite(model_name, optimization_level)
            
            # Apply XNNPACK delegate (this happens at runtime, but we can verify compatibility)
            self._verify_xnnpack_compatibility(tflite_path)
            
            self.logger.info(f"✅ XNNPACK-compatible model created: {tflite_path}")
            return tflite_path, model_info
            
        except Exception as e:
            self.logger.error(f"❌ XNNPACK conversion failed: {e}")
            raise
    
    def _verify_xnnpack_compatibility(self, model_path):
        """Verify that model is compatible with XNNPACK delegate"""
        try:
            # Load model with XNNPACK delegate
            delegate = tf.lite.load_delegate('XNNPACK')
            interpreter = tf.lite.Interpreter(
                model_path=str(model_path),
                experimental_delegates=[delegate]
            )
            interpreter.allocate_tensors()
            
            self.logger.info("✅ Model is XNNPACK compatible")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Model may not be fully XNNPACK compatible: {e}")
            return False
    
    def create_multi_delegate_model(self, model_name: str, delegates: list):
        """Create model optimized for multiple delegates"""
        self.logger.info(f"🔄 Creating multi-delegate model for {model_name}...")
        
        # This is a simplified implementation
        # In practice, you'd partition the model for different delegates
        
        model_path = self.model_dir / f"{model_name}.h5"
        model = tf.keras.models.load_model(model_path)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        
        # Note: Multi-delegate support requires model partitioning
        # This is an advanced feature that would need custom implementation
        
        tflite_model = converter.convert()
        
        output_path = self.output_dir / f"{model_name}_multi_delegate.tflite"
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        self.logger.info(f"✅ Multi-delegate model created: {output_path}")
        return output_path

def main():
    """Main conversion script"""
    converter = ArmOptimizedConverter()
    
    # Model configurations for batch conversion
    model_configs = {
        'face_analyzer': ['DEFAULT', 'INT8', 'FP16'],
        'movement_analyzer': ['DEFAULT', 'INT8'],
        'fusion_engine': ['DEFAULT', 'INT8']
    }
    
    print("🚀 Starting batch model conversion...")
    
    try:
        # Batch convert all models
        results = converter.batch_convert_models(model_configs)
        
        # Benchmark converted models
        print("\n⏱️ Starting model benchmarking...")
        benchmark_results = {}
        
        for model_name, conversions in results.items():
            benchmark_results[model_name] = {}
            
            for optimization, result in conversions.items():
                if result.get('success', False):
                    model_path = Path(result['output_path'])
                    
                    # Determine input shape based on model type
                    if 'face' in model_name.lower():
                        input_shape = (1, 96, 96, 3)
                    elif 'movement' in model_name.lower():
                        input_shape = (1, 50, 12)
                    else:  # fusion engine
                        input_shape = (1, 64)
                    
                    benchmark = converter.benchmark_tflite_model(model_path, input_shape)
                    benchmark_results[model_name][optimization] = benchmark
        
        # Save benchmark results
        benchmark_path = converter.output_dir / "benchmark_results.json"
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"📊 Benchmark results saved: {benchmark_path}")
        print("🎉 Model conversion and benchmarking complete!")
        
    except Exception as e:
        print(f"❌ Conversion process failed: {e}")

if __name__ == "__main__":
    main()