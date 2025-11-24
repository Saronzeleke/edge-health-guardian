# models/conversion/quantize_models.py
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import os
from pathlib import Path

class ModelQuantizer:
    def __init__(self, model_dir="models/trained_models"):
        self.model_dir = Path(model_dir)
        self.output_dir = Path("models/optimized_models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_representative_dataset(self, input_shape, num_samples=100):
        """Create representative dataset for quantization calibration"""
        def _representative_dataset():
            for _ in range(num_samples):
                if len(input_shape) == 4:  # Image data
                    data = np.random.randint(0, 255, size=(1,) + input_shape).astype(np.float32)
                else:  # Feature data
                    data = np.random.randn(1, *input_shape).astype(np.float32)
                yield [data]
        return _representative_dataset
    
    def quantize_face_model(self):
        """Quantize face analysis model to INT8"""
        print("🔧 Quantizing face analysis model...")
        
        # Load trained model
        model_path = self.model_dir / "face_analyzer.h5"
        if not model_path.exists():
            raise FileNotFoundError(f"Face model not found: {model_path}")
        
        # Load and prepare for quantization
        model = tf.keras.models.load_model(model_path)
        
        # Apply quantization aware training
        quantize_model = tfmot.quantization.keras.quantize_model
        q_aware_model = quantize_model(model)
        
        # Recompile quantized model
        q_aware_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Convert to TFLite with INT8 quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set representative dataset for full INT8 quantization
        representative_dataset = self.create_representative_dataset((96, 96, 3))
        converter.representative_dataset = representative_dataset
        
        # Ensure full INT8 quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        # Enable XNNPACK
        converter.experimental_new_converter = True
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save quantized model
        output_path = self.output_dir / "face_analyzer_int8.tflite"
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ Face model quantized and saved: {output_path}")
        return output_path
    
    def quantize_movement_model(self):
        """Quantize movement analysis model to INT8"""
        print("🔧 Quantizing movement analysis model...")
        
        model_path = self.model_dir / "movement_analyzer.h5"
        if not model_path.exists():
            raise FileNotFoundError(f"Movement model not found: {model_path}")
        
        model = tf.keras.models.load_model(model_path)
        
        # Convert to TFLite with optimization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Representative dataset for movement features
        representative_dataset = self.create_representative_dataset((50, 12))  # 50 timesteps, 12 features
        converter.representative_dataset = representative_dataset
        
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        output_path = self.output_dir / "movement_analyzer_int8.tflite"
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ Movement model quantized and saved: {output_path}")
        return output_path
    
    def quantize_fusion_model(self):
        """Quantize sensor fusion model to INT8"""
        print("🔧 Quantizing sensor fusion model...")
        
        model_path = self.model_dir / "fusion_engine.h5"
        if not model_path.exists():
            raise FileNotFoundError(f"Fusion model not found: {model_path}")
        
        model = tf.keras.models.load_model(model_path)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        representative_dataset = self.create_representative_dataset((1, 64))  # Fused features
        converter.representative_dataset = representative_dataset
        
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        output_path = self.output_dir / "fusion_engine_int8.tflite"
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ Fusion model quantized and saved: {output_path}")
        return output_path
    
    def convert_to_fp16(self, model_path):
        """Convert model to FP16 for GPU acceleration"""
        print(f"🔧 Converting {model_path} to FP16...")
        
        model = tf.keras.models.load_model(model_path)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        output_path = self.output_dir / f"{model_path.stem}_fp16.tflite"
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ FP16 model saved: {output_path}")
        return output_path
    
    def benchmark_quantized_models(self):
        """Benchmark quantized models for performance comparison"""
        print("📊 Benchmarking quantized models...")
        
        benchmarks = {}
        
        for model_file in self.output_dir.glob("*.tflite"):
            print(f"⏱️  Benchmarking {model_file.name}...")
            
            # Load model and create interpreter
            interpreter = tf.lite.Interpreter(model_path=str(model_file))
            interpreter.allocate_tensors()
            
            # Get input details
            input_details = interpreter.get_input_details()
            input_shape = input_details[0]['shape']
            
            # Prepare input data
            if input_details[0]['dtype'] == np.uint8:
                input_data = np.random.randint(0, 255, size=input_shape).astype(np.uint8)
            else:
                input_data = np.random.randn(*input_shape).astype(np.float32)
            
            # Benchmark inference time
            import time
            times = []
            for _ in range(100):
                start_time = time.perf_counter()
                
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                
                inference_time = time.perf_counter() - start_time
                times.append(inference_time * 1000)  # Convert to ms
            
            avg_time = np.mean(times[10:])  # Skip first 10 for warmup
            std_time = np.std(times[10:])
            
            benchmarks[model_file.name] = {
                'average_time_ms': avg_time,
                'std_time_ms': std_time,
                'model_size_mb': model_file.stat().st_size / (1024 * 1024)
            }
            
            print(f"   ⏱️  Average inference: {avg_time:.2f} ± {std_time:.2f} ms")
            print(f"   📦 Model size: {benchmarks[model_file.name]['model_size_mb']:.2f} MB")
        
        return benchmarks

def main():
    quantizer = ModelQuantizer()
    
    try:
        # Quantize all models
        face_model = quantizer.quantize_face_model()
        movement_model = quantizer.quantize_movement_model()
        fusion_model = quantizer.quantize_fusion_model()
        
        # Benchmark models
        benchmarks = quantizer.benchmark_quantized_models()
        
        print("\n🎯 Quantization Complete!")
        print("=" * 50)
        for model_name, stats in benchmarks.items():
            print(f"{model_name}:")
            print(f"  └── Inference: {stats['average_time_ms']:.2f} ms")
            print(f"  └── Size: {stats['model_size_mb']:.2f} MB")
        
    except Exception as e:
        print(f"❌ Quantization failed: {e}")

if __name__ == "__main__":
    main()