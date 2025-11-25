import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import tensorflow_model_optimization as tfmot

class StressDetectionModel:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape
        self.model = None
        
    def build_temporal_attention_model(self):
        """Build model with temporal attention for micro-expressions"""
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Spatial feature extraction
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Temporal attention for sequence processing
        x = layers.Reshape((1, -1))(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(64)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        attended_features = layers.multiply([x, attention])
        attended_features = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended_features)
        
        # Output layers
        x = layers.Dense(128, activation='relu')(attended_features)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Multi-output for different health indicators
        stress_output = layers.Dense(1, activation='sigmoid', name='stress')(x)
        fatigue_output = layers.Dense(1, activation='sigmoid', name='fatigue')(x)
        anomaly_output = layers.Dense(1, activation='sigmoid', name='anomaly')(x)
        
        self.model = models.Model(
            inputs=inputs, 
            outputs=[stress_output, fatigue_output, anomaly_output]
        )
        
        return self.model
    
    def compile_model(self):
        """Compile model with appropriate losses and metrics"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'stress': 'binary_crossentropy',
                'fatigue': 'binary_crossentropy', 
                'anomaly': 'binary_crossentropy'
            },
            metrics={
                'stress': ['accuracy', 'precision', 'recall'],
                'fatigue': ['accuracy', 'precision', 'recall'],
                'anomaly': ['accuracy', 'precision', 'recall']
            }
        )
    
    def quantize_model(self):
        """Apply quantization-aware training for Arm optimization"""
        # Apply quantization aware training
        quantize_model = tfmot.quantization.keras.quantize_model
        self.model = quantize_model(self.model)
        
        # Recompile after quantization
        self.compile_model()

# models/conversion/convert_to_tflite.py
import tensorflow as tf
import numpy as np

class TFLiteConverter:
    def __init__(self):
        self.converter = None
        
    def convert_to_int8(self, model_path: str, representative_dataset):
        """Convert model to INT8 TFLite with optimization"""
        self.converter = tf.lite.TFLiteConverter.from_keras_model(model_path)
        
        # Arm optimization settings
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.converter.representative_dataset = representative_dataset
        self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        self.converter.inference_input_type = tf.int8
        self.converter.inference_output_type = tf.int8
        
        # Enable XNNPACK delegation
        self.converter.experimental_new_converter = True
        self.converter._experimental_default_to_single_batch_in_tensor_list_ops = True
        
        tflite_model = self.converter.convert()
        
        return tflite_model
    
    def convert_to_fp16(self, model_path: str):
        """Convert model to FP16 for GPU acceleration"""
        self.converter = tf.lite.TFLiteConverter.from_keras_model(model_path)
        
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = self.converter.convert()
        
        return tflite_model

def create_representative_dataset():
    """Create representative dataset for quantization calibration"""
    def _representative_dataset():
        for _ in range(100):
            # Generate random data in the expected input range
            data = np.random.rand(1, 96, 96, 3).astype(np.float32) * 255.0
            yield [data]
    return _representative_dataset