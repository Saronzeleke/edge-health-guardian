# models/training/stress_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_model_optimization as tfmot
from typing import Generator
from pathlib import Path

class StressDetectionModel:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape
        self.model: tf.keras.Model = None

    def build_temporal_attention_model(self) -> tf.keras.Model:
        """Build CNN + LSTM + temporal attention for stress, fatigue, anomaly detection"""
        inputs = tf.keras.Input(shape=self.input_shape)

        # Spatial feature extraction (CNN)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)

        # Temporal attention for sequence processing (1-step sequence)
        x = layers.Reshape((1, -1))(x)
        x = layers.LSTM(64, return_sequences=True)(x)

        # Attention mechanism over the time axis
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(64)(attention)
        attention = layers.Permute([2, 1])(attention)

        attended_features = layers.multiply([x, attention])
        attended_features = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(attended_features)

        # Fully connected layers
        x = layers.Dense(128, activation='relu')(attended_features)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        # Multi-output for stress, fatigue, anomaly
        stress_output = layers.Dense(1, activation='sigmoid', name='stress')(x)
        fatigue_output = layers.Dense(1, activation='sigmoid', name='fatigue')(x)
        anomaly_output = layers.Dense(1, activation='sigmoid', name='anomaly')(x)

        self.model = models.Model(inputs=inputs, outputs=[stress_output, fatigue_output, anomaly_output])
        return self.model

    def compile_model(self, lr: float = 1e-3):
        """Compile model with losses and metrics"""
        if self.model is None:
            raise ValueError("Build the model first (call build_temporal_attention_model).")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss={
                'stress': 'binary_crossentropy',
                'fatigue': 'binary_crossentropy',
                'anomaly': 'binary_crossentropy'
            },
            metrics={
                'stress': [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')],
                'fatigue': [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall')],
                'anomaly': [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall')]
            }
        )

    def quantize_model(self):
        """Apply quantization-aware training (QAT)"""
        if self.model is None:
            raise ValueError("Build the model first (call build_temporal_attention_model).")
        quantize_model_fn = tfmot.quantization.keras.quantize_model
        self.model = quantize_model_fn(self.model)
        self.compile_model()

    def train(self,
              train_input,
              train_labels,
              val_input,
              val_labels,
              epochs: int = 10,
              batch_size: int = 32,
              callbacks_list = None,
              verbose: int = 1):
        """Train the model"""
        if self.model is None:
            raise ValueError("Build and compile the model before training.")
        history = self.model.fit(
            train_input,
            train_labels,
            validation_data=(val_input, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose
        )
        return history

    def save_as_tflite(self,
                       filepath: str,
                       representative_gen: Generator = None,
                       full_integer: bool = True):
        """Convert model to TFLite (optionally INT8 quantized)"""
        if self.model is None:
            raise ValueError("Model must be built/loaded before saving to TFLite.")

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if representative_gen is not None and full_integer:
            converter.representative_dataset = representative_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        tf.io.gfile.makedirs(str(Path(filepath).parent))
        with open(filepath, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ TFLite model saved to: {filepath}")

# Representative generator helper
def make_representative_gen_from_numpy(x_samples: np.ndarray, num_steps: int = 100):
    """Yield representative samples for TFLite INT8 quantization"""
    def rep_gen():
        count = 0
        for i in range(x_samples.shape[0]):
            yield [x_samples[i:i+1].astype(np.float32)]
            count += 1
            if count >= num_steps:
                break
    return rep_gen