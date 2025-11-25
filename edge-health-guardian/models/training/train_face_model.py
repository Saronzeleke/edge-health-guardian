#!/usr/bin/env python3
"""
train_face_model_v2.py

Improved StressDetectionModel training + QAT + TFLite export helper.
Drop into your repo (e.g. models/training/) and run.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, Generator, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_model_optimization as tfmot

# -------------------------
# Model
# -------------------------
class StressDetectionModel:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape
        self.model: tf.keras.Model = None

    def build_temporal_attention_model(self) -> tf.keras.Model:
        """Build model with temporal attention for micro-expressions"""
        inputs = tf.keras.Input(shape=self.input_shape)

        # Spatial feature extraction
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
        x = layers.Reshape((1, -1))(x)                         # shape: (batch, 1, features)
        x = layers.LSTM(64, return_sequences=True)(x)         # shape: (batch, 1, 64)

        # Attention mechanism over the time axis (here time dim = 1)
        attention = layers.Dense(1, activation='tanh')(x)     # (batch, 1, 1)
        attention = layers.Flatten()(attention)               # (batch, 1)
        attention = layers.Activation('softmax')(attention)   # (batch, 1)
        attention = layers.RepeatVector(64)(attention)        # (batch, 64, 1)
        attention = layers.Permute([2, 1])(attention)         # (batch, 1, 64)

        attended_features = layers.multiply([x, attention])   # (batch, 1, 64)
        attended_features = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(attended_features)  # (batch, 64)

        # Output layers
        x = layers.Dense(128, activation='relu')(attended_features)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        # Multi-output for different health indicators
        stress_output = layers.Dense(1, activation='sigmoid', name='stress')(x)
        fatigue_output = layers.Dense(1, activation='sigmoid', name='fatigue')(x)
        anomaly_output = layers.Dense(1, activation='sigmoid', name='anomaly')(x)

        self.model = models.Model(inputs=inputs, outputs=[stress_output, fatigue_output, anomaly_output])
        return self.model

    def compile_model(self, lr: float = 1e-3):
        """Compile model with explicit tf.keras.metrics"""
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
        """
        Wrap model with quantize_model (QAT). After wrapping, recompile.
        Use QAT only when you intend to continue training.
        """
        if self.model is None:
            raise ValueError("Build the model first (call build_temporal_attention_model).")

        # Use tfmot wrapper
        quantize_scope = tfmot.quantization.keras.quantize_scope
        quantize_model_fn = tfmot.quantization.keras.quantize_model

        # Wrap model
        q_model = quantize_model_fn(self.model)
        self.model = q_model

        # Recompile (metrics/losses same as before)
        self.compile_model()

    # Training helper
    
    def train(self,
              train_input,
              train_labels,
              val_input,
              val_labels,
              epochs: int = 10,
              batch_size: int = 32,
              callbacks_list = None,
              verbose: int = 1):
        """
        Train the model.

        train_input/train_labels may be:
          - numpy arrays (train_input shape (N,H,W,C) and train_labels dict of arrays)
          - or tf.data.Dataset producing (x, y) where x:array, y:dict
        """
        if self.model is None:
            raise ValueError("Build and compile the model before training.")

        # If numpy arrays provided:
        if isinstance(train_input, np.ndarray):
            history = self.model.fit(
                train_input,
                train_labels,
                validation_data=(val_input, val_labels),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks_list,
                verbose=verbose
            )
        else:
            # assume tf.data.Dataset
            history = self.model.fit(
                train_input,
                validation_data=(val_input),
                epochs=epochs,
                callbacks=callbacks_list,
                verbose=verbose
            )
        return history

  
    # TFLite export helper
   
    def save_as_tflite(self,
                       filepath: str,
                       representative_gen: Generator = None,
                       full_integer: bool = True):
        """
        Convert and save the (optionally quantized) Keras model to TFLite.

        - If full_integer=True and representative_gen provided, converter will target INT8.
        - representative_gen yields lists of numpy arrays matching input signatures: e.g. [input_tensor]
        """
        if self.model is None:
            raise ValueError("Model must be built/loaded before saving to TFLite.")

        # Use TFLiteConverter from Keras model (supports QAT-wrapped models)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if representative_gen is not None and full_integer:
            converter.representative_dataset = representative_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # Use int8 for maximum portability on accelerators; some stacks prefer uint8
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        else:
            # fallback: dynamic range quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ TFLite model written to: {filepath}")

# Representative generator helper

def make_representative_gen_from_numpy(x_samples: np.ndarray, num_steps: int = 100):
    """
    Creates a callable that yields representative samples for TFLite converter.
    x_samples: np.ndarray (N, H, W, C)
    """
    def rep_gen():
        count = 0
        for i in range(x_samples.shape[0]):
            yield [x_samples[i:i+1].astype(np.float32)]
            count += 1
            if count >= num_steps:
                break
    return rep_gen

# CLI demo (synthetic data)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--qat', action='store_true', help='Enable quantization-aware training (QAT)')
    p.add_argument('--export_tflite', action='store_true', help='Export TFLite model after training')
    p.add_argument('--out', type=str, default='models/trained_models', help='Output directory')
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building model...")
    sd = StressDetectionModel((args.img_size, args.img_size, 3))
    sd.build_temporal_attention_model()
    sd.compile_model()

    # Optional QAT wrapper (wrap BEFORE additional training epochs for QAT)
    if args.qat:
        print("Wrapping model for QAT (quantization-aware training)...")
        sd.quantize_model()

    # Create synthetic data for demo (replace with real data loader)
    N = 200
    X = np.random.rand(N, args.img_size, args.img_size, 3).astype(np.float32)
    y = {
        'stress': np.random.randint(0, 2, size=(N, 1)).astype(np.float32),
        'fatigue': np.random.randint(0, 2, size=(N, 1)).astype(np.float32),
        'anomaly': np.random.randint(0, 2, size=(N, 1)).astype(np.float32)
    }

    split = int(0.8 * N)
    X_train, X_val = X[:split], X[split:]
    y_train = {k: v[:split] for k, v in y.items()}
    y_val = {k: v[split:] for k, v in y.items()}

    # Callbacks
    ckpt = out_dir / 'stress_best.h5'
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt), monitor='val_loss', save_best_only=True)
    ]

    print("Starting training (synthetic demo)...")
    history = sd.train(X_train, y_train, X_val, y_val,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       callbacks_list=callbacks_list,
                       verbose=1)

    # Save final Keras model
    keras_path = out_dir / 'stress_detector_final.h5'
    sd.model.save(str(keras_path))
    print(f"✅ Saved Keras model to: {keras_path}")

    # TFLite export (if requested)
    if args.export_tflite:
        print("Exporting TFLite...")
        rep_gen = make_representative_gen_from_numpy(X_train, num_steps=200)
        tflite_path = out_dir / 'stress_detector_quant.tflite'
        sd.save_as_tflite(str(tflite_path), representative_gen=rep_gen, full_integer=True)

if __name__ == "__main__":
    main()
