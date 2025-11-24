# models/training/train_movement_model.py
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import logging

class MovementModelTrainer:
    def __init__(self, sequence_length=50, feature_dim=12):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = None
        self.scaler = StandardScaler()
        
    def build_lstm_model(self):
        """Build LSTM model for movement pattern analysis"""
        inputs = tf.keras.Input(shape=(self.sequence_length, self.feature_dim))
        
        # Bidirectional LSTM for temporal patterns
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        
        # Apply attention
        attended = layers.Dot(axes=1)([x, attention])
        
        # Output layers
        x = layers.Dense(128, activation='relu')(attended)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # Multi-output for different movement patterns
        stress_output = layers.Dense(1, activation='sigmoid', name='movement_stress')(x)
        fatigue_output = layers.Dense(1, activation='sigmoid', name='movement_fatigue')(x)
        anomaly_output = layers.Dense(1, activation='sigmoid', name='movement_anomaly')(x)
        
        self.model = models.Model(
            inputs=inputs, 
            outputs=[stress_output, fatigue_output, anomaly_output]
        )
        
        return self.model
    
    def prepare_sequence_data(self, data, labels, sequence_length=50):
        """Convert time series data to sequences"""
        sequences = []
        sequence_labels = []
        
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:(i + sequence_length)])
            sequence_labels.append(labels[i + sequence_length])
        
        return np.array(sequences), np.array(sequence_labels)
    
    def load_synthetic_movement_data(self):
        """Generate synthetic movement data for training"""
        print("📊 Generating synthetic movement data...")
        
        num_samples = 10000
        time_steps = 1000
        
        # Generate realistic movement patterns
        data = []
        labels = []
        
        for i in range(num_samples):
            # Base movement patterns
            t = np.linspace(0, 10, time_steps)
            
            # Normal movement (smooth patterns)
            if i < num_samples * 0.7:  # 70% normal
                # Smooth accelerometer data
                accel_x = np.sin(t) + 0.1 * np.random.normal(size=time_steps)
                accel_y = np.cos(t) + 0.1 * np.random.normal(size=time_steps)
                accel_z = 9.8 + 0.05 * np.random.normal(size=time_steps)
                
                # Smooth gyroscope data
                gyro_x = 0.1 * np.sin(2*t) + 0.05 * np.random.normal(size=time_steps)
                gyro_y = 0.1 * np.cos(2*t) + 0.05 * np.random.normal(size=time_steps)
                gyro_z = 0.05 * np.random.normal(size=time_steps)
                
                stress_level = np.random.uniform(0, 0.3)
                fatigue_level = np.random.uniform(0, 0.4)
                anomaly_level = 0.0
                
            else:  # 30% stressed/fatigued/anomalous
                # Stressed movement (more jittery)
                if i < num_samples * 0.85:
                    accel_x = np.sin(t) + 0.3 * np.random.normal(size=time_steps)
                    accel_y = np.cos(t) + 0.3 * np.random.normal(size=time_steps)
                    accel_z = 9.8 + 0.2 * np.random.normal(size=time_steps)
                    
                    # Tremor in gyroscope
                    tremor_freq = 8  # 8Hz tremor
                    gyro_x = 0.1 * np.sin(2*t) + 0.1 * np.sin(tremor_freq*t) + 0.1 * np.random.normal(size=time_steps)
                    gyro_y = 0.1 * np.cos(2*t) + 0.1 * np.cos(tremor_freq*t) + 0.1 * np.random.normal(size=time_steps)
                    gyro_z = 0.1 * np.random.normal(size=time_steps)
                    
                    stress_level = np.random.uniform(0.6, 1.0)
                    fatigue_level = np.random.uniform(0.2, 0.6)
                    anomaly_level = np.random.uniform(0, 0.3)
                
                # Fatigued movement (slower, less coordinated)
                else:
                    accel_x = 0.5 * np.sin(0.5*t) + 0.2 * np.random.normal(size=time_steps)
                    accel_y = 0.5 * np.cos(0.5*t) + 0.2 * np.random.normal(size=time_steps)
                    accel_z = 9.8 + 0.3 * np.random.normal(size=time_steps)
                    
                    gyro_x = 0.05 * np.sin(t) + 0.1 * np.random.normal(size=time_steps)
                    gyro_y = 0.05 * np.cos(t) + 0.1 * np.random.normal(size=time_steps)
                    gyro_z = 0.1 * np.random.normal(size=time_steps)
                    
                    stress_level = np.random.uniform(0.2, 0.5)
                    fatigue_level = np.random.uniform(0.7, 1.0)
                    anomaly_level = np.random.uniform(0.1, 0.5)
            
            # Combine features
            features = np.column_stack([
                accel_x, accel_y, accel_z,
                gyro_x, gyro_y, gyro_z,
                np.gradient(accel_x), np.gradient(accel_y),  # Jerk features
                np.convolve(accel_x, np.ones(10)/10, mode='same'),  # Smoothed
                np.convolve(accel_y, np.ones(10)/10, mode='same'),
                np.sqrt(accel_x**2 + accel_y**2 + accel_z**2),  # Magnitude
                np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
            ])
            
            data.append(features)
            labels.append([stress_level, fatigue_level, anomaly_level])
        
        return np.array(data), np.array(labels)
    
    def train_model(self, epochs=100, batch_size=32):
        """Train the movement analysis model"""
        print("🏋️ Training movement analysis model...")
        
        # Load/generate data
        X, y = self.load_synthetic_movement_data()
        
        # Prepare sequence data
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X)):
            sequences, sequence_labels = self.prepare_sequence_data(
                X[i], y[i], self.sequence_length
            )
            X_sequences.extend(sequences)
            y_sequences.extend(sequence_labels)
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_sequences, y_sequences, test_size=0.2, random_state=42
        )
        
        # Build model
        self.build_lstm_model()
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'movement_stress': 'mse',
                'movement_fatigue': 'mse',
                'movement_anomaly': 'mse'
            },
            metrics={
                'movement_stress': ['mae'],
                'movement_fatigue': ['mae'],
                'movement_anomaly': ['mae']
            }
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                'models/trained_models/movement_analyzer_best.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train,
            {
                'movement_stress': y_train[:, 0],
                'movement_fatigue': y_train[:, 1],
                'movement_anomaly': y_train[:, 2]
            },
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                X_test,
                {
                    'movement_stress': y_test[:, 0],
                    'movement_fatigue': y_test[:, 1],
                    'movement_anomaly': y_test[:, 2]
                }
            ),
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save final model
        self.model.save('models/trained_models/movement_analyzer.h5')
        
        # Save training history
        with open('models/training_history/movement_training_history.json', 'w') as f:
            json.dump(history.history, f)
        
        print("✅ Movement model training complete!")
        return history

def main():
    trainer = MovementModelTrainer()
    history = trainer.train_model(epochs=50)
    
    # Evaluate model
    if trainer.model:
        print("\n📊 Model Evaluation:")
        print("Model architecture summary:")
        trainer.model.summary()

if __name__ == "__main__":
    main()