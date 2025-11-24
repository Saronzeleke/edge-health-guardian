# models/training/fusion_trainer.py
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import json

class SensorFusionTrainer:
    def __init__(self):
        self.model = None
        self.face_feature_dim = 32  # Output from face model
        self.movement_feature_dim = 24  # Output from movement model
        self.hr_feature_dim = 3  # Heart rate features
        
    def build_fusion_model(self):
        """Build sensor fusion model with attention mechanism"""
        # Face features input
        face_input = tf.keras.Input(shape=(self.face_feature_dim,), name='face_features')
        
        # Movement features input  
        movement_input = tf.keras.Input(shape=(self.movement_feature_dim,), name='movement_features')
        
        # Heart rate features input (optional)
        hr_input = tf.keras.Input(shape=(self.hr_feature_dim,), name='hr_features')
        
        # Feature transformation with batch normalization
        face_branch = layers.Dense(64, activation='relu')(face_input)
        face_branch = layers.BatchNormalization()(face_branch)
        face_branch = layers.Dropout(0.2)(face_branch)
        
        movement_branch = layers.Dense(64, activation='relu')(movement_input)
        movement_branch = layers.BatchNormalization()(movement_branch)
        movement_branch = layers.Dropout(0.2)(movement_branch)
        
        hr_branch = layers.Dense(16, activation='relu')(hr_input)
        hr_branch = layers.BatchNormalization()(hr_branch)
        hr_branch = layers.Dropout(0.2)(hr_branch)
        
        # Concatenate all features
        concatenated = layers.concatenate([face_branch, movement_branch, hr_branch])
        
        # Attention mechanism for feature weighting
        attention = layers.Dense(concatenated.shape[-1], activation='tanh')(concatenated)
        attention = layers.Dense(concatenated.shape[-1], activation='softmax')(attention)
        
        # Apply attention
        attended_features = layers.multiply([concatenated, attention])
        
        # Main fusion network
        x = layers.Dense(128, activation='relu')(attended_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(32, activation='relu')(x)
        
        # Multi-output for final health assessment
        stress_output = layers.Dense(1, activation='sigmoid', name='final_stress')(x)
        fatigue_output = layers.Dense(1, activation='sigmoid', name='final_fatigue')(x)
        anomaly_output = layers.Dense(1, activation='sigmoid', name='final_anomaly')(x)
        
        self.model = models.Model(
            inputs=[face_input, movement_input, hr_input],
            outputs=[stress_output, fatigue_output, anomaly_output]
        )
        
        return self.model
    
    def generate_fusion_data(self, num_samples=5000):
        """Generate synthetic fusion training data"""
        print("📊 Generating fusion training data...")
        
        # Face features (from face model output)
        face_features = np.random.normal(0, 1, (num_samples, self.face_feature_dim))
        
        # Movement features (from movement model output)
        movement_features = np.random.normal(0, 1, (num_samples, self.movement_feature_dim))
        
        # Heart rate features (mean, std, variability)
        hr_features = np.column_stack([
            np.random.normal(72, 10, num_samples),  # Mean HR
            np.random.normal(3, 1, num_samples),    # HR std
            np.random.normal(0.1, 0.05, num_samples)  # HR variability
        ])
        
        # Generate labels based on feature combinations
        stress_labels = np.zeros(num_samples)
        fatigue_labels = np.zeros(num_samples)
        anomaly_labels = np.zeros(num_samples)
        
        for i in range(num_samples):
            # Stress correlates with high face tension and movement tremors
            face_stress_indicator = np.mean(face_features[i, :8])  # First 8 features related to tension
            movement_stress_indicator = np.mean(movement_features[i, 6:12])  # Tremor-related features
            hr_stress_indicator = hr_features[i, 2]  # HR variability
            
            stress_score = (
                0.4 * self._sigmoid(face_stress_indicator) +
                0.4 * self._sigmoid(movement_stress_indicator) + 
                0.2 * hr_stress_indicator
            )
            stress_labels[i] = np.clip(stress_score + np.random.normal(0, 0.1), 0, 1)
            
            # Fatigue correlates with specific patterns
            face_fatigue_indicator = np.mean(face_features[i, 8:16])  # Eye-related features
            movement_fatigue_indicator = np.mean(movement_features[i, :6])  # Movement smoothness
            
            fatigue_score = (
                0.5 * self._sigmoid(face_fatigue_indicator) +
                0.5 * self._sigmoid(movement_fatigue_indicator)
            )
            fatigue_labels[i] = np.clip(fatigue_score + np.random.normal(0, 0.1), 0, 1)
            
            # Anomaly detection (rare events)
            anomaly_score = np.abs(face_features[i, 0]) * np.abs(movement_features[i, 0])
            anomaly_labels[i] = 1.0 if anomaly_score > 1.5 and np.random.random() < 0.1 else 0.0
        
        return [face_features, movement_features, hr_features], [stress_labels, fatigue_labels, anomaly_labels]
    
    def _sigmoid(self, x):
        """Helper sigmoid function"""
        return 1 / (1 + np.exp(-x))
    
    def train_fusion_model(self, epochs=100, batch_size=32):
        """Train the sensor fusion model"""
        print("🏋️ Training sensor fusion model...")
        
        # Generate training data
        X, y = self.generate_fusion_data(5000)
        
        # Split data
        X_train, X_test = [], []
        y_train, y_test = [], []
        
        for i in range(3):  # For each input type
            X_tr, X_te, y_tr, y_te = train_test_split(
                X[i], y[i], test_size=0.2, random_state=42
            )
            X_train.append(X_tr)
            X_test.append(X_te)
        
        # Build model
        self.build_fusion_model()
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'final_stress': 'binary_crossentropy',
                'final_fatigue': 'binary_crossentropy',
                'final_anomaly': 'binary_crossentropy'
            },
            metrics={
                'final_stress': ['accuracy', 'precision', 'recall'],
                'final_fatigue': ['accuracy', 'precision', 'recall'],
                'final_anomaly': ['accuracy', 'precision', 'recall']
            }
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                'models/trained_models/fusion_engine_best.h5',
                monitor='val_final_stress_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            {
                'face_features': X_train[0],
                'movement_features': X_train[1],
                'hr_features': X_train[2]
            },
            {
                'final_stress': y_train[0],
                'final_fatigue': y_train[1],
                'final_anomaly': y_train[2]
            },
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                {
                    'face_features': X_test[0],
                    'movement_features': X_test[1],
                    'hr_features': X_test[2]
                },
                {
                    'final_stress': y_test[0],
                    'final_fatigue': y_test[1],
                    'final_anomaly': y_test[2]
                }
            ),
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save final model
        self.model.save('models/trained_models/fusion_engine.h5')
        
        # Save training history
        with open('models/training_history/fusion_training_history.json', 'w') as f:
            json.dump(history.history, f)
        
        print("✅ Sensor fusion model training complete!")
        return history

def main():
    trainer = SensorFusionTrainer()
    history = trainer.train_fusion_model(epochs=80)
    
    if trainer.model:
        print("\n📊 Fusion Model Summary:")
        trainer.model.summary()

if __name__ == "__main__":
    main()