# src/utils/preprocessor.py
import numpy as np
from scipy import signal
import cv2
from typing import Union, List

class DataPreprocessor:
    """Data preprocessing utilities optimized for Arm devices"""
    
    @staticmethod
    def preprocess_face_image(image: np.ndarray, target_size: tuple = (96, 96)) -> np.ndarray:
        """Preprocess face image for model input"""
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        image = cv2.resize(image, target_size)
        
        # Normalize to [0, 1] range
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension if needed
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    @staticmethod
    def preprocess_imu_data(accel_data: List[float], gyro_data: List[float]) -> np.ndarray:
        """Preprocess IMU data for movement analysis"""
        accel_array = np.array(accel_data)
        gyro_array = np.array(gyro_data)
        
        # Remove gravity component from accelerometer (high-pass filter)
        accel_filtered = signal.filtfilt([1, -1], [1, -0.99], accel_array, axis=0)
        
        # Normalize data
        accel_normalized = (accel_filtered - np.mean(accel_filtered, axis=0)) / (np.std(accel_filtered, axis=0) + 1e-8)
        gyro_normalized = (gyro_array - np.mean(gyro_array, axis=0)) / (np.std(gyro_array, axis=0) + 1e-8)
        
        # Combine features
        features = np.column_stack([accel_normalized, gyro_normalized])
        
        return features.astype(np.float32)
    
    @staticmethod
    def normalize_features(features: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """Normalize features using specified method"""
        if method == 'zscore':
            return (features - np.mean(features)) / (np.std(features) + 1e-8)
        elif method == 'minmax':
            return (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)
        elif method == 'robust':
            median = np.median(features)
            iqr = np.percentile(features, 75) - np.percentile(features, 25)
            return (features - median) / (iqr + 1e-8)
        else:
            return features
    
    @staticmethod
    def extract_temporal_features(signal_data: np.ndarray, window_size: int = 50) -> np.ndarray:
        """Extract temporal features from signal data"""
        features = []
        
        for i in range(len(signal_data) - window_size + 1):
            window = signal_data[i:i + window_size]
            
            # Statistical features
            mean = np.mean(window, axis=0)
            std = np.std(window, axis=0)
            variance = np.var(window, axis=0)
            rms = np.sqrt(np.mean(window**2, axis=0))
            
            # Frequency features (simplified FFT)
            fft_vals = np.abs(np.fft.rfft(window, axis=0))
            dominant_freq = np.argmax(fft_vals, axis=0)
            spectral_energy = np.sum(fft_vals**2, axis=0)
            
            window_features = np.concatenate([
                mean, std, variance, rms, dominant_freq, spectral_energy
            ])
            features.append(window_features)
        
        return np.array(features)

class ImageAugmenter:
    """Image augmentation for training data (on-device)"""
    
    @staticmethod
    def augment_face_image(image: np.ndarray) -> List[np.ndarray]:
        """Apply augmentations to face images"""
        augmented = [image]
        
        # Flip horizontally
        flipped = cv2.flip(image, 1)
        augmented.append(flipped)
        
        # Brightness variations
        for alpha in [0.8, 1.2]:
            bright = np.clip(image * alpha, 0, 1)
            augmented.append(bright)
        
        # Small rotations
        for angle in [-5, 5]:
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            augmented.append(rotated)
        
        return augmented