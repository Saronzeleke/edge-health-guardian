# src/sensors/imu_processor.py
import numpy as np
from scipy import signal
from collections import deque
import threading
from typing import Dict, List, Optional

class MovementPatternAnalyzer:
    def __init__(self, sampling_rate: int = 100):
        self.sampling_rate = sampling_rate
        self.accel_buffer = deque(maxlen=sampling_rate * 5)  # 5-second buffer
        self.gyro_buffer = deque(maxlen=sampling_rate * 5)
        
        # Frequency bands for physiological analysis
        self.lf_band = (0.04, 0.15)  # Low frequency (stress-related)
        self.hf_band = (0.15, 0.4)   # High frequency (parasympathetic)
        
        self.lock = threading.Lock()
        
    def add_imu_data(self, accelerometer: List[float], gyroscope: List[float]):
        """Add new IMU data to buffers"""
        with self.lock:
            self.accel_buffer.append(accelerometer)
            self.gyro_buffer.append(gyroscope)
            
    def extract_movement_features(self) -> Dict:
        """Extract movement-based health features"""
        with self.lock:
            if len(self.accel_buffer) < self.sampling_rate:
                return {}
                
            accel_data = np.array(list(self.accel_buffer))
            gyro_data = np.array(list(self.gyro_buffer))
            
            features = {}
            
            # Time-domain features
            features.update(self._extract_time_domain_features(accel_data, 'accel'))
            features.update(self._extract_time_domain_features(gyro_data, 'gyro'))
            
            # Frequency-domain features
            features.update(self._extract_frequency_features(accel_data, 'accel'))
            features.update(self._extract_frequency_features(gyro_data, 'gyro'))
            
            # Movement pattern analysis
            features.update(self._analyze_movement_patterns(accel_data, gyro_data))
            
            return features
    
    def _extract_time_domain_features(self, data: np.ndarray, sensor_type: str) -> Dict:
        """Extract time-domain statistical features"""
        features = {}
        
        # Statistical moments
        features[f'{sensor_type}_mean'] = np.mean(data, axis=0)
        features[f'{sensor_type}_std'] = np.std(data, axis=0)
        features[f'{sensor_type}_variance'] = np.var(data, axis=0)
        
        # Signal magnitude area (SMA) - indicator of overall movement intensity
        sma = np.mean(np.sum(np.abs(data), axis=1))
        features[f'{sensor_type}_sma'] = sma
        
        # Zero-crossing rate - indicator of movement smoothness
        zero_crossings = np.sum(np.diff(np.sign(data), axis=0) != 0, axis=0)
        features[f'{sensor_type}_zero_crossing'] = zero_crossings / len(data)
        
        return features
    
    def _extract_frequency_features(self, data: np.ndarray, sensor_type: str) -> Dict:
        """Extract frequency-domain features using FFT"""
        features = {}
        
        for axis in range(data.shape[1]):
            axis_data = data[:, axis]
            
            # Remove DC component
            axis_data = axis_data - np.mean(axis_data)
            
            # Apply Hanning window
            windowed_data = axis_data * np.hanning(len(axis_data))
            
            # Compute FFT
            fft_data = np.fft.rfft(windowed_data)
            freqs = np.fft.rfftfreq(len(windowed_data), 1.0/self.sampling_rate)
            power_spectrum = np.abs(fft_data) ** 2
            
            # Extract band power
            lf_mask = (freqs >= self.lf_band[0]) & (freqs <= self.lf_band[1])
            hf_mask = (freqs >= self.hf_band[0]) & (freqs <= self.hf_band[1])
            
            lf_power = np.sum(power_spectrum[lf_mask])
            hf_power = np.sum(power_spectrum[hf_mask])
            total_power = np.sum(power_spectrum)
            
            features[f'{sensor_type}_axis{axis}_lf_power'] = lf_power
            features[f'{sensor_type}_axis{axis}_hf_power'] = hf_power
            features[f'{sensor_type}_axis{axis}_lf_hf_ratio'] = lf_power / (hf_power + 1e-8)
            features[f'{sensor_type}_axis{axis}_total_power'] = total_power
            
        return features
    
    def _analyze_movement_patterns(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> Dict:
        """Analyze specific movement patterns indicative of stress/fatigue"""
        features = {}
        
        # Tremor detection in hands
        tremor_features = self._detect_tremors(accel_data, gyro_data)
        features.update(tremor_features)
        
        # Postural sway analysis
        sway_features = self._analyze_postural_sway(accel_data)
        features.update(sway_features)
        
        # Movement smoothness (jerk analysis)
        smoothness_features = self._analyze_movement_smoothness(accel_data)
        features.update(smoothness_features)
        
        return features
    
    def _detect_tremors(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> Dict:
        """Detect physiological tremors (4-12 Hz)"""
        features = {}
        
        for i, data in enumerate([accel_data, gyro_data]):
            sensor_type = 'accel' if i == 0 else 'gyro'
            
            for axis in range(data.shape[1]):
                axis_data = data[:, axis]
                
                # Bandpass filter for tremor frequencies (4-12 Hz)
                nyquist = self.sampling_rate / 2
                low = 4.0 / nyquist
                high = 12.0 / nyquist
                b, a = signal.butter(4, [low, high], btype='band')
                filtered_data = signal.filtfilt(b, a, axis_data)
                
                # Tremor intensity
                tremor_intensity = np.std(filtered_data)
                features[f'{sensor_type}_axis{axis}_tremor_intensity'] = tremor_intensity
                
        return features
    
    def _analyze_postural_sway(self, accel_data: np.ndarray) -> Dict:
        """Analyze postural sway from accelerometer data"""
        features = {}
        
        # Use low-frequency components (< 0.5 Hz) for postural sway
        nyquist = self.sampling_rate / 2
        low = 0.5 / nyquist
        b, a = signal.butter(4, low, btype='low')
        
        for axis in range(min(2, accel_data.shape[1])):  # Only x and y axes
            axis_data = accel_data[:, axis]
            sway_data = signal.filtfilt(b, a, axis_data)
            
            # Sway metrics
            features[f'sway_axis{axis}_range'] = np.ptp(sway_data)
            features[f'sway_axis{axis}_rms'] = np.sqrt(np.mean(sway_data**2))
            features[f'sway_axis{axis}_velocity'] = np.mean(np.abs(np.diff(sway_data)))
            
        return features
    
    def _analyze_movement_smoothness(self, accel_data: np.ndarray) -> Dict:
        """Analyze movement smoothness using jerk (derivative of acceleration)"""
        features = {}
        
        # Calculate jerk (third derivative of position)
        jerk = np.diff(accel_data, n=2, axis=0)
        
        # Normalized jerk score (lower = smoother movements)
        time_span = len(accel_data) / self.sampling_rate
        movement_amplitude = np.ptp(accel_data, axis=0)
        
        for axis in range(jerk.shape[1]):
            normalized_jerk = np.sqrt(
                np.sum(jerk[:, axis]**2) * time_span**5 / movement_amplitude[axis]**2
            )
            features[f'movement_smoothness_axis{axis}'] = 1.0 / (1.0 + normalized_jerk)
            
        return features