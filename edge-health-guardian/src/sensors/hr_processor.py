# src/sensors/hr_processor.py
import numpy as np
from scipy import signal, fft
from collections import deque
import threading
from typing import Dict, List, Optional, Tuple

class HeartRateAnalyzer:
    def __init__(self, sampling_rate: int = 100):
        self.sampling_rate = sampling_rate
        self.ppg_buffer = deque(maxlen=sampling_rate * 10)  # 10-second buffer
        self.ecg_buffer = deque(maxlen=sampling_rate * 10)
        
        # HRV analysis parameters
        self.hrv_features = {}
        self.last_r_peaks = []
        
        self.lock = threading.Lock()
        
    def add_ppg_data(self, ppg_signal: List[float]):
        """Add PPG (Photoplethysmogram) data"""
        with self.lock:
            self.ppg_buffer.extend(ppg_signal)
            
    def add_ecg_data(self, ecg_signal: List[float]):
        """Add ECG data"""
        with self.lock:
            self.ecg_buffer.extend(ecg_signal)
    
    def extract_heart_rate_features(self) -> Dict:
        """Extract comprehensive heart rate features"""
        with self.lock:
            features = {}
            
            # Process PPG data if available
            if len(self.ppg_buffer) >= self.sampling_rate * 5:  # At least 5 seconds
                ppg_features = self._analyze_ppg_signal(list(self.ppg_buffer))
                features.update(ppg_features)
            
            # Process ECG data if available
            if len(self.ecg_buffer) >= self.sampling_rate * 5:
                ecg_features = self._analyze_ecg_signal(list(self.ecg_buffer))
                features.update(ecg_features)
            
            # Calculate Heart Rate Variability (HRV) metrics
            if self.last_r_peaks and len(self.last_r_peaks) > 10:
                hrv_features = self._calculate_hrv_metrics(self.last_r_peaks)
                features.update(hrv_features)
            
            return features
    
    def _analyze_ppg_signal(self, ppg_signal: List[float]) -> Dict:
        """Analyze PPG signal for heart rate and stress indicators"""
        features = {}
        
        # Preprocess signal
        ppg_signal = np.array(ppg_signal)
        ppg_filtered = self._bandpass_filter(ppg_signal, low=0.5, high=5.0)
        
        # Find peaks (heart beats)
        peaks, properties = signal.find_peaks(
            ppg_filtered, 
            height=np.mean(ppg_filtered) + np.std(ppg_filtered),
            distance=self.sampling_rate * 0.4  # Minimum 0.4s between beats
        )
        
        if len(peaks) >= 2:
            # Calculate heart rate
            rr_intervals = np.diff(peaks) / self.sampling_rate  # Convert to seconds
            heart_rate = 60.0 / np.mean(rr_intervals)  # BPM
            
            features['heart_rate_bpm'] = heart_rate
            features['hr_confidence'] = len(peaks) / (len(ppg_signal) / self.sampling_rate)
            
            # PPG waveform analysis
            waveform_features = self._analyze_ppg_waveform(ppg_filtered, peaks)
            features.update(waveform_features)
        
        return features
    
    def _analyze_ecg_signal(self, ecg_signal: List[float]) -> Dict:
        """Analyze ECG signal for R-peaks and HRV"""
        features = {}
        
        ecg_signal = np.array(ecg_signal)
        
        # Bandpass filter for QRS complex
        ecg_filtered = self._bandpass_filter(ecg_signal, low=5.0, high=15.0)
        
        # Find R-peaks using Pan-Tompkins algorithm
        r_peaks = self._pan_tompkins_detection(ecg_filtered)
        self.last_r_peaks = r_peaks
        
        if len(r_peaks) >= 2:
            # Calculate heart rate from R-R intervals
            rr_intervals = np.diff(r_peaks) / self.sampling_rate
            heart_rate = 60.0 / np.mean(rr_intervals)
            
            features['ecg_heart_rate_bpm'] = heart_rate
            features['r_peak_count'] = len(r_peaks)
            
            # Additional ECG features
            ecg_morphology = self._analyze_ecg_morphology(ecg_filtered, r_peaks)
            features.update(ecg_morphology)
        
        return features
    
    def _pan_tompkins_detection(self, ecg_signal: np.ndarray) -> List[int]:
        """Pan-Tompkins QRS detection algorithm"""
        # Derivative
        derivative = np.diff(ecg_signal)
        
        # Squaring
        squared = derivative ** 2
        
        # Moving window integration
        window_size = int(0.15 * self.sampling_rate)  # 150ms window
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        # Adaptive thresholding
        threshold = np.mean(integrated) + 0.5 * np.std(integrated)
        
        # Find peaks above threshold
        peaks, _ = signal.find_peaks(
            integrated, 
            height=threshold,
            distance=int(0.3 * self.sampling_rate)  # Minimum 300ms between QRS
        )
        
        return peaks.tolist()
    
    def _analyze_ppg_waveform(self, ppg_signal: np.ndarray, peaks: List[int]) -> Dict:
        """Analyze PPG waveform morphology"""
        features = {}
        
        if len(peaks) < 3:
            return features
        
        # Analyze individual pulses
        systolic_peaks = []
        diastolic_troughs = []
        augmentation_indices = []
        
        for i in range(1, len(peaks)-1):
            pulse_segment = ppg_signal[peaks[i-1]:peaks[i+1]]
            
            # Find systolic peak (maximum)
            systolic_idx = np.argmax(pulse_segment)
            systolic_peaks.append(pulse_segment[systolic_idx])
            
            # Find diastolic trough (minimum after systolic peak)
            if systolic_idx < len(pulse_segment) - 10:
                diastolic_segment = pulse_segment[systolic_idx:]
                diastolic_idx = np.argmin(diastolic_segment)
                diastolic_troughs.append(diastolic_segment[diastolic_idx])
                
                # Augmentation index (indicator of arterial stiffness)
                pulse_height = systolic_peaks[-1] - diastolic_troughs[-1]
                if pulse_height > 0:
                    aug_point = systolic_idx + diastolic_idx
                    aug_height = pulse_segment[aug_point] - diastolic_troughs[-1]
                    augmentation_indices.append(aug_height / pulse_height)
        
        if systolic_peaks and diastolic_troughs:
            features['ppg_pulse_height_mean'] = np.mean([
                sys - dia for sys, dia in zip(systolic_peaks, diastolic_troughs)
            ])
            features['ppg_pulse_height_std'] = np.std([
                sys - dia for sys, dia in zip(systolic_peaks, diastolic_troughs)
            ])
        
        if augmentation_indices:
            features['augmentation_index'] = np.mean(augmentation_indices)
        
        return features
    
    def _analyze_ecg_morphology(self, ecg_signal: np.ndarray, r_peaks: List[int]) -> Dict:
        """Analyze ECG waveform morphology"""
        features = {}
        
        if len(r_peaks) < 3:
            return features
        
        # Analyze QRS complex width and amplitude
        qrs_widths = []
        qrs_amplitudes = []
        
        for r_peak in r_peaks:
            # QRS complex analysis window
            start_idx = max(0, r_peak - int(0.1 * self.sampling_rate))
            end_idx = min(len(ecg_signal), r_peak + int(0.1 * self.sampling_rate))
            
            qrs_segment = ecg_signal[start_idx:end_idx]
            qrs_amplitudes.append(np.max(qrs_segment) - np.min(qrs_segment))
            
            # Simple QRS width estimation
            threshold = 0.5 * (np.max(qrs_segment) + np.min(qrs_segment))
            above_threshold = qrs_segment > threshold
            if np.any(above_threshold):
                qrs_width = np.sum(above_threshold) / self.sampling_rate
                qrs_widths.append(qrs_width)
        
        if qrs_widths:
            features['qrs_width_mean'] = np.mean(qrs_widths)
            features['qrs_width_std'] = np.std(qrs_widths)
        
        if qrs_amplitudes:
            features['qrs_amplitude_mean'] = np.mean(qrs_amplitudes)
            features['qrs_amplitude_std'] = np.std(qrs_amplitudes)
        
        return features
    
    def _calculate_hrv_metrics(self, r_peaks: List[int]) -> Dict:
        """Calculate Heart Rate Variability metrics"""
        features = {}
        
        # Convert to R-R intervals in milliseconds
        rr_intervals = np.diff(r_peaks) * (1000.0 / self.sampling_rate)  # ms
        
        if len(rr_intervals) < 5:
            return features
        
        # Time-domain metrics
        features['hrv_mean_rr'] = np.mean(rr_intervals)
        features['hrv_std_rr'] = np.std(rr_intervals)  # SDNN
        features['hrv_rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        
        # Frequency-domain metrics (using Lomb-Scargle for uneven sampling)
        try:
            # Create time points for RR intervals
            time_points = np.cumsum(rr_intervals) / 1000.0  # Convert to seconds
            
            # Remove DC component
            rr_normalized = rr_intervals - np.mean(rr_intervals)
            
            # Lomb-Scargle periodogram
            frequencies = np.linspace(0.003, 0.4, 1000)  # VLF, LF, HF bands
            power = signal.lombscargle(time_points, rr_normalized, frequencies, normalize=True)
            
            # Frequency bands
            vlf_mask = (frequencies >= 0.003) & (frequencies < 0.04)
            lf_mask = (frequencies >= 0.04) & (frequencies < 0.15)
            hf_mask = (frequencies >= 0.15) & (frequencies <= 0.4)
            
            vlf_power = np.trapz(power[vlf_mask], frequencies[vlf_mask])
            lf_power = np.trapz(power[lf_mask], frequencies[lf_mask])
            hf_power = np.trapz(power[hf_mask], frequencies[hf_mask])
            total_power = vlf_power + lf_power + hf_power
            
            features['hrv_lf_power'] = lf_power
            features['hrv_hf_power'] = hf_power
            features['hrv_lf_hf_ratio'] = lf_power / (hf_power + 1e-8)
            features['hrv_total_power'] = total_power
            
        except Exception as e:
            print(f"HRV frequency analysis failed: {e}")
        
        # Non-linear metrics (approximate)
        features['hrv_sd1'] = np.std(rr_intervals) / np.sqrt(2)  # Poincaré plot SD1
        features['hrv_sd2'] = np.std(rr_intervals) * np.sqrt(2)  # Poincaré plot SD2
        
        return features
    
    def _bandpass_filter(self, signal_data: np.ndarray, low: float, high: float) -> np.ndarray:
        """Apply bandpass filter to signal"""
        nyquist = self.sampling_rate / 2
        low_normalized = low / nyquist
        high_normalized = high / nyquist
        
        b, a = signal.butter(4, [low_normalized, high_normalized], btype='band')
        filtered_signal = signal.filtfilt(b, a, signal_data)
        
        return filtered_signal
    
    def get_stress_indicators(self) -> Dict:
        """Get stress indicators from heart rate analysis"""
        features = self.extract_heart_rate_features()
        
        stress_indicators = {}
        
        # High heart rate indicator
        if 'heart_rate_bpm' in features:
            hr = features['heart_rate_bpm']
            if hr > 100:
                stress_indicators['high_heart_rate'] = min(1.0, (hr - 100) / 40)
        
        # Low HRV (high stress indicator)
        if 'hrv_rmssd' in features:
            rmssd = features['hrv_rmssd']
            if rmssd < 20:  # Low HRV
                stress_indicators['low_hrv'] = min(1.0, (20 - rmssd) / 15)
        
        # LF/HF ratio (sympathetic activity)
        if 'hrv_lf_hf_ratio' in features:
            lf_hf_ratio = features['hrv_lf_hf_ratio']
            if lf_hf_ratio > 3.0:  # High sympathetic activity
                stress_indicators['high_sympathetic'] = min(1.0, lf_hf_ratio / 6.0)
        
        # Overall stress score from HR features
        if stress_indicators:
            stress_indicators['hr_based_stress'] = np.mean(list(stress_indicators.values()))
        
        return stress_indicators