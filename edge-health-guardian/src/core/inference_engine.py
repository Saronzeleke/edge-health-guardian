# src/core/inference_engine.py
import tensorflow as tf
import numpy as np
import time
import logging
from typing import Dict, List, Optional

class ArmOptimizedInference:
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.delegate = None
        
    def load_model(self, model_path: str):
        """Load TFLite model with XNNPACK delegate for Arm optimization"""
        try:
            # Enable XNNPACK delegate for Arm CPU acceleration
            self.delegate = tf.lite.LoadDelegate('XNNPACK')
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path,
                experimental_delegates=[self.delegate]
            )
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logging.info(f"Model loaded successfully with XNNPACK delegate")
            
        except Exception as e:
            logging.warning(f"XNNPACK not available, falling back to CPU: {e}")
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
    
    def preprocess_input(self, input_data: np.ndarray) -> np.ndarray:
        """Optimized preprocessing for Arm NEON"""
        # Convert to FP32 if needed
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        # Normalize to [-1, 1] range
        input_data = (input_data - 127.5) / 127.5
        
        return input_data
    
    def inference(self, input_data: np.ndarray) -> np.ndarray:
        """Perform optimized inference"""
        if self.interpreter is None:
            raise ValueError("Model not loaded")
        
        # Preprocess input
        processed_data = self.preprocess_input(input_data)
        
        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'], 
            processed_data
        )
        
        # Run inference
        start_time = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - start_time
        
        # Get output
        output_data = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
        
        return output_data, inference_time

class HealthInferenceEngine:
    def __init__(self):
        self.face_analyzer = ArmOptimizedInference()
        self.movement_analyzer = ArmOptimizedInference()
        self.fusion_engine = ArmOptimizedInference()
        
        self.face_model_path = "models/optimized_models/face_analyzer_int8.tflite"
        self.movement_model_path = "models/optimized_models/movement_analyzer_int8.tflite"
        self.fusion_model_path = "models/optimized_models/fusion_engine_int8.tflite"
        
    def initialize_models(self):
        """Initialize all models with optimization"""
        self.face_analyzer.load_model(self.face_model_path)
        self.movement_analyzer.load_model(self.movement_model_path)
        self.fusion_engine.load_model(self.fusion_model_path)
        
    def process_health_data(self, face_data: np.ndarray, 
                          movement_data: np.ndarray,
                          hr_data: Optional[np.ndarray] = None) -> Dict:
        """Main inference pipeline"""
        
        # Parallel inference for face and movement data
        face_output, face_time = self.face_analyzer.inference(face_data)
        movement_output, movement_time = self.movement_analyzer.inference(movement_data)
        
        # Prepare fusion input
        fusion_input = np.concatenate([face_output, movement_output], axis=-1)
        if hr_data is not None:
            fusion_input = np.concatenate([fusion_input, hr_data], axis=-1)
            
        # Final fusion inference
        health_output, fusion_time = self.fusion_engine.inference(fusion_input)
        
        return {
            'stress_score': health_output[0][0],
            'fatigue_level': health_output[0][1],
            'anomaly_confidence': health_output[0][2],
            'inference_times': {
                'face': face_time,
                'movement': movement_time,
                'fusion': fusion_time
            }
        }