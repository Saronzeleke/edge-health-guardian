import cv2
import numpy as np
import threading
from collections import deque
from typing import Optional, Callable, Dict, List, Union
import mediapipe as mp
import logging
import time

logger = logging.getLogger(__name__)


class FaceMicroExpressionAnalyzer:
    def __init__(self, buffer_size: int = 30):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.eyebrow_indices = [70, 63, 105, 66, 107, 55, 65, 52]
        self.eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173]
        self.mouth_indices = [61, 84, 17, 314, 405, 320, 307, 375]

        self.previous_landmarks = None
        self.expression_buffer = deque(maxlen=buffer_size)

    def _validate_frame(self, frame: np.ndarray) -> bool:
        if frame is None or not isinstance(frame, np.ndarray):
            return False
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return False
        if frame.dtype != np.uint8:
            return False
        return True

    def extract_micro_expressions(self, frame: np.ndarray) -> Dict:
        features = {'success': 0.0, 'timestamp': time.time()}

        if not self._validate_frame(frame):
            logger.warning("Invalid frame input for micro-expression analysis")
            return features

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                features['face_detected'] = 0.0
                return features

            landmarks = results.multi_face_landmarks[0]
            current_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])

            features.update({
                'face_detected': 1.0,
                'eyebrow_tension': self._calculate_facial_displacement(current_landmarks, self.eyebrow_indices),
                'eye_aperture': self._calculate_eye_aperture(current_landmarks),
                'mouth_tension': self._calculate_mouth_tension(current_landmarks),
                'facial_symmetry': self._calculate_facial_symmetry(current_landmarks)
            })

            if self.previous_landmarks is not None:
                motion_vectors = current_landmarks - self.previous_landmarks
                features['micro_expression_intensity'] = float(np.mean(np.linalg.norm(motion_vectors, axis=1)))
            else:
                features['micro_expression_intensity'] = 0.0

            features['success'] = 1.0
            self.previous_landmarks = current_landmarks

            self.expression_buffer.append(features)
            features.update(self._aggregate_expression_features())

        except Exception as e:
            logger.error(f"Micro-expression extraction failed: {str(e)}", exc_info=True)
            features['error'] = str(e)

        return features

    def _calculate_facial_displacement(self, landmarks: np.ndarray, indices: List[int]) -> float:
        try:
            region = landmarks[indices]
            centroid = np.mean(region, axis=0)
            return float(np.mean(np.linalg.norm(region - centroid, axis=1)))
        except Exception:
            return 0.0

    def _calculate_eye_aperture(self, landmarks: np.ndarray) -> float:
        try:
            left_eye_height = np.linalg.norm(landmarks[159] - landmarks[145])
            right_eye_height = np.linalg.norm(landmarks[386] - landmarks[374])
            return float((left_eye_height + right_eye_height) / 2.0)
        except Exception:
            return 0.0

    def _calculate_mouth_tension(self, landmarks: np.ndarray) -> float:
        try:
            left_tension = np.linalg.norm(landmarks[61] - landmarks[13])
            right_tension = np.linalg.norm(landmarks[291] - landmarks[13])
            return float((left_tension + right_tension) / 2.0)
        except Exception:
            return 0.0

    def _calculate_facial_symmetry(self, landmarks: np.ndarray) -> float:
        try:
            left_features = landmarks[[33, 7, 163, 144, 145, 153, 154, 155]]
            right_features = landmarks[[362, 382, 381, 380, 374, 373, 390, 249]]
            right_features[:, 0] = 1.0 - right_features[:, 0]
            return float(1.0 / (1.0 + np.mean(np.linalg.norm(left_features - right_features, axis=1))))
        except Exception:
            return 0.0

    def _aggregate_expression_features(self) -> Dict:
        if not self.expression_buffer:
            return {}
        aggregated = {}
        keys = [k for k in self.expression_buffer[0].keys() if k not in ['success', 'timestamp', 'error']]
        for key in keys:
            values = [f[key] for f in self.expression_buffer if key in f and isinstance(f[key], (int, float))]
            if values:
                aggregated[f'{key}_mean'] = float(np.mean(values))
                aggregated[f'{key}_std'] = float(np.std(values))
                aggregated[f'{key}_trend'] = float(np.polyfit(range(len(values)), values, 1)[0]) if len(values) > 1 else 0.0
        return aggregated


class CameraProcessor:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self.frame_callback = None
        self.face_analyzer = FaceMicroExpressionAnalyzer()
        self.process_thread = None
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0
        self.model_input_size = (96, 96)  
    def start_capture(self, callback: Callable):
        """Start camera capture with robust initialization"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False

            # Configure camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
            logger.info(f"Model input size: {self.model_input_size}")

            self.frame_callback = callback
            self.is_running = True

            self.process_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.process_thread.start()
            
            logger.info("Camera capture started successfully")
            return True

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            self._safe_camera_release()
            return False

    def _safe_camera_release(self):
        """Safely release camera resources"""
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"Error releasing camera: {e}")
            finally:
                self.cap = None

    def _capture_loop(self):
        """Main capture loop with comprehensive error handling"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running:
            try:
                # Read frame from camera
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_errors += 1
                    logger.warning(f"Failed to read frame (error {consecutive_errors}/{max_consecutive_errors})")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error("Max consecutive frame read errors reached")
                        break
                    
                    time.sleep(0.1)  
                    continue
                
                # Reset error counter on successful frame read
                consecutive_errors = 0
                
                # Calculate FPS
                self._update_fps()
                
                # Process frame and extract features
                features = self.face_analyzer.extract_micro_expressions(frame)
                
                # Prepare callback data as a SINGLE dictionary
                callback_data = {
                    'frame_tensor': self._frame_to_tensor(frame),  
                    'features': features,                       
                    'metadata': {
                        'frame_count': self.frame_count,
                        'timestamp': time.time(),
                        'fps': self.fps,
                        'original_frame_shape': frame.shape,
                        'model_input_shape': self.model_input_size
                    }
                }
                
                # Execute callback with SINGLE argument
                if self.frame_callback:
                    try:
                        self.frame_callback(callback_data)
                    except Exception as e:
                        logger.error(f"Frame callback execution failed: {e}")
                        # Don't break the loop for callback errors

                self.frame_count += 1

            except Exception as e:
                logger.error(f"Unexpected error in capture loop: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    break
                time.sleep(0.1)

        logger.info("Camera capture loop ended")
        self._safe_camera_release()

    def _frame_to_tensor(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to 4D tensor format for ML models (96x96 as required)"""
        try:
            if frame is None:
                logger.warning("Received None frame in tensor conversion")
                return np.zeros((1, 96, 96, 3), dtype=np.float32)
            
            # Resize frame to match model input size (96x96)
            frame_resized = cv2.resize(frame, self.model_input_size)
            
            # Convert BGR to RGB (MediaPipe uses RGB, models might expect RGB)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values to [0, 1]
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            
            # Add batch dimension: (1, 96, 96, 3)
            tensor = np.expand_dims(frame_normalized, axis=0)
            
            logger.debug(f"Frame tensor created: {tensor.shape}, range: [{tensor.min():.3f}, {tensor.max():.3f}]")
            return tensor
            
        except Exception as e:
            logger.error(f"Frame to tensor conversion failed: {e}")
            # Return tensor with correct model input shape
            return np.zeros((1, 96, 96, 3), dtype=np.float32)

    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        
        if time_diff > 0:
            current_fps = 1.0 / time_diff
            # Smooth FPS calculation
            self.fps = 0.8 * self.fps + 0.2 * current_fps
        
        self.last_frame_time = current_time

    def stop_capture(self):
        """Stop camera capture gracefully"""
        self.is_running = False
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=5.0)
            if self.process_thread.is_alive():
                logger.warning("Camera thread did not terminate gracefully")
        
        self._safe_camera_release()
        logger.info("Camera capture stopped")

    def get_camera_info(self) -> Dict:
        """Get camera information and status"""
        if not self.cap:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'running' if self.is_running else 'stopped',
            'camera_id': self.camera_id,
            'frame_count': self.frame_count,
            'fps': self.fps,
            'model_input_size': self.model_input_size,
            'is_opened': self.cap.isOpened()
        }

    def set_model_input_size(self, width: int, height: int):
        """Set the model input size (for different models)"""
        self.model_input_size = (width, height)
        logger.info(f"Model input size set to: {self.model_input_size}")