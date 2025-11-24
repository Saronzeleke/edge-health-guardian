# src/sensors/camera_processor.py
import cv2
import numpy as np
import threading
from typing import Optional, Callable, Dict
import mediapipe as mp

class FaceMicroExpressionAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key landmarks for micro-expression analysis
        self.eyebrow_indices = [70, 63, 105, 66, 107, 55, 65, 52]
        self.eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173]
        self.mouth_indices = [61, 84, 17, 314, 405, 320, 307, 375]
        
        self.previous_landmarks = None
        self.expression_buffer = deque(maxlen=30)
        
    def extract_micro_expressions(self, frame: np.ndarray) -> Optional[Dict]:
        """Extract micro-expression features from face landmarks"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        current_landmarks = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
        
        features = {}
        
        # Calculate eyebrow movements
        eyebrow_displacement = self.calculate_facial_displacement(
            current_landmarks, self.eyebrow_indices
        )
        features['eyebrow_tension'] = eyebrow_displacement
        
        # Calculate eye aperture changes
        eye_aperture = self.calculate_eye_aperture(current_landmarks)
        features['eye_aperture'] = eye_aperture
        
        # Calculate mouth corner movements
        mouth_tension = self.calculate_mouth_tension(current_landmarks)
        features['mouth_tension'] = mouth_tension
        
        # Calculate overall facial symmetry
        symmetry_score = self.calculate_facial_symmetry(current_landmarks)
        features['facial_symmetry'] = symmetry_score
        
        # Temporal analysis for micro-expressions
        if self.previous_landmarks is not None:
            motion_vectors = current_landmarks - self.previous_landmarks
            micro_expression_intensity = np.mean(np.linalg.norm(motion_vectors, axis=1))
            features['micro_expression_intensity'] = micro_expression_intensity
        else:
            features['micro_expression_intensity'] = 0.0
            
        self.previous_landmarks = current_landmarks
        self.expression_buffer.append(features)
        
        return self.aggregate_expression_features()
    
    def calculate_facial_displacement(self, landmarks: np.ndarray, indices: List[int]) -> float:
        """Calculate displacement for specific facial regions"""
        region_landmarks = landmarks[indices]
        centroid = np.mean(region_landmarks, axis=0)
        displacements = np.linalg.norm(region_landmarks - centroid, axis=1)
        return np.mean(displacements)
    
    def calculate_eye_aperture(self, landmarks: np.ndarray) -> float:
        """Calculate eye aperture ratio"""
        # Vertical eye points
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        right_eye_top = landmarks[386]
        right_eye_bottom = landmarks[374]
        
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        
        return (left_eye_height + right_eye_height) / 2.0
    
    def calculate_mouth_tension(self, landmarks: np.ndarray) -> float:
        """Calculate mouth corner tension"""
        left_mouth_corner = landmarks[61]
        right_mouth_corner = landmarks[291]
        mouth_center = landmarks[13]
        
        left_tension = np.linalg.norm(left_mouth_corner - mouth_center)
        right_tension = np.linalg.norm(right_mouth_corner - mouth_center)
        
        return (left_tension + right_tension) / 2.0
    
    def calculate_facial_symmetry(self, landmarks: np.ndarray) -> float:
        """Calculate facial symmetry score"""
        left_face_indices = [33, 7, 163, 144, 145, 153, 154, 155]
        right_face_indices = [362, 382, 381, 380, 374, 373, 390, 249]
        
        left_features = landmarks[left_face_indices]
        right_features = landmarks[right_face_indices]
        
        # Mirror right features
        right_features_mirrored = right_features.copy()
        right_features_mirrored[:, 0] = 1.0 - right_features_mirrored[:, 0]  # Mirror x-coordinate
        
        symmetry_errors = np.linalg.norm(left_features - right_features_mirrored, axis=1)
        symmetry_score = 1.0 / (1.0 + np.mean(symmetry_errors))
        
        return symmetry_score
    
    def aggregate_expression_features(self) -> Dict:
        """Aggregate features over time window"""
        if not self.expression_buffer:
            return {}
            
        aggregated = {}
        for key in self.expression_buffer[0].keys():
            values = [frame[key] for frame in self.expression_buffer]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_trend'] = np.polyfit(range(len(values)), values, 1)[0]
            
        return aggregated

class CameraProcessor:
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self.frame_callback = None
        self.face_analyzer = FaceMicroExpressionAnalyzer()
        self.process_thread = None
        
    def start_capture(self, callback: Callable):
        """Start camera capture with callback"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.frame_callback = callback
        self.is_running = True
        
        self.process_thread = threading.Thread(target=self._capture_loop)
        self.process_thread.start()
        
    def _capture_loop(self):
        """Main capture loop"""
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Extract facial features
            expression_features = self.face_analyzer.extract_micro_expressions(frame)
            
            if expression_features and self.frame_callback:
                self.frame_callback(frame, expression_features)
                
    def stop_capture(self):
        """Stop camera capture"""
        self.is_running = False
        if self.process_thread:
            self.process_thread.join()
        if self.cap:
            self.cap.release()