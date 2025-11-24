# src/sensors/windows_camera.py
import cv2
import threading
from typing import Optional, Callable
import logging

class WindowsCameraProcessor:
    """Windows-optimized camera processor with DirectShow support"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self.frame_callback = None
        self.process_thread = None
        self.logger = logging.getLogger("WindowsCamera")
        
    def start_capture(self, callback: Callable, use_dshow: bool = True):
        """Start camera capture with Windows optimizations"""
        try:
            # Try DirectShow first (better for Windows)
            if use_dshow:
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                self.logger.error("Failed to open camera")
                return False
            
            # Configure camera settings for Windows
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus for consistency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            
            self.frame_callback = callback
            self.is_running = True
            
            self.process_thread = threading.Thread(target=self._capture_loop)
            self.process_thread.daemon = True
            self.process_thread.start()
            
            self.logger.info("Windows camera capture started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start camera capture: {e}")
            return False
    
    def _capture_loop(self):
        """Main capture loop for Windows"""
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning("Failed to read frame from camera")
                break
                
            if self.frame_callback:
                self.frame_callback(frame)
    
    def stop_capture(self):
        """Stop camera capture"""
        self.is_running = False
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        self.logger.info("Camera capture stopped")
    
    def list_available_cameras(self) -> list:
        """List available cameras on Windows system"""
        available_cameras = []
        
        # Test camera indices from 0 to 10
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        
        self.logger.info(f"Found {len(available_cameras)} available cameras: {available_cameras}")
        return available_cameras