# src/ui/cli_interface.py
import os
import time
import threading
from typing import Dict, List
from datetime import datetime

class HealthMonitorCLI:
    def __init__(self):
        self.last_update = None
        self.display_data = {}
        self.lock = threading.Lock()
        
    def initialize(self):
        """Initialize the CLI interface"""
        os.system('clear')
        print("🛡️  Edge Health Guardian - Real-time Health Monitoring")
        print("=" * 60)
        print("Monitoring: Stress | Fatigue | Health Anomalies")
        print("Status: Initializing sensors...")
        print("-" * 60)
        
    def update_display(self, health_results: Dict, health_history: List):
        """Update the CLI display with latest health data"""
        with self.lock:
            self.display_data = {
                'health_results': health_results,
                'health_history': health_history,
                'timestamp': datetime.now()
            }
            
        self._refresh_display()
    
    def _refresh_display(self):
        """Refresh the CLI display"""
        os.system('clear')
        
        data = self.display_data
        results = data.get('health_results', {})
        
        print("🛡️  Edge Health Guardian - Real-time Health Monitoring")
        print("=" * 60)
        print(f"Last Update: {data.get('timestamp', 'N/A')}")
        print("-" * 60)
        
        # Health metrics
        if results:
            stress_score = results.get('stress_score', 0)
            fatigue_level = results.get('fatigue_level', 0)
            anomaly_confidence = results.get('anomaly_confidence', 0)
            
            # Stress display
            stress_bar = self._create_progress_bar(stress_score, 20)
            stress_status = "🟢 Normal" if stress_score < 0.5 else "🟡 Elevated" if stress_score < 0.8 else "🔴 High"
            print(f"Stress:       {stress_bar} {stress_score:.2f} {stress_status}")
            
            # Fatigue display  
            fatigue_bar = self._create_progress_bar(fatigue_level, 20)
            fatigue_status = "🟢 Alert" if fatigue_level < 0.5 else "🟡 Tired" if fatigue_level < 0.8 else "🔴 Fatigued"
            print(f"Fatigue:      {fatigue_bar} {fatigue_level:.2f} {fatigue_status}")
            
            # Anomaly display
            anomaly_bar = self._create_progress_bar(anomaly_confidence, 20)
            anomaly_status = "🟢 Normal" if anomaly_confidence < 0.3 else "🟡 Watch" if anomaly_confidence < 0.7 else "🔴 Anomaly"
            print(f"Anomaly:      {anomaly_bar} {anomaly_confidence:.2f} {anomaly_status}")
            
            # Inference times
            times = results.get('inference_times', {})
            total_time = sum(times.values()) * 1000  # Convert to milliseconds
            print(f"Response:     {total_time:.1f} ms")
            
        else:
            print("Waiting for sensor data...")
        
        print("-" * 60)
        
        # System status
        history = data.get('health_history', [])
        if len(history) > 1:
            recent_stress = [h['results']['stress_score'] for h in history[-10:]]
            avg_stress = sum(recent_stress) / len(recent_stress)
            
            print(f"Trend:        {self._get_trend_arrow(recent_stress)}")
            print(f"Session:      {len(history)} samples collected")
            
        print("=" * 60)
        print("Press Ctrl+C to exit | 🔒 All processing is local")
    
    def _create_progress_bar(self, value: float, length: int) -> str:
        """Create a visual progress bar"""
        bars = int(value * length)
        return "█" * bars + "░" * (length - bars)
    
    def _get_trend_arrow(self, values: List[float]) -> str:
        """Get trend arrow based on recent values"""
        if len(values) < 2:
            return "→ Stable"
            
        if values[-1] > values[0] + 0.1:
            return "↑ Increasing"
        elif values[-1] < values[0] - 0.1:
            return "↓ Decreasing"
        else:
            return "→ Stable"