# src/utils/alert_system.py
import pygame
import threading
import time
from typing import Dict, List
import logging

class AlertSystem:
    """Multi-modal alert system for health notifications"""
    
    def __init__(self):
        self.active_alerts = []
        self.alert_history = []
        self.audio_enabled = True
        self.visual_enabled = True
        self.haptic_enabled = False  # For mobile devices
        
        # Initialize pygame for audio (if available)
        try:
            pygame.mixer.init()
            self.audio_available = True
        except:
            self.audio_available = False
            logging.warning("Audio alerts not available")
    
    def trigger_alert(self, alert_data: Dict):
        """Trigger a health alert"""
        alert_id = f"{alert_data['type']}_{int(time.time())}"
        
        alert = {
            'id': alert_id,
            'type': alert_data['type'],
            'level': alert_data['level'],
            'message': alert_data['message'],
            'timestamp': time.time(),
            'acknowledged': False
        }
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Trigger alert modalities based on level
        if alert_data['level'] == 'critical':
            self._trigger_critical_alert(alert)
        elif alert_data['level'] == 'warning':
            self._trigger_warning_alert(alert)
        else:
            self._trigger_info_alert(alert)
        
        logging.warning(f"ALERT: {alert_data['type']} - {alert_data['message']}")
    
    def _trigger_critical_alert(self, alert: Dict):
        """Trigger critical alert (highest priority)"""
        if self.audio_enabled and self.audio_available:
            self._play_alert_sound('critical')
        
        if self.visual_enabled:
            self._show_visual_alert(alert, 'critical')
        
        # Repeat alert until acknowledged
        alert_thread = threading.Thread(target=self._repeat_alert, args=(alert,))
        alert_thread.daemon = True
        alert_thread.start()
    
    def _trigger_warning_alert(self, alert: Dict):
        """Trigger warning alert (medium priority)"""
        if self.audio_enabled and self.audio_available:
            self._play_alert_sound('warning')
        
        if self.visual_enabled:
            self._show_visual_alert(alert, 'warning')
    
    def _trigger_info_alert(self, alert: Dict):
        """Trigger info alert (low priority)"""
        if self.visual_enabled:
            self._show_visual_alert(alert, 'info')
    
    def _play_alert_sound(self, alert_type: str):
        """Play alert sound based on type"""
        try:
            if alert_type == 'critical':
                # Generate critical beep sound
                self._generate_beep(880, 500)  # High frequency, longer duration
                time.sleep(0.2)
                self._generate_beep(880, 500)
            elif alert_type == 'warning':
                self._generate_beep(660, 300)  # Medium frequency
            else:
                self._generate_beep(440, 200)  # Low frequency, short
        except Exception as e:
            logging.error(f"Audio alert failed: {e}")
    
    def _generate_beep(self, frequency: int, duration: int):
        """Generate beep sound (simplified - in real implementation use proper audio)"""
        # This is a placeholder - real implementation would use pygame or similar
        print(f"\aBEEP! Frequency: {frequency}Hz, Duration: {duration}ms")
    
    def _show_visual_alert(self, alert: Dict, level: str):
        """Show visual alert"""
        colors = {
            'critical': '\033[91m',  # Red
            'warning': '\033[93m',   # Yellow
            'info': '\033[94m'       # Blue
        }
        
        reset_color = '\033[0m'
        
        print(f"\n{colors[level]}⚠️  HEALTH ALERT ⚠️{reset_color}")
        print(f"{colors[level]}Type: {alert['type']}{reset_color}")
        print(f"{colors[level]}Level: {alert['level']}{reset_color}")
        print(f"{colors[level]}Message: {alert['message']}{reset_color}")
        print(f"{colors[level]}Time: {time.ctime(alert['timestamp'])}{reset_color}")
        print(f"{colors[level]}{'='*50}{reset_color}\n")
    
    def _repeat_alert(self, alert: Dict):
        """Repeat critical alert until acknowledged"""
        while not alert['acknowledged'] and alert in self.active_alerts:
            time.sleep(10)  # Repeat every 10 seconds
            if not alert['acknowledged']:
                self._trigger_critical_alert(alert)
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge and deactivate an alert"""
        for alert in self.active_alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                self.active_alerts.remove(alert)
                logging.info(f"Alert acknowledged: {alert_id}")
                break
    
    def get_active_alerts(self) -> List[Dict]:
        """Get list of active alerts"""
        return self.active_alerts.copy()
    
    def get_alert_summary(self) -> Dict:
        """Get alert statistics"""
        critical_count = sum(1 for alert in self.active_alerts if alert['level'] == 'critical')
        warning_count = sum(1 for alert in self.active_alerts if alert['level'] == 'warning')
        info_count = sum(1 for alert in self.active_alerts if alert['level'] == 'info')
        
        return {
            'total_active': len(self.active_alerts),
            'critical_alerts': critical_count,
            'warning_alerts': warning_count,
            'info_alerts': info_count,
            'total_history': len(self.alert_history)
        }
    
    def clear_old_alerts(self, max_age_hours: int = 24):
        """Clear alerts older than specified age"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        self.alert_history = [
            alert for alert in self.alert_history
            if current_time - alert['timestamp'] <= max_age_seconds
        ]