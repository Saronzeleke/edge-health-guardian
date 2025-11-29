# src/utils/alert_system.py
import pygame
import threading
import time
from typing import Dict, List
import logging
import numpy as np

class AlertSystem:
    """Multi-modal alert system for health notifications"""
    
    def __init__(self):
        self.active_alerts = []
        self.alert_history = []
        self.audio_enabled = True
        self.visual_enabled = True
        self.haptic_enabled = False  # For mobile devices
        
        # Initialize pygame for audio
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.audio_available = True
            logging.info("Audio system initialized successfully")
        except Exception as e:
            self.audio_available = False
            logging.warning(f"Audio alerts not available: {e}")
    
    def trigger_alert(self, alert_data: Dict):
        """Trigger a health alert"""
        alert_id = f"{alert_data['type']}_{int(time.time())}"
        
        alert = {
            'id': alert_id,
            'type': alert_data['type'],
            'level': alert_data['level'],
            'value': alert_data.get('value', 0),
            'message': alert_data['message'],
            'timestamp': time.time(),
            'acknowledged': False
        }
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Trigger alert modalities based on level
        if alert_data['level'] == 'critical':
            self._trigger_critical_alert(alert)
        elif alert_data['level'] == 'high':
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
        """Play alert sound based on type using pygame"""
        try:
            if not self.audio_available:
                return
                
            if alert_type == 'critical':
                # Triple beep for critical alerts
                self._generate_beep_sound(880, 400)  # High A
                time.sleep(0.15)
                self._generate_beep_sound(880, 400)
                time.sleep(0.15)
                self._generate_beep_sound(880, 400)
            elif alert_type == 'warning':
                # Double beep for warnings
                self._generate_beep_sound(660, 300)  # Middle E
                time.sleep(0.2)
                self._generate_beep_sound(660, 300)
            else:
                # Single beep for info
                self._generate_beep_sound(440, 200)  # Low A
                
        except Exception as e:
            logging.error(f"Audio alert failed: {e}")
            self.audio_available = False
    
    def _generate_beep_sound(self, frequency: int, duration: int):
        """Generate and play a beep sound using pygame"""
        try:
            sample_rate = 22050
            n_samples = int(round(duration * 0.001 * sample_rate))
            
            # Generate sine wave
            buf = np.zeros((n_samples, 2), dtype=np.int16)
            max_amplitude = 32767  # Max for 16-bit audio
            
            for i in range(n_samples):
                t = float(i) / sample_rate  # time in seconds
                # Sine wave for the given frequency
                sample = max_amplitude * 0.5 * np.sin(2 * np.pi * frequency * t)
                buf[i][0] = int(sample)  # left channel
                buf[i][1] = int(sample)  # right channel
            
            # Create pygame sound and play
            sound = pygame.sndarray.make_sound(buf)
            sound.play()
            
            # Wait for sound to finish playing
            pygame.time.wait(duration)
            
        except Exception as e:
            logging.error(f"Beep generation failed: {e}")
            # Fallback to system beep
            print("\a")  # System bell
    
    def _show_visual_alert(self, alert: Dict, level: str):
        """Show visual alert in console with colors"""
        try:
            colors = {
                'critical': '\033[91m',  # Red
                'warning': '\033[93m',   # Yellow  
                'info': '\033[94m',      # Blue
                'high': '\033[93m'       # Yellow for high level
            }
            
            reset_color = '\033[0m'
            
            print(f"\n{colors[level]}{'='*60}{reset_color}")
            print(f"{colors[level]}⚠️  HEALTH ALERT ⚠️{reset_color}")
            print(f"{colors[level]}Type: {alert['type'].upper()}{reset_color}")
            print(f"{colors[level]}Level: {alert['level'].upper()}{reset_color}")
            print(f"{colors[level]}Value: {alert.get('value', 0):.3f}{reset_color}")
            print(f"{colors[level]}Message: {alert['message']}{reset_color}")
            print(f"{colors[level]}Time: {time.ctime(alert['timestamp'])}{reset_color}")
            print(f"{colors[level]}{'='*60}{reset_color}\n")
            
        except Exception as e:
            # Fallback without colors if encoding issues
            print(f"\n{'='*60}")
            print(f"HEALTH ALERT - {alert['level'].upper()}")
            print(f"Type: {alert['type']}")
            print(f"Message: {alert['message']}")
            print(f"Time: {time.ctime(alert['timestamp'])}")
            print(f"{'='*60}\n")
    
    def _repeat_alert(self, alert: Dict):
        """Repeat critical alert until acknowledged"""
        repeat_count = 0
        max_repeats = 3  # Only repeat 3 times to avoid spam
        
        while (not alert['acknowledged'] and 
               alert in self.active_alerts and 
               repeat_count < max_repeats):
            time.sleep(10)  # Repeat every 10 seconds
            if not alert['acknowledged']:
                if self.visual_enabled:
                    self._show_visual_alert(alert, 'critical')
                repeat_count += 1
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge and deactivate an alert"""
        for alert in self.active_alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                self.active_alerts.remove(alert)
                logging.info(f"Alert acknowledged: {alert_id}")
                break
    
    def acknowledge_all_alerts(self):
        """Acknowledge all active alerts"""
        for alert in self.active_alerts:
            alert['acknowledged'] = True
        self.active_alerts.clear()
        logging.info("All alerts acknowledged")
    
    def get_active_alerts(self) -> List[Dict]:
        """Get list of active alerts"""
        return self.active_alerts.copy()
    
    def get_alert_summary(self) -> Dict:
        """Get alert statistics"""
        critical_count = sum(1 for alert in self.active_alerts if alert['level'] == 'critical')
        warning_count = sum(1 for alert in self.active_alerts if alert['level'] in ['warning', 'high'])
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
    
    def enable_audio(self, enabled: bool = True):
        """Enable or disable audio alerts"""
        self.audio_enabled = enabled
        logging.info(f"Audio alerts {'enabled' if enabled else 'disabled'}")
    
    def enable_visual(self, enabled: bool = True):
        """Enable or disable visual alerts"""
        self.visual_enabled = enabled
        logging.info(f"Visual alerts {'enabled' if enabled else 'disabled'}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            pygame.mixer.quit()
        except:
            pass