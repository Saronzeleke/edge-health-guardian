# deployment/raspberry_pi_setup.py
import subprocess
import sys
import os

def setup_raspberry_pi():
    """Optimize Raspberry Pi for Edge Health Guardian"""
    
    print("🔧 Optimizing Raspberry Pi for health monitoring...")
    
    # Enable camera interface
    subprocess.run(['sudo', 'raspi-config', 'nonint', 'do_camera', '0'])
    
    # Increase GPU memory for camera processing
    with open('/boot/config.txt', 'a') as f:
        f.write('\n# Edge Health Guardian Optimizations\n')
        f.write('gpu_mem=128\n')
        f.write('start_x=1\n')
        f.write('disable_camera_led=1\n')
    
    # Set CPU governor to performance for consistent inference times
    subprocess.run(['sudo', 'cpufreq-set', '-g', 'performance'])
    
    # Enable hardware acceleration
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OMP_NUM_THREADS'] = '4'
    
    print("✅ Raspberry Pi optimization complete!")
    print("📍 Please reboot for changes to take effect")

if __name__ == "__main__":
    setup_raspberry_pi()