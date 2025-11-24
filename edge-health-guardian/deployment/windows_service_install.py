# deployment/windows_service_install.py
import os
import sys
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket

class EdgeHealthGuardianService(win32serviceutil.ServiceFramework):
    """Windows Service for Edge Health Guardian"""
    
    _svc_name_ = "EdgeHealthGuardian"
    _svc_display_name_ = "Edge Health Guardian Health Monitor"
    _svc_description_ = "On-device AI health monitoring system"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
    
    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
    
    def SvcDoRun(self):
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        self.main()
    
    def main(self):
        # Add the src directory to Python path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(script_dir, '..', 'src')
        sys.path.insert(0, src_dir)
        
        # Activate virtual environment
        venv_path = os.path.join(script_dir, '..', 'health-guardian-env')
        if os.path.exists(venv_path):
            activate_script = os.path.join(venv_path, 'Scripts', 'activate_this.py')
            with open(activate_script) as f:
                exec(f.read(), {'__file__': activate_script})
        
        # Import and run the main application
        from main import EdgeHealthGuardian
        
        health_guardian = EdgeHealthGuardian()
        health_guardian.start_monitoring()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(EdgeHealthGuardianService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(EdgeHealthGuardianService)