# src/ui/web_dashboard.py
from flask import Flask, render_template, jsonify, Response
import threading
import time
import json
from datetime import datetime, timedelta
import numpy as np

class HealthDashboard:
    def __init__(self, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.health_data = []
        self.alerts = []
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/health-data')
        def get_health_data():
            """Get current health data"""
            if self.health_data:
                return jsonify(self.health_data[-1])
            return jsonify({})
        
        @self.app.route('/api/health-history')
        def get_health_history():
            """Get health data history"""
            return jsonify(self.health_data[-100:])  # Last 100 points
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get current alerts"""
            return jsonify(self.alerts)
        
        @self.app.route('/api/system-status')
        def get_system_status():
            """Get system status"""
            return jsonify({
                'status': 'running',
                'uptime': self.get_uptime(),
                'sensors_connected': self.get_sensor_status(),
                'last_update': datetime.now().isoformat()
            })
        
        @self.app.route('/stream')
        def stream():
            """Server-sent events for real-time updates"""
            def event_stream():
                while True:
                    if self.health_data:
                        latest_data = self.health_data[-1]
                        yield f"data: {json.dumps(latest_data)}\n\n"
                    time.sleep(1)
            return Response(event_stream(), mimetype="text/event-stream")
    
    def update_health_data(self, health_results: dict):
        """Update health data for dashboard"""
        timestamp = datetime.now().isoformat()
        
        data_point = {
            'timestamp': timestamp,
            'stress_score': health_results.get('stress_score', 0),
            'fatigue_level': health_results.get('fatigue_level', 0),
            'anomaly_confidence': health_results.get('anomaly_confidence', 0),
            'inference_times': health_results.get('inference_times', {}),
            'sensor_confidence': health_results.get('sensor_confidence', {})
        }
        
        self.health_data.append(data_point)
        
        # Keep only last hour of data (assuming 1Hz updates)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        self.health_data = [
            point for point in self.health_data 
            if datetime.fromisoformat(point['timestamp']) > one_hour_ago
        ]
        
        # Check for new alerts
        self._check_alerts(data_point)
    
    def _check_alerts(self, data_point: dict):
        """Check for alert conditions"""
        alerts = []
        
        if data_point['stress_score'] > 0.8:
            alerts.append({
                'type': 'high_stress',
                'level': 'warning',
                'message': f'High stress level detected: {data_point["stress_score"]:.2f}',
                'timestamp': data_point['timestamp']
            })
        
        if data_point['fatigue_level'] > 0.8:
            alerts.append({
                'type': 'high_fatigue', 
                'level': 'warning',
                'message': f'High fatigue level detected: {data_point["fatigue_level"]:.2f}',
                'timestamp': data_point['timestamp']
            })
        
        if data_point['anomaly_confidence'] > 0.9:
            alerts.append({
                'type': 'health_anomaly',
                'level': 'critical',
                'message': 'Potential health anomaly detected!',
                'timestamp': data_point['timestamp']
            })
        
        # Add new alerts
        for alert in alerts:
            if not any(a['timestamp'] == alert['timestamp'] for a in self.alerts):
                self.alerts.append(alert)
        
        # Keep only recent alerts (last 24 hours)
        one_day_ago = datetime.now() - timedelta(days=1)
        self.alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > one_day_ago
        ]
    
    def get_uptime(self) -> str:
        """Get system uptime (simplified)"""
        return "2 hours 15 minutes"  # In real implementation, calculate actual uptime
    
    def get_sensor_status(self) -> dict:
        """Get sensor connection status"""
        return {
            'camera': 'connected',
            'imu': 'connected', 
            'heart_rate': 'not_connected'
        }
    
    def start_dashboard(self):
        """Start the web dashboard"""
        print(f"🌐 Starting Health Dashboard at http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=False, threaded=True)

# HTML Template (would be in templates/index.html)
"""
<!DOCTYPE html>
<html>
<head>
    <title>Edge Health Guardian Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .dashboard { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .chart-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .alert-critical { background: #ffebee; border-left: 4px solid #f44336; }
        .alert-warning { background: #fff3e0; border-left: 4px solid #ff9800; }
        .progress-bar { height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; transition: width 0.3s ease; }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🛡️ Edge Health Guardian</h1>
            <p>Real-time Health Monitoring Dashboard</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <h3>Stress Level</h3>
                <div class="progress-bar">
                    <div class="progress-fill" id="stress-bar" style="width: 0%; background: #4caf50;"></div>
                </div>
                <div id="stress-value">0.00</div>
            </div>
            
            <div class="metric-card">
                <h3>Fatigue Level</h3>
                <div class="progress-bar">
                    <div class="progress-fill" id="fatigue-bar" style="width: 0%; background: #ff9800;"></div>
                </div>
                <div id="fatigue-value">0.00</div>
            </div>
            
            <div class="metric-card">
                <h3>Anomaly Confidence</h3>
                <div class="progress-bar">
                    <div class="progress-fill" id="anomaly-bar" style="width: 0%; background: #f44336;"></div>
                </div>
                <div id="anomaly-value">0.00</div>
            </div>
        </div>

        <div class="chart-container">
            <h3>Health Trends</h3>
            <canvas id="healthChart" width="400" height="200"></canvas>
        </div>

        <div class="chart-container">
            <h3>System Alerts</h3>
            <div id="alerts-container"></div>
        </div>
    </div>

    <script>
        // Real-time updates using Server-Sent Events
        const eventSource = new EventSource('/stream');
        
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };

        function updateDashboard(data) {
            // Update metric bars
            document.getElementById('stress-bar').style.width = (data.stress_score * 100) + '%';
            document.getElementById('stress-value').textContent = data.stress_score.toFixed(2);
            
            document.getElementById('fatigue-bar').style.width = (data.fatigue_level * 100) + '%';
            document.getElementById('fatigue-value').textContent = data.fatigue_level.toFixed(2);
            
            document.getElementById('anomaly-bar').style.width = (data.anomaly_confidence * 100) + '%';
            document.getElementById('anomaly-value').textContent = data.anomaly_confidence.toFixed(2);
        }

        // Chart initialization would go here
        // Alert display logic would go here
    </script>
</body>
</html>
"""

def main():
    """Main function to run the web dashboard"""
    dashboard = HealthDashboard()
    
    # Start dashboard in separate thread
    dashboard_thread = threading.Thread(target=dashboard.start_dashboard)
    dashboard_thread.daemon = True
    dashboard_thread.start()
    
    # Simulate some health data updates
    try:
        while True:
            # In real implementation, this would come from the health monitoring system
            simulated_data = {
                'stress_score': np.random.uniform(0, 1),
                'fatigue_level': np.random.uniform(0, 1),
                'anomaly_confidence': np.random.uniform(0, 0.3),
                'inference_times': {
                    'face': np.random.uniform(10, 30),
                    'movement': np.random.uniform(5, 15),
                    'fusion': np.random.uniform(2, 8)
                },
                'sensor_confidence': {
                    'face': np.random.uniform(0.7, 1.0),
                    'movement': np.random.uniform(0.8, 1.0),
                    'hr': np.random.uniform(0, 1.0)
                }
            }
            
            dashboard.update_health_data(simulated_data)
            time.sleep(2)  # Update every 2 seconds for demo
            
    except KeyboardInterrupt:
        print("Dashboard stopped")

if __name__ == "__main__":
    main()