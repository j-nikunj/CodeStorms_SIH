# MERGED DELIVERABLE 4: WEB API AND SERVICES LAYER
# File: web_api_services.py

"""
Web API and Services Layer for Intelligent Train Traffic Control System
Combines RESTful API, dashboard services, and web interface components
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import threading
import time
import uuid
from dataclasses import asdict

from core_data_layer import Train, Section, TrainType, TrainStatus, Constraint, create_sample_train, create_sample_section
from ai_decision_engine import RealTimeDecisionEngine
from simulation_validation_engine import SimulationValidationEngine

logger = logging.getLogger(__name__)

class TrafficControlAPI:
    """
    Comprehensive API layer providing RESTful endpoints for the traffic control system
    """
    
    def __init__(self, enable_sample_data: bool = True):
        # Core components
        self.decision_engine = RealTimeDecisionEngine(use_rl=True, use_simulation=False)
        self.sim_val_engine = SimulationValidationEngine()
        
        # API configuration
        self.api_version = "1.0"
        self.max_requests_per_minute = 100
        self.request_timeout_seconds = 30.0
        
        # Session management
        self.active_sessions = {}
        self.session_timeout_minutes = 60
        
        # Rate limiting
        self.request_counts = {}
        self.rate_limit_window = timedelta(minutes=1)
        
        # Request tracking
        self.request_history = []
        self.api_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time_ms': 0.0,
            'active_sessions': 0,
            'api_uptime_hours': 0.0
        }
        
        # Thread safety
        self._api_lock = threading.Lock()
        self._session_lock = threading.Lock()
        
        # System startup
        self.api_start_time = datetime.now()
        
        if enable_sample_data:
            self._initialize_sample_data()
        
        logger.info("Traffic Control API initialized")
    
    def _initialize_sample_data(self):
        """Initialize system with sample data"""
        try:
            # Create sample trains
            base_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
            trains = [
                Train("12951", TrainType.EXPRESS, 1, base_time, base_time + timedelta(minutes=10), 
                     "Mumbai Central", "New Delhi", 5, TrainStatus.SCHEDULED, 120.0, 400.0, 1200, ["SEC001"]),
                Train("LOCAL001", TrainType.LOCAL, 3, base_time + timedelta(minutes=15), 
                     base_time + timedelta(minutes=17), "Mumbai Central", "Virar", 3, TrainStatus.RUNNING, 80.0, 300.0, 1500, ["SEC001", "SEC002"]),
                Train("FREIGHT001", TrainType.FREIGHT, 4, base_time + timedelta(minutes=60), 
                     base_time + timedelta(minutes=65), "Borivali", "Virar", 8, TrainStatus.DELAYED, 45.0, 800.0, 0, ["SEC002"])
            ]
            
            # Create sample sections
            sections = [
                Section("SEC001", "Mumbai Central", "Borivali", 25.0, 100.0, 4, True, 0.5, 2.0, True),
                Section("SEC002", "Borivali", "Virar", 18.0, 80.0, 3, False, 1.2, 2.0, True),
                Section("SEC003", "Mumbai Central", "Churchgate", 12.0, 60.0, 6, True, 0.0, 1.5, True)
            ]
            
            # Add to system
            for section in sections:
                self.decision_engine.cp_solver.add_section(section)
                
            for train in trains:
                self.decision_engine.cp_solver.add_train(train)
            
            logger.info(f"Sample data initialized: {len(trains)} trains, {len(sections)} sections")
            
        except Exception as e:
            logger.error(f"Failed to initialize sample data: {e}")
    
    def create_session(self, user_id: str, role: str = "controller") -> Dict:
        """Create a new user session"""
        session_id = str(uuid.uuid4())
        
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'role': role,
            'login_time': datetime.now(),
            'last_activity': datetime.now(),
            'active': True
        }
        
        with self._session_lock:
            self.active_sessions[session_id] = session_data
        
        return {
            'session_id': session_id,
            'user_id': user_id,
            'role': role,
            'login_time': session_data['login_time'].isoformat(),
            'success': True
        }
    
    def validate_session(self, session_id: str) -> bool:
        """Validate if session is active and not expired"""
        with self._session_lock:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            
            # Check if session expired
            if (datetime.now() - session['last_activity']).total_seconds() > (self.session_timeout_minutes * 60):
                session['active'] = False
                return False
            
            # Update last activity
            session['last_activity'] = datetime.now()
            return session['active']
    
    def _create_api_response(self, success: bool, data: Any = None, error: str = None) -> Dict:
        """Create standardized API response"""
        response = {
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'api_version': self.api_version
        }
        
        if success:
            response['data'] = data
        else:
            response['error'] = error
        
        return response
    
    def _log_api_request(self, endpoint: str, method: str, session_id: str = None,
                        success: bool = True, response_time_ms: float = 0.0):
        """Log API request for monitoring"""
        with self._api_lock:
            self.api_metrics['total_requests'] += 1
            
            if success:
                self.api_metrics['successful_requests'] += 1
            else:
                self.api_metrics['failed_requests'] += 1
    
    # ==================== CORE API ENDPOINTS ====================
    
    def get_system_status(self, session_id: str = None) -> Dict:
        """GET /api/system/status - Get comprehensive system status"""
        start_time = datetime.now()
        
        try:
            if session_id and not self.validate_session(session_id):
                return self._create_api_response(False, error="Invalid or expired session")
            
            # Get status from decision engine
            system_status = self.decision_engine.get_system_status()
            
            # Add API-specific metrics
            with self._api_lock:
                api_uptime = (datetime.now() - self.api_start_time).total_seconds() / 3600
                self.api_metrics['api_uptime_hours'] = api_uptime
                self.api_metrics['active_sessions'] = len([s for s in self.active_sessions.values() if s['active']])
            
            system_status['api_metrics'] = self.api_metrics.copy()
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._log_api_request('/api/system/status', 'GET', session_id, True, response_time)
            
            return self._create_api_response(True, system_status)
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._log_api_request('/api/system/status', 'GET', session_id, False, response_time)
            return self._create_api_response(False, error=f"Failed to get system status: {str(e)}")
    
    def make_scheduling_decision(self, section_id: str, request_data: Dict, session_id: str = None) -> Dict:
        """POST /api/sections/{section_id}/schedule - Make scheduling decision"""
        start_time = datetime.now()
        
        try:
            if session_id and not self.validate_session(session_id):
                return self._create_api_response(False, error="Invalid or expired session")
            
            # Extract request parameters
            optimization_objective = request_data.get('optimization_objective', 'balanced')
            emergency_mode = request_data.get('emergency_mode', False)
            user_preferences = request_data.get('user_preferences', {})
            
            # Get current trains and sections from solver
            trains = self.decision_engine.cp_solver.trains
            sections = self.decision_engine.cp_solver.sections
            
            if not trains:
                return self._create_api_response(False, error=f"No trains found for section {section_id}")
            
            # Get section object
            target_section = next((s for s in sections if s.section_id == section_id), None)
            if not target_section:
                return self._create_api_response(False, error=f"Section {section_id} not found")
            
            # Make decision using decision engine
            decision = self.decision_engine.make_decision(
                section_id=section_id,
                trains=trains,
                sections=[target_section],
                emergency_mode=emergency_mode,
                optimization_objective=optimization_objective,
                user_preferences=user_preferences
            )
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._log_api_request(f'/api/sections/{section_id}/schedule', 'POST', session_id, 
                                decision.get('success', False), response_time)
            
            return self._create_api_response(decision.get('success', False), decision)
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._log_api_request(f'/api/sections/{section_id}/schedule', 'POST', session_id, False, response_time)
            return self._create_api_response(False, error=f"Scheduling decision failed: {str(e)}")
    
    def run_scenario_analysis(self, request_data: Dict, session_id: str = None) -> Dict:
        """POST /api/scenarios/analyze - Run what-if scenario analysis"""
        start_time = datetime.now()
        
        try:
            if session_id and not self.validate_session(session_id):
                return self._create_api_response(False, error="Invalid or expired session")
            
            # Get current system state
            trains = self.decision_engine.cp_solver.trains
            sections = self.decision_engine.cp_solver.sections
            
            # Run comprehensive analysis
            result = self.sim_val_engine.run_comprehensive_analysis(trains, sections)
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._log_api_request('/api/scenarios/analyze', 'POST', session_id, True, response_time)
            
            return self._create_api_response(True, result)
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._log_api_request('/api/scenarios/analyze', 'POST', session_id, False, response_time)
            return self._create_api_response(False, error=f"Scenario analysis failed: {str(e)}")
    
    def get_dashboard_data(self, session_id: str = None) -> Dict:
        """GET /api/dashboard - Get comprehensive dashboard data"""
        start_time = datetime.now()
        
        try:
            if session_id and not self.validate_session(session_id):
                return self._create_api_response(False, error="Invalid or expired session")
            
            # Get system status
            system_status = self.decision_engine.get_system_status()
            
            # Get train and section data
            trains = self.decision_engine.cp_solver.trains
            sections = self.decision_engine.cp_solver.sections
            
            # Calculate key metrics
            key_metrics = {
                'total_trains': len(trains),
                'delayed_trains': len([t for t in trains if t.delay_minutes > 0]),
                'average_delay': sum(t.delay_minutes for t in trains) / max(len(trains), 1),
                'on_time_percentage': len([t for t in trains if t.delay_minutes <= 5]) / max(len(trains), 1) * 100,
                'total_sections': len(sections),
                'average_utilization': sum(s.get_utilization() for s in sections) / max(len(sections), 1)
            }
            
            dashboard_data = {
                'system_status': system_status,
                'key_metrics': key_metrics,
                'trains': [t.to_dict() for t in trains[:10]],  # Latest 10 trains
                'sections': [s.to_dict() for s in sections],
                'timestamp': datetime.now().isoformat(),
                'refresh_interval_seconds': 30
            }
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._log_api_request('/api/dashboard', 'GET', session_id, True, response_time)
            
            return self._create_api_response(True, dashboard_data)
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self._log_api_request('/api/dashboard', 'GET', session_id, False, response_time)
            return self._create_api_response(False, error=f"Failed to get dashboard data: {str(e)}")

class DashboardService:
    """Dashboard service for web interface visualization"""
    
    def __init__(self, api: TrafficControlAPI):
        self.api = api
        
    def generate_dashboard_html(self) -> str:
        """Generate HTML dashboard"""
        dashboard_data = self.api.get_dashboard_data()
        
        if not dashboard_data['success']:
            return "<html><body><h1>Dashboard Error</h1></body></html>"
        
        data = dashboard_data['data']
        key_metrics = data['key_metrics']
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Train Control Dashboard</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 20px; 
                          border: 1px solid #ccc; border-radius: 5px; }}
                .header {{ color: #2c3e50; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
            </style>
        </head>
        <body>
            <h1 class="header">🚄 Intelligent Train Traffic Control Dashboard</h1>
            <p>Last Updated: {data['timestamp']}</p>
            
            <h2>Key Metrics</h2>
            <div class="metric">
                <h3>Total Trains</h3>
                <p style="font-size: 24px;">{key_metrics['total_trains']}</p>
            </div>
            <div class="metric">
                <h3>Delayed Trains</h3>
                <p style="font-size: 24px; color: {'red' if key_metrics['delayed_trains'] > 0 else 'green'};">
                    {key_metrics['delayed_trains']}</p>
            </div>
            <div class="metric">
                <h3>On-Time Performance</h3>
                <p style="font-size: 24px;">{key_metrics['on_time_percentage']:.1f}%</p>
            </div>
            <div class="metric">
                <h3>Average Delay</h3>
                <p style="font-size: 24px;">{key_metrics['average_delay']:.1f} min</p>
            </div>
            
            <h2>System Status</h2>
            <p>Health: <span class="{'success' if data['system_status']['system_health'] == 'HEALTHY' else 'warning'}">
                {data['system_status']['system_health']}</span></p>
            <p>Total Decisions: {data['system_status']['performance_metrics']['total_decisions']}</p>
            
            <h2>Active Trains</h2>
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr>
                    <th>Train ID</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Delay (min)</th>
                    <th>Current Location</th>
                    <th>Destination</th>
                </tr>
        """
        
        for train_data in data.get('trains', [])[:5]:  # Show first 5 trains
            html += f"""
                <tr>
                    <td>{train_data['train_id']}</td>
                    <td>{train_data['train_type']}</td>
                    <td>{train_data['status']}</td>
                    <td style="color: {'red' if train_data['delay_minutes'] > 0 else 'green'};">
                        {train_data['delay_minutes']}</td>
                    <td>{train_data['current_location']}</td>
                    <td>{train_data['destination']}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html

class WebServices:
    """Combined web services for the traffic control system"""
    
    def __init__(self):
        self.api = TrafficControlAPI(enable_sample_data=True)
        self.dashboard = DashboardService(self.api)
        
        logger.info("Web Services initialized")
    
    def start_simple_server(self, port: int = 8000):
        """Start a simple HTTP server"""
        import http.server
        import socketserver
        from urllib.parse import urlparse, parse_qs
        
        class RequestHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, web_services=None, **kwargs):
                self.web_services = web_services
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                
                if path == '/api/status':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = self.web_services.api.get_system_status()
                    self.wfile.write(json.dumps(response).encode())
                
                elif path == '/api/dashboard':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = self.web_services.api.get_dashboard_data()
                    self.wfile.write(json.dumps(response).encode())
                
                elif path == '/dashboard':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    html = self.web_services.dashboard.generate_dashboard_html()
                    self.wfile.write(html.encode())
                
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b'Not Found')
        
        handler = lambda *args, **kwargs: RequestHandler(*args, web_services=self, **kwargs)
        
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"🌐 Web server running at http://localhost:{port}")
            print(f"📊 Dashboard: http://localhost:{port}/dashboard")
            print(f"🔧 API Status: http://localhost:{port}/api/status")
            httpd.serve_forever()

if __name__ == "__main__":
    # Test the web services
    print("Testing Web API and Services...")
    
    # Create web services
    web_services = WebServices()
    
    # Test API endpoints
    status_response = web_services.api.get_system_status()
    print(f"System status success: {status_response['success']}")
    
    # Test session creation
    session_response = web_services.api.create_session("test_user", "controller")
    session_id = session_response['session_id']
    print(f"Session created: {session_id}")
    
    # Test dashboard data
    dashboard_response = web_services.api.get_dashboard_data(session_id)
    print(f"Dashboard data success: {dashboard_response['success']}")
    
    print("Web API and Services test completed!")
    print("To start the web server, run: web_services.start_simple_server()")
