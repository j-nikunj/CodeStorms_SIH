#!/usr/bin/env python3
# MERGED DELIVERABLE 7: MAIN APPLICATION ORCHESTRATOR
# File: main_application.py

"""
Main Application Orchestrator for Intelligent Train Traffic Control System
Unified entry point that orchestrates all system components

This is the primary entry point for the complete Intelligent Train Traffic Control System.
It integrates all components and provides multiple operational modes.
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import all major system components
from core_data_layer import DataManager, Train, Section, create_sample_train, create_sample_section
from ai_decision_engine import RealTimeDecisionEngine
from simulation_validation_engine import SimulationValidationEngine
from web_api_services import WebServices
from operations_utilities import OperationsOrchestrator, ConfigurationManager
from demonstration_examples import ScenarioRunner, InteractiveDemoRunner, BenchmarkRunner

# Configure root logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_control_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntelligentTrainControlSystem:
    """
    Main orchestrator class for the Intelligent Train Traffic Control System
    Coordinates all subsystems and provides unified interface
    """
    
    def __init__(self, config_file: str = "config.ini"):
        """Initialize the complete system"""
        print("🚄 Initializing Intelligent Train Traffic Control System...")
        
        # System information
        self.system_info = {
            'name': 'Intelligent Train Traffic Control System',
            'version': '1.0.0',
            'build_date': '2024-01-01',
            'description': 'AI-powered railway traffic management and optimization system'
        }
        
        self.startup_time = datetime.now()
        self.config_file = config_file
        self.system_running = False
        self.shutdown_requested = False
        
        # Initialize configuration manager
        self.config_manager = ConfigurationManager(config_file)
        
        # Initialize all major components
        self._initialize_components()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Intelligent Train Traffic Control System initialized")
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            print("  🔧 Initializing core components...")
            
            # Core data management
            self.data_manager = DataManager()
            print("    ✅ Data Manager initialized")
            
            # AI decision engine
            use_rl = self.config_manager.get("AI", "use_reinforcement_learning", "true").lower() == "true"
            self.decision_engine = RealTimeDecisionEngine(use_rl=use_rl, use_simulation=False)
            print("    ✅ AI Decision Engine initialized")
            
            # Simulation and validation
            self.sim_val_engine = SimulationValidationEngine()
            print("    ✅ Simulation & Validation Engine initialized")
            
            # Web services and API
            self.web_services = WebServices()
            print("    ✅ Web Services initialized")
            
            # Operations and utilities
            self.operations = OperationsOrchestrator()
            print("    ✅ Operations Orchestrator initialized")
            
            # Demonstration runner
            self.demo_runner = ScenarioRunner()
            print("    ✅ Demonstration Runner initialized")
            
            print("  ✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise RuntimeError(f"System initialization failed: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        signal_names = {signal.SIGINT: "SIGINT", signal.SIGTERM: "SIGTERM"}
        signal_name = signal_names.get(signum, f"Signal {signum}")
        
        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        print(f"\n🛑 Received {signal_name}, shutting down gracefully...")
        
        self.shutdown_requested = True
        self.stop()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            uptime = (datetime.now() - self.startup_time).total_seconds()
            
            status = {
                'system_info': self.system_info,
                'startup_time': self.startup_time.isoformat(),
                'uptime_seconds': uptime,
                'uptime_formatted': self._format_duration(uptime),
                'system_running': self.system_running,
                'components': {}
            }
            
            # Get status from each component
            if hasattr(self, 'decision_engine'):
                status['components']['decision_engine'] = self.decision_engine.get_system_status()
            
            if hasattr(self, 'operations'):
                status['components']['operations'] = self.operations.run_full_system_check()
            
            if hasattr(self, 'web_services'):
                api_status = self.web_services.api.get_system_status()
                status['components']['web_api'] = api_status.get('data', {})
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                'error': f"Failed to get system status: {e}",
                'timestamp': datetime.now().isoformat()
            }
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def start_production_mode(self):
        """Start the system in production mode"""
        print("🚀 Starting system in PRODUCTION mode...")
        
        try:
            # Pre-flight checks
            print("  🔍 Performing pre-flight checks...")
            health_check = self.operations.run_full_system_check()
            
            if health_check['overall_status'] != 'HEALTHY':
                print("  ⚠️  Warning: System health check shows issues")
                logger.warning(f"System health issues detected: {health_check}")
                
                response = input("  Continue anyway? (y/N): ").strip().lower()
                if response != 'y':
                    print("  ❌ Production start aborted")
                    return False
            else:
                print("    ✅ Pre-flight checks passed")
            
            # Initialize with sample data if none exists
            if not self.data_manager.get_all_trains():
                print("  📊 Loading initial data...")
                self._load_initial_data()
            
            # Start monitoring
            print("  📊 Starting system monitoring...")
            monitoring_interval = int(self.config_manager.get("SYSTEM", "monitoring_interval", "60"))
            # Note: In a real system, you would start monitoring here
            
            # Start web API server in background thread
            print("  🌐 Starting web API server...")
            api_port = int(self.config_manager.get("SYSTEM", "api_port", "8000"))
            
            # Start server in separate thread
            server_thread = threading.Thread(
                target=self._start_web_server, 
                args=(api_port,),
                daemon=True
            )
            server_thread.start()
            
            self.system_running = True
            
            print(f"  ✅ System running in production mode")
            print(f"     📊 Dashboard: http://localhost:{api_port}/dashboard")
            print(f"     🔧 API: http://localhost:{api_port}/api/status")
            print(f"     📝 Logs: {os.path.abspath('traffic_control_system.log')}")
            print()
            print("  Press Ctrl+C to stop the system gracefully")
            
            # Main production loop
            self._run_production_loop()
            
            return True
            
        except Exception as e:
            logger.error(f"Production mode startup failed: {e}")
            print(f"  ❌ Production mode failed: {e}")
            return False
    
    def _start_web_server(self, port: int):
        """Start web server in background"""
        try:
            self.web_services.start_simple_server(port)
        except Exception as e:
            logger.error(f"Web server failed: {e}")
            print(f"  ⚠️  Web server error: {e}")
    
    def _run_production_loop(self):
        """Main production system loop"""
        try:
            while self.system_running and not self.shutdown_requested:
                # System maintenance checks every 10 minutes
                time.sleep(600)  # 10 minutes
                
                if not self.shutdown_requested:
                    try:
                        # Perform routine maintenance
                        self.operations.perform_maintenance()
                        
                        # Log system status
                        status = self.get_system_status()
                        logger.info(f"System status check: uptime={status.get('uptime_formatted', 'unknown')}")
                        
                    except Exception as e:
                        logger.error(f"Production loop maintenance failed: {e}")
                
        except KeyboardInterrupt:
            logger.info("Production loop interrupted by user")
        except Exception as e:
            logger.error(f"Production loop failed: {e}")
            print(f"  ❌ Production loop error: {e}")
    
    def start_development_mode(self):
        """Start the system in development mode"""
        print("🛠️  Starting system in DEVELOPMENT mode...")
        
        try:
            # Load sample data
            print("  📊 Loading sample data...")
            self._load_initial_data()
            
            # Show system status
            status = self.get_system_status()
            print(f"    ✅ System initialized with sample data")
            print(f"    🔧 Components: {len(status.get('components', {}))}")
            
            # Start development interface
            print("  🖥️  Starting development interface...")
            self._run_development_interface()
            
            return True
            
        except Exception as e:
            logger.error(f"Development mode startup failed: {e}")
            print(f"  ❌ Development mode failed: {e}")
            return False
    
    def _run_development_interface(self):
        """Interactive development interface"""
        print("\n" + "="*60)
        print("🛠️  DEVELOPMENT MODE INTERFACE")
        print("="*60)
        
        while not self.shutdown_requested:
            print("\nAvailable actions:")
            print("  1. Show system status")
            print("  2. Run AI decision test")
            print("  3. Run simulation analysis")
            print("  4. Start web server")
            print("  5. Run demonstration")
            print("  6. Performance benchmark")
            print("  7. System maintenance")
            print("  0. Exit development mode")
            
            try:
                choice = input("\nSelect action (0-7): ").strip()
                
                if choice == '0':
                    print("Exiting development mode...")
                    break
                
                elif choice == '1':
                    self._show_system_status()
                
                elif choice == '2':
                    self._run_ai_decision_test()
                
                elif choice == '3':
                    self._run_simulation_analysis()
                
                elif choice == '4':
                    self._start_dev_web_server()
                
                elif choice == '5':
                    self._run_development_demo()
                
                elif choice == '6':
                    self._run_performance_benchmark()
                
                elif choice == '7':
                    self._run_system_maintenance()
                
                else:
                    print("Invalid choice. Please select 0-7.")
                
                if choice != '0':
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\nExiting development mode...")
                break
            except Exception as e:
                print(f"Error: {e}")
                input("Press Enter to continue...")
    
    def _show_system_status(self):
        """Show detailed system status"""
        print("\n📊 SYSTEM STATUS")
        print("-" * 40)
        
        status = self.get_system_status()
        
        print(f"System: {status['system_info']['name']}")
        print(f"Version: {status['system_info']['version']}")
        print(f"Uptime: {status['uptime_formatted']}")
        print(f"Running: {status['system_running']}")
        
        if 'components' in status:
            print(f"\nComponents ({len(status['components'])}):")
            for comp_name, comp_status in status['components'].items():
                health = comp_status.get('system_health', comp_status.get('overall_status', 'UNKNOWN'))
                print(f"  {comp_name}: {health}")
    
    def _run_ai_decision_test(self):
        """Run AI decision engine test"""
        print("\n🧠 AI DECISION ENGINE TEST")
        print("-" * 40)
        
        try:
            # Get current system data
            trains = self.data_manager.get_all_trains()
            sections = self.data_manager.get_all_sections()
            
            if not trains or not sections:
                print("No data available. Loading sample data...")
                self._load_initial_data()
                trains = self.data_manager.get_all_trains()
                sections = self.data_manager.get_all_sections()
            
            print(f"Testing with {len(trains)} trains and {len(sections)} sections")
            
            # Test decision for first section
            if sections:
                test_section = sections[0]
                print(f"Making decision for section: {test_section.section_id}")
                
                decision = self.decision_engine.make_decision(
                    section_id=test_section.section_id,
                    trains=trains,
                    sections=[test_section],
                    optimization_objective='balanced'
                )
                
                print(f"Decision result:")
                print(f"  Success: {decision.get('success', False)}")
                print(f"  Confidence: {decision.get('confidence', 0.0):.2f}")
                print(f"  Emergency mode: {decision.get('emergency_mode', False)}")
                if 'alternatives' in decision:
                    print(f"  Alternatives: {len(decision['alternatives'])}")
        
        except Exception as e:
            print(f"AI decision test failed: {e}")
    
    def _run_simulation_analysis(self):
        """Run simulation and analysis"""
        print("\n⚡ SIMULATION ANALYSIS")
        print("-" * 40)
        
        try:
            trains = self.data_manager.get_all_trains()
            sections = self.data_manager.get_all_sections()
            
            if not trains or not sections:
                print("Loading sample data for analysis...")
                self._load_initial_data()
                trains = self.data_manager.get_all_trains()
                sections = self.data_manager.get_all_sections()
            
            print(f"Running analysis with {len(trains)} trains and {len(sections)} sections...")
            
            result = self.sim_val_engine.run_comprehensive_analysis(trains, sections)
            
            print("Analysis results:")
            print(f"  Simulation success: {result.get('simulation_success', False)}")
            print(f"  Validation success: {result.get('validation_success', False)}")
            
            if 'recommendations' in result:
                print(f"  Recommendations: {len(result['recommendations'])}")
                for i, rec in enumerate(result['recommendations'][:3], 1):
                    print(f"    {i}. {rec.get('description', 'No description')}")
        
        except Exception as e:
            print(f"Simulation analysis failed: {e}")
    
    def _start_dev_web_server(self):
        """Start web server for development"""
        print("\n🌐 STARTING WEB SERVER")
        print("-" * 40)
        
        port = int(self.config_manager.get("SYSTEM", "api_port", "8000"))
        print(f"Starting web server on port {port}...")
        print("Press Ctrl+C to stop the server")
        print(f"Dashboard: http://localhost:{port}/dashboard")
        print(f"API Status: http://localhost:{port}/api/status")
        
        try:
            self.web_services.start_simple_server(port)
        except KeyboardInterrupt:
            print("\nWeb server stopped by user")
        except Exception as e:
            print(f"Web server error: {e}")
    
    def _run_development_demo(self):
        """Run demonstration in development mode"""
        print("\n🎬 RUNNING DEMONSTRATION")
        print("-" * 40)
        
        try:
            result = self.demo_runner.run_basic_scheduling_demo()
            print(f"Demo completed: {result['status']}")
            if 'duration_seconds' in result:
                print(f"Duration: {result['duration_seconds']:.1f} seconds")
        
        except Exception as e:
            print(f"Demonstration failed: {e}")
    
    def _run_performance_benchmark(self):
        """Run performance benchmark"""
        print("\n🏁 PERFORMANCE BENCHMARK")
        print("-" * 40)
        
        try:
            benchmark_runner = BenchmarkRunner()
            result = benchmark_runner.run_performance_benchmark(20, 5)  # Small scale for dev
            
            if result['status'] == 'COMPLETED' and 'performance_metrics' in result:
                metrics = result['performance_metrics']
                print("Benchmark results:")
                print(f"  Average response time: {metrics['avg_decision_time_ms']:.2f}ms")
                print(f"  Success rate: {metrics['success_rate_percent']:.1f}%")
                print(f"  Throughput: {metrics['throughput_decisions_per_second']:.2f} decisions/sec")
        
        except Exception as e:
            print(f"Performance benchmark failed: {e}")
    
    def _run_system_maintenance(self):
        """Run system maintenance"""
        print("\n🔧 SYSTEM MAINTENANCE")
        print("-" * 40)
        
        try:
            success = self.operations.perform_maintenance()
            if success:
                print("✅ Maintenance completed successfully")
            else:
                print("⚠️  Maintenance completed with issues")
        
        except Exception as e:
            print(f"System maintenance failed: {e}")
    
    def _load_initial_data(self):
        """Load initial sample data"""
        try:
            # Create sample data
            base_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
            
            # Sample trains
            sample_trains = [
                create_sample_train("SAMPLE001", base_time),
                create_sample_train("SAMPLE002", base_time + timedelta(minutes=15)),
                create_sample_train("SAMPLE003", base_time + timedelta(minutes=30))
            ]
            
            # Sample sections
            sample_sections = [
                create_sample_section("SAMPLE_SEC001"),
                create_sample_section("SAMPLE_SEC002")
            ]
            
            # Add to data manager
            for train in sample_trains:
                self.data_manager.add_train(train)
            
            for section in sample_sections:
                self.data_manager.add_section(section)
            
            # Also add to decision engine
            for section in sample_sections:
                self.decision_engine.cp_solver.add_section(section)
            
            for train in sample_trains:
                self.decision_engine.cp_solver.add_train(train)
            
            logger.info(f"Loaded {len(sample_trains)} trains and {len(sample_sections)} sections")
            
        except Exception as e:
            logger.error(f"Failed to load initial data: {e}")
            raise
    
    def start_demo_mode(self):
        """Start the system in demonstration mode"""
        print("🎬 Starting system in DEMONSTRATION mode...")
        
        try:
            # Initialize with sample data
            self._load_initial_data()
            
            # Start interactive demo
            interactive_demo = InteractiveDemoRunner()
            interactive_demo.run_interactive()
            
            return True
            
        except Exception as e:
            logger.error(f"Demo mode startup failed: {e}")
            print(f"  ❌ Demo mode failed: {e}")
            return False
    
    def stop(self):
        """Stop the system gracefully"""
        print("🛑 Stopping system...")
        logger.info("System shutdown initiated")
        
        try:
            self.system_running = False
            
            # Stop any background services
            # Note: In a real system, you would clean up resources here
            
            # Save any persistent data
            if hasattr(self, 'data_manager'):
                self.data_manager.save_data()
            
            print("  ✅ System stopped gracefully")
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")
            print(f"  ⚠️  Shutdown error: {e}")

def create_argument_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Intelligent Train Traffic Control System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_application.py --mode production
  python main_application.py --mode development
  python main_application.py --mode demo
  python main_application.py --status
  
For more information, see the documentation.
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['production', 'development', 'demo'],
        default='development',
        help='System operation mode (default: development)'
    )
    
    parser.add_argument(
        '--config',
        default='config.ini',
        help='Configuration file path (default: config.ini)'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show system status and exit'
    )
    
    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version information and exit'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run system validation and exit'
    )
    
    return parser

def show_welcome_banner():
    """Show welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           🚄 INTELLIGENT TRAIN TRAFFIC CONTROL SYSTEM                       ║
║                                                                              ║
║  AI-powered railway traffic management and optimization system               ║
║                                                                              ║
║  Features:                                                                   ║
║  • Real-time train scheduling and conflict resolution                       ║
║  • AI-powered decision making with reinforcement learning                   ║
║  • Constraint programming optimization                                       ║
║  • Traffic simulation and scenario analysis                                 ║
║  • Web-based dashboard and REST API                                         ║
║  • Comprehensive monitoring and maintenance tools                           ║
║                                                                              ║
║  Version: 1.0.0                                                             ║
║  Build Date: 2024-01-01                                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main entry point for the application"""
    
    # Show welcome banner
    show_welcome_banner()
    
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Handle version request
    if args.version:
        print("Intelligent Train Traffic Control System v1.0.0")
        print("Build Date: 2024-01-01")
        return 0
    
    try:
        # Initialize the system
        system = IntelligentTrainControlSystem(config_file=args.config)
        
        # Handle status request
        if args.status:
            status = system.get_system_status()
            print("\n📊 SYSTEM STATUS")
            print("="*50)
            print(json.dumps(status, indent=2, default=str))
            return 0
        
        # Handle validation request
        if args.validate:
            print("\n✅ SYSTEM VALIDATION")
            print("="*50)
            health_check = system.operations.run_full_system_check()
            print(f"Overall Status: {health_check['overall_status']}")
            
            for component, comp_status in health_check.get('components', {}).items():
                status_text = comp_status.get('status', comp_status.get('overall_status', 'UNKNOWN'))
                print(f"  {component}: {status_text}")
            
            return 0 if health_check['overall_status'] == 'HEALTHY' else 1
        
        # Start system in requested mode
        if args.mode == 'production':
            success = system.start_production_mode()
        elif args.mode == 'development':
            success = system.start_development_mode()
        elif args.mode == 'demo':
            success = system.start_demo_mode()
        else:
            print(f"Unknown mode: {args.mode}")
            return 1
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n🛑 System interrupted by user")
        return 130  # Standard Unix exit code for Ctrl+C
    
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        print(f"\n❌ System startup failed: {e}")
        return 1
    
    finally:
        # Cleanup
        try:
            if 'system' in locals():
                system.stop()
        except:
            pass

if __name__ == "__main__":
    exit(main())
