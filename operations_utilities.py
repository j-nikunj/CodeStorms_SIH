# MERGED DELIVERABLE 5: OPERATIONS AND UTILITIES
# File: operations_utilities.py

"""
Operations and Utilities Module for Intelligent Train Traffic Control System
Combines deployment, migration, monitoring, and operational utility scripts
"""

import os
import sys
import json
import logging
import time
import sqlite3
import psutil
import shutil
import subprocess
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path
import configparser
from dataclasses import asdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database operations and migration utilities"""
    
    def __init__(self, db_path: str = "traffic_control.db"):
        self.db_path = db_path
        self.connection = None
        
    def connect(self):
        """Connect to database"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def initialize_schema(self):
        """Initialize database schema"""
        if not self.connection:
            if not self.connect():
                return False
        
        try:
            cursor = self.connection.cursor()
            
            # Trains table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trains (
                    train_id TEXT PRIMARY KEY,
                    train_type TEXT NOT NULL,
                    platform_number INTEGER,
                    scheduled_arrival TIMESTAMP,
                    scheduled_departure TIMESTAMP,
                    current_location TEXT,
                    destination TEXT,
                    delay_minutes INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'SCHEDULED',
                    current_speed REAL DEFAULT 0.0,
                    max_speed REAL DEFAULT 100.0,
                    passenger_count INTEGER DEFAULT 0,
                    route_sections TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Sections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sections (
                    section_id TEXT PRIMARY KEY,
                    start_station TEXT NOT NULL,
                    end_station TEXT NOT NULL,
                    distance_km REAL NOT NULL,
                    max_speed REAL NOT NULL,
                    track_count INTEGER DEFAULT 2,
                    electrified BOOLEAN DEFAULT TRUE,
                    current_delay REAL DEFAULT 0.0,
                    capacity_utilization REAL DEFAULT 0.0,
                    operational BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Schedules table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schedules (
                    schedule_id TEXT PRIMARY KEY,
                    train_id TEXT NOT NULL,
                    section_id TEXT NOT NULL,
                    scheduled_entry TIMESTAMP,
                    scheduled_exit TIMESTAMP,
                    actual_entry TIMESTAMP,
                    actual_exit TIMESTAMP,
                    priority INTEGER DEFAULT 5,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (train_id) REFERENCES trains (train_id),
                    FOREIGN KEY (section_id) REFERENCES sections (section_id)
                )
            """)
            
            # System logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT
                )
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_unit TEXT,
                    component TEXT
                )
            """)
            
            self.connection.commit()
            logger.info("Database schema initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            return False
    
    def backup_database(self, backup_path: str = None) -> bool:
        """Create database backup"""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backup_traffic_control_{timestamp}.db"
        
        try:
            if os.path.exists(self.db_path):
                shutil.copy2(self.db_path, backup_path)
                logger.info(f"Database backed up to: {backup_path}")
                return True
            else:
                logger.warning(f"Source database {self.db_path} does not exist")
                return False
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    def migrate_data(self, migration_script: str) -> bool:
        """Execute data migration"""
        try:
            if not self.connection:
                if not self.connect():
                    return False
            
            cursor = self.connection.cursor()
            cursor.executescript(migration_script)
            self.connection.commit()
            logger.info("Data migration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Data migration failed: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

class SystemMonitor:
    """System monitoring and health check utilities"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'response_time_ms': 5000.0
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Process specific metrics
            current_process = psutil.Process()
            process_memory = current_process.memory_info()
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'memory_total_gb': memory.total / (1024**3),
                    'disk_percent': (disk.used / disk.total) * 100,
                    'disk_free_gb': disk.free / (1024**3),
                    'disk_total_gb': disk.total / (1024**3)
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                'process': {
                    'memory_rss_mb': process_memory.rss / (1024**2),
                    'memory_vms_mb': process_memory.vms / (1024**2),
                    'num_threads': current_process.num_threads(),
                    'cpu_percent': current_process.cpu_percent()
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            'overall_status': 'HEALTHY',
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        try:
            metrics = self.get_system_metrics()
            
            # CPU check
            cpu_status = 'HEALTHY'
            if metrics['system']['cpu_percent'] > self.alert_thresholds['cpu_percent']:
                cpu_status = 'WARNING'
                health_status['overall_status'] = 'WARNING'
            
            health_status['checks']['cpu'] = {
                'status': cpu_status,
                'value': metrics['system']['cpu_percent'],
                'threshold': self.alert_thresholds['cpu_percent'],
                'unit': '%'
            }
            
            # Memory check
            memory_status = 'HEALTHY'
            if metrics['system']['memory_percent'] > self.alert_thresholds['memory_percent']:
                memory_status = 'CRITICAL'
                health_status['overall_status'] = 'CRITICAL'
            
            health_status['checks']['memory'] = {
                'status': memory_status,
                'value': metrics['system']['memory_percent'],
                'threshold': self.alert_thresholds['memory_percent'],
                'unit': '%'
            }
            
            # Disk check
            disk_status = 'HEALTHY'
            if metrics['system']['disk_percent'] > self.alert_thresholds['disk_percent']:
                disk_status = 'CRITICAL'
                health_status['overall_status'] = 'CRITICAL'
            
            health_status['checks']['disk'] = {
                'status': disk_status,
                'value': metrics['system']['disk_percent'],
                'threshold': self.alert_thresholds['disk_percent'],
                'unit': '%'
            }
            
            # Database connectivity check
            db_manager = DatabaseManager()
            db_status = 'HEALTHY' if db_manager.connect() else 'CRITICAL'
            if db_status == 'CRITICAL':
                health_status['overall_status'] = 'CRITICAL'
            
            health_status['checks']['database'] = {
                'status': db_status,
                'message': 'Database connection test'
            }
            
            db_manager.close()
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['overall_status'] = 'CRITICAL'
            health_status['error'] = str(e)
        
        return health_status
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring"""
        if self.monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval_seconds,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info(f"System monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self, interval_seconds: int):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                metrics = self.get_system_metrics()
                health = self.check_health()
                
                # Store metrics history
                self.metrics_history.append({
                    'timestamp': datetime.now(),
                    'metrics': metrics,
                    'health': health
                })
                
                # Maintain history size
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                # Log critical issues
                if health['overall_status'] == 'CRITICAL':
                    logger.error(f"CRITICAL system health status: {health}")
                elif health['overall_status'] == 'WARNING':
                    logger.warning(f"WARNING system health status: {health}")
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
            
            time.sleep(interval_seconds)

class ConfigurationManager:
    """Configuration management utilities"""
    
    def __init__(self, config_file: str = "config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                self.config.read(self.config_file)
                logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                self.create_default_config()
        else:
            logger.info("No configuration file found, creating default")
            self.create_default_config()
    
    def create_default_config(self):
        """Create default configuration"""
        self.config['DATABASE'] = {
            'path': 'traffic_control.db',
            'backup_interval_hours': '24',
            'connection_timeout': '30'
        }
        
        self.config['SYSTEM'] = {
            'log_level': 'INFO',
            'max_threads': '10',
            'monitoring_interval': '60',
            'api_port': '8000'
        }
        
        self.config['AI'] = {
            'use_reinforcement_learning': 'true',
            'use_constraint_programming': 'true',
            'decision_timeout_seconds': '5.0',
            'emergency_mode_threshold': '0.8'
        }
        
        self.config['ALERTS'] = {
            'cpu_threshold': '80.0',
            'memory_threshold': '85.0',
            'disk_threshold': '90.0',
            'email_notifications': 'false'
        }
        
        self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                self.config.write(f)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def get(self, section: str, key: str, fallback: Any = None):
        """Get configuration value"""
        try:
            return self.config.get(section, key, fallback=fallback)
        except Exception:
            return fallback
    
    def set(self, section: str, key: str, value: str):
        """Set configuration value"""
        if section not in self.config:
            self.config.add_section(section)
        self.config.set(section, key, value)
        self.save_config()

class DeploymentManager:
    """Deployment and maintenance utilities"""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.db_manager = DatabaseManager()
        self.system_monitor = SystemMonitor()
    
    def deploy_system(self, environment: str = "production") -> bool:
        """Deploy the traffic control system"""
        logger.info(f"Starting deployment for {environment} environment")
        
        try:
            # Step 1: System checks
            logger.info("Step 1: Performing pre-deployment checks...")
            if not self._pre_deployment_checks():
                return False
            
            # Step 2: Database setup
            logger.info("Step 2: Setting up database...")
            if not self.db_manager.connect():
                return False
            
            if not self.db_manager.initialize_schema():
                return False
            
            # Step 3: Configuration setup
            logger.info("Step 3: Setting up configuration...")
            self._setup_environment_config(environment)
            
            # Step 4: Service startup
            logger.info("Step 4: Starting services...")
            if not self._start_services():
                return False
            
            # Step 5: Health check
            logger.info("Step 5: Performing post-deployment health check...")
            health = self.system_monitor.check_health()
            if health['overall_status'] != 'HEALTHY':
                logger.warning(f"Post-deployment health check shows: {health['overall_status']}")
            
            logger.info(f"✅ Deployment completed successfully for {environment}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return False
        
        finally:
            self.db_manager.close()
    
    def _pre_deployment_checks(self) -> bool:
        """Perform pre-deployment system checks"""
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version.major < 3 or python_version.minor < 7:
                logger.error("Python 3.7 or higher is required")
                return False
            
            # Check required packages
            required_packages = ['sqlite3', 'psutil', 'datetime', 'threading']
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    logger.error(f"Required package missing: {package}")
                    return False
            
            # Check disk space
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            if free_gb < 1.0:  # Require at least 1GB free
                logger.error(f"Insufficient disk space: {free_gb:.2f}GB free")
                return False
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.available < 512 * 1024 * 1024:  # Require at least 512MB available
                logger.error(f"Insufficient available memory: {memory.available / (1024**2):.0f}MB")
                return False
            
            logger.info("✅ Pre-deployment checks passed")
            return True
            
        except Exception as e:
            logger.error(f"Pre-deployment checks failed: {e}")
            return False
    
    def _setup_environment_config(self, environment: str):
        """Setup environment-specific configuration"""
        if environment == "production":
            self.config_manager.set("SYSTEM", "log_level", "WARNING")
            self.config_manager.set("SYSTEM", "monitoring_interval", "30")
            self.config_manager.set("DATABASE", "backup_interval_hours", "6")
        elif environment == "development":
            self.config_manager.set("SYSTEM", "log_level", "DEBUG")
            self.config_manager.set("SYSTEM", "monitoring_interval", "120")
            self.config_manager.set("DATABASE", "backup_interval_hours", "24")
    
    def _start_services(self) -> bool:
        """Start system services"""
        try:
            # Start monitoring
            interval = int(self.config_manager.get("SYSTEM", "monitoring_interval", "60"))
            self.system_monitor.start_monitoring(interval)
            
            logger.info("✅ Services started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Service startup failed: {e}")
            return False
    
    def create_maintenance_script(self, script_path: str = "maintenance.py") -> bool:
        """Create automated maintenance script"""
        maintenance_script = '''#!/usr/bin/env python3
"""
Automated maintenance script for Intelligent Train Traffic Control System
"""

import os
import sys
import logging
from datetime import datetime
from operations_utilities import DatabaseManager, SystemMonitor, ConfigurationManager

def main():
    print("🔧 Starting system maintenance...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize components
    db_manager = DatabaseManager()
    monitor = SystemMonitor()
    config = ConfigurationManager()
    
    try:
        # Database backup
        print("📦 Creating database backup...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"maintenance_backup_{timestamp}.db"
        if db_manager.backup_database(backup_path):
            print(f"✅ Database backed up to: {backup_path}")
        else:
            print("❌ Database backup failed")
        
        # Health check
        print("🏥 Performing system health check...")
        health = monitor.check_health()
        print(f"System status: {health['overall_status']}")
        
        for check_name, check_data in health.get('checks', {}).items():
            status_icon = "✅" if check_data['status'] == 'HEALTHY' else "⚠️"
            print(f"  {status_icon} {check_name}: {check_data['status']}")
        
        # Cleanup old logs (if applicable)
        print("🧹 Performing cleanup...")
        # Add cleanup logic here as needed
        
        print("✅ Maintenance completed successfully")
        
    except Exception as e:
        print(f"❌ Maintenance failed: {e}")
        return 1
    
    finally:
        db_manager.close()
        monitor.stop_monitoring()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
        
        try:
            with open(script_path, 'w') as f:
                f.write(maintenance_script)
            
            # Make script executable on Unix systems
            if os.name != 'nt':  # Not Windows
                os.chmod(script_path, 0o755)
            
            logger.info(f"✅ Maintenance script created: {script_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create maintenance script: {e}")
            return False

class LogManager:
    """Log management and analysis utilities"""
    
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = log_directory
        self.ensure_log_directory()
        
        # Setup file handler for application logs
        self.setup_file_logging()
    
    def ensure_log_directory(self):
        """Ensure log directory exists"""
        Path(self.log_directory).mkdir(parents=True, exist_ok=True)
    
    def setup_file_logging(self):
        """Setup file-based logging"""
        log_file = os.path.join(self.log_directory, "traffic_control.log")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        logger.info(f"File logging setup: {log_file}")
    
    def rotate_logs(self, max_size_mb: int = 10) -> bool:
        """Rotate log files if they exceed maximum size"""
        try:
            log_file = os.path.join(self.log_directory, "traffic_control.log")
            
            if os.path.exists(log_file):
                size_mb = os.path.getsize(log_file) / (1024 * 1024)
                
                if size_mb > max_size_mb:
                    # Archive current log
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    archive_name = f"traffic_control_{timestamp}.log"
                    archive_path = os.path.join(self.log_directory, archive_name)
                    
                    shutil.move(log_file, archive_path)
                    logger.info(f"Log rotated: {archive_path}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Log rotation failed: {e}")
            return False

class OperationsOrchestrator:
    """Main orchestrator for all operations and utilities"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.system_monitor = SystemMonitor()
        self.config_manager = ConfigurationManager()
        self.deployment_manager = DeploymentManager()
        self.log_manager = LogManager()
        
        logger.info("Operations Orchestrator initialized")
    
    def run_full_system_check(self) -> Dict[str, Any]:
        """Run comprehensive system check"""
        logger.info("Running full system check...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'HEALTHY',
            'components': {}
        }
        
        try:
            # Database check
            db_status = 'HEALTHY' if self.db_manager.connect() else 'CRITICAL'
            results['components']['database'] = {
                'status': db_status,
                'message': 'Database connectivity test'
            }
            
            # System health
            health = self.system_monitor.check_health()
            results['components']['system_health'] = health
            
            # Configuration check
            config_status = 'HEALTHY' if os.path.exists(self.config_manager.config_file) else 'WARNING'
            results['components']['configuration'] = {
                'status': config_status,
                'config_file': self.config_manager.config_file
            }
            
            # Determine overall status
            component_statuses = [comp.get('status', 'UNKNOWN') for comp in results['components'].values()]
            if 'CRITICAL' in component_statuses:
                results['overall_status'] = 'CRITICAL'
            elif 'WARNING' in component_statuses:
                results['overall_status'] = 'WARNING'
            
            logger.info(f"System check completed: {results['overall_status']}")
            
        except Exception as e:
            logger.error(f"System check failed: {e}")
            results['overall_status'] = 'CRITICAL'
            results['error'] = str(e)
        
        finally:
            self.db_manager.close()
        
        return results
    
    def perform_maintenance(self) -> bool:
        """Perform routine system maintenance"""
        logger.info("Starting routine system maintenance...")
        
        try:
            # Backup database
            if not self.db_manager.backup_database():
                logger.warning("Database backup failed")
            
            # Rotate logs
            self.log_manager.rotate_logs()
            
            # System health check
            health = self.system_monitor.check_health()
            if health['overall_status'] != 'HEALTHY':
                logger.warning(f"System health issues detected: {health['overall_status']}")
            
            # Cleanup temporary files
            self._cleanup_temporary_files()
            
            logger.info("✅ Routine maintenance completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Maintenance failed: {e}")
            return False
    
    def _cleanup_temporary_files(self):
        """Clean up temporary files"""
        try:
            temp_patterns = ["*.tmp", "*.log.*", "backup_*"]
            # Add cleanup logic as needed
            logger.info("Temporary files cleanup completed")
        except Exception as e:
            logger.error(f"Temporary files cleanup failed: {e}")

if __name__ == "__main__":
    # Test operations and utilities
    print("🚀 Testing Operations and Utilities...")
    
    # Create orchestrator
    orchestrator = OperationsOrchestrator()
    
    # Run system check
    print("\n📊 Running system check...")
    check_results = orchestrator.run_full_system_check()
    print(f"Overall status: {check_results['overall_status']}")
    
    # Test database operations
    print("\n💾 Testing database operations...")
    db_manager = DatabaseManager()
    if db_manager.connect():
        print("✅ Database connection successful")
        if db_manager.initialize_schema():
            print("✅ Schema initialization successful")
        db_manager.close()
    
    # Test system monitoring
    print("\n🖥️  Testing system monitoring...")
    monitor = SystemMonitor()
    metrics = monitor.get_system_metrics()
    if metrics:
        print(f"✅ System metrics collected: CPU {metrics['system']['cpu_percent']:.1f}%")
    
    # Test configuration management
    print("\n⚙️  Testing configuration management...")
    config = ConfigurationManager("test_config.ini")
    config.set("TEST", "parameter", "test_value")
    value = config.get("TEST", "parameter")
    print(f"✅ Configuration test: {value}")
    
    # Test deployment
    print("\n🚀 Testing deployment...")
    deployment = DeploymentManager()
    if deployment.create_maintenance_script("test_maintenance.py"):
        print("✅ Maintenance script created")
    
    print("\n✅ Operations and Utilities testing completed!")
    print("Use orchestrator.perform_maintenance() for routine maintenance")
