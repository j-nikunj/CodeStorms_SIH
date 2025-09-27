# MERGED DELIVERABLE 1: CORE DATA LAYER
# File: core_data_layer.py

"""
Core Data Layer for Intelligent Train Traffic Control System
Combines data models and data management functionality into a unified layer
"""

import sqlite3
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import os
import hashlib
import uuid
import pickle

logger = logging.getLogger(__name__)

# ==================== CORE DATA MODELS ====================

class TrainType(Enum):
    """Train type classifications"""
    EXPRESS = "EXPRESS"
    LOCAL = "LOCAL"  
    FREIGHT = "FREIGHT"
    MAINTENANCE = "MAINTENANCE"
    SPECIAL = "SPECIAL"

class TrainStatus(Enum):
    """Train operational status"""
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    DELAYED = "DELAYED"
    HELD = "HELD"
    COMPLETED = "COMPLETED"

class Priority(Enum):
    """Priority levels for trains"""
    CRITICAL = 1    # VIP, Emergency
    HIGH = 2        # Express trains
    MEDIUM = 3      # Local trains
    LOW = 4         # Freight
    LOWEST = 5      # Maintenance

@dataclass
class Train:
    """
    Comprehensive train model with all operational parameters
    """
    train_id: str
    train_type: TrainType
    priority: int  # 1=highest, 5=lowest
    scheduled_arrival: datetime
    scheduled_departure: datetime
    current_location: str
    destination: str
    delay_minutes: int = 0
    status: TrainStatus = TrainStatus.SCHEDULED
    speed_kmph: float = 80.0
    length_meters: float = 400.0
    passenger_load: int = 0
    route_sections: List[str] = None
    
    def __post_init__(self):
        if self.route_sections is None:
            self.route_sections = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'train_id': self.train_id,
            'train_type': self.train_type.value,
            'priority': self.priority,
            'scheduled_arrival': self.scheduled_arrival.isoformat(),
            'scheduled_departure': self.scheduled_departure.isoformat(),
            'current_location': self.current_location,
            'destination': self.destination,
            'delay_minutes': self.delay_minutes,
            'status': self.status.value,
            'speed_kmph': self.speed_kmph,
            'length_meters': self.length_meters,
            'passenger_load': self.passenger_load,
            'route_sections': self.route_sections
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Train':
        """Create Train instance from dictionary"""
        return cls(
            train_id=data['train_id'],
            train_type=TrainType(data['train_type']),
            priority=data['priority'],
            scheduled_arrival=datetime.fromisoformat(data['scheduled_arrival']),
            scheduled_departure=datetime.fromisoformat(data['scheduled_departure']),
            current_location=data['current_location'],
            destination=data['destination'],
            delay_minutes=data.get('delay_minutes', 0),
            status=TrainStatus(data.get('status', 'SCHEDULED')),
            speed_kmph=data.get('speed_kmph', 80.0),
            length_meters=data.get('length_meters', 400.0),
            passenger_load=data.get('passenger_load', 0),
            route_sections=data.get('route_sections', [])
        )
    
    def get_actual_arrival(self) -> datetime:
        """Get actual arrival time including delays"""
        return self.scheduled_arrival + timedelta(minutes=self.delay_minutes)
    
    def get_actual_departure(self) -> datetime:
        """Get actual departure time including delays"""
        return self.scheduled_departure + timedelta(minutes=self.delay_minutes)
    
    def is_delayed(self) -> bool:
        """Check if train is delayed"""
        return self.delay_minutes > 0 or self.status == TrainStatus.DELAYED
    
    def get_priority_weight(self) -> float:
        """Get priority weight for optimization (lower number = higher priority)"""
        base_weight = self.priority
        
        # Adjust based on train type
        if self.train_type == TrainType.EXPRESS:
            base_weight *= 0.8
        elif self.train_type == TrainType.FREIGHT:
            base_weight *= 1.2
        elif self.train_type == TrainType.MAINTENANCE:
            base_weight *= 1.5
            
        # Penalty for delays
        if self.is_delayed():
            base_weight *= (1 + self.delay_minutes / 60.0)
            
        return base_weight

@dataclass
class Section:
    """
    Railway section with infrastructure constraints and capabilities
    """
    section_id: str
    start_station: str
    end_station: str
    length_km: float
    max_speed_kmph: float
    capacity_trains: int
    has_loop: bool = False
    gradient_percent: float = 0.0
    signal_spacing_km: float = 2.0
    electrified: bool = True
    maintenance_windows: List[Dict] = None
    current_occupancy: List[str] = None
    
    def __post_init__(self):
        if self.current_occupancy is None:
            self.current_occupancy = []
        if self.maintenance_windows is None:
            self.maintenance_windows = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'section_id': self.section_id,
            'start_station': self.start_station,
            'end_station': self.end_station,
            'length_km': self.length_km,
            'max_speed_kmph': self.max_speed_kmph,
            'capacity_trains': self.capacity_trains,
            'has_loop': self.has_loop,
            'gradient_percent': self.gradient_percent,
            'signal_spacing_km': self.signal_spacing_km,
            'electrified': self.electrified,
            'maintenance_windows': self.maintenance_windows,
            'current_occupancy': self.current_occupancy
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Section':
        """Create Section instance from dictionary"""
        return cls(
            section_id=data['section_id'],
            start_station=data['start_station'],
            end_station=data['end_station'],
            length_km=data['length_km'],
            max_speed_kmph=data['max_speed_kmph'],
            capacity_trains=data['capacity_trains'],
            has_loop=data.get('has_loop', False),
            gradient_percent=data.get('gradient_percent', 0.0),
            signal_spacing_km=data.get('signal_spacing_km', 2.0),
            electrified=data.get('electrified', True),
            maintenance_windows=data.get('maintenance_windows', []),
            current_occupancy=data.get('current_occupancy', [])
        )
    
    def get_utilization(self) -> float:
        """Get current utilization percentage"""
        if self.capacity_trains == 0:
            return 0.0
        return len(self.current_occupancy) / self.capacity_trains
    
    def can_accommodate(self, train: Train) -> bool:
        """Check if section can accommodate another train"""
        if len(self.current_occupancy) >= self.capacity_trains:
            return False
        
        # Check if train is compatible with electrification
        if not self.electrified and train.train_type == TrainType.EXPRESS:
            return False
            
        return True
    
    def add_train(self, train_id: str) -> bool:
        """Add train to section occupancy"""
        if len(self.current_occupancy) < self.capacity_trains:
            self.current_occupancy.append(train_id)
            return True
        return False
    
    def remove_train(self, train_id: str) -> bool:
        """Remove train from section occupancy"""
        if train_id in self.current_occupancy:
            self.current_occupancy.remove(train_id)
            return True
        return False
    
    def calculate_travel_time(self, train: Train) -> float:
        """Calculate travel time for a train through this section (in minutes)"""
        # Base speed consideration
        effective_speed = min(train.speed_kmph, self.max_speed_kmph)
        
        # Adjust for gradient
        if self.gradient_percent > 0:
            speed_reduction = self.gradient_percent * 0.1  # 10% reduction per 1% gradient
            effective_speed *= (1 - speed_reduction)
        
        # Adjust for train type
        type_factors = {
            TrainType.EXPRESS: 1.1,      # 10% faster
            TrainType.LOCAL: 1.0,        # Base speed
            TrainType.FREIGHT: 0.7,      # 30% slower
            TrainType.MAINTENANCE: 0.5,  # 50% slower
            TrainType.SPECIAL: 0.9       # 10% slower
        }
        
        effective_speed *= type_factors.get(train.train_type, 1.0)
        
        # Ensure minimum speed
        effective_speed = max(effective_speed, 10.0)  # Minimum 10 kmph
        
        # Calculate time in minutes
        travel_time_hours = self.length_km / effective_speed
        return travel_time_hours * 60.0

@dataclass
class Constraint:
    """
    Operational constraints for railway scheduling
    """
    constraint_id: str
    constraint_type: str  # 'SAFETY', 'CAPACITY', 'PRIORITY', 'MAINTENANCE', 'SIGNAL'
    description: str
    applies_to: List[str]  # train_ids or section_ids
    start_time: datetime
    end_time: datetime
    severity: int = 1  # 1=critical, 5=minor
    parameters: Dict = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
    
    def is_active(self, current_time: datetime = None) -> bool:
        """Check if constraint is currently active"""
        if current_time is None:
            current_time = datetime.now()
        return self.start_time <= current_time <= self.end_time
    
    def affects_train(self, train_id: str) -> bool:
        """Check if constraint affects specific train"""
        return train_id in self.applies_to
    
    def affects_section(self, section_id: str) -> bool:
        """Check if constraint affects specific section"""
        return section_id in self.applies_to

@dataclass
class ScheduleEntry:
    """
    Individual schedule entry for a train in a section
    """
    train_id: str
    section_id: str
    planned_start_time: datetime
    planned_end_time: datetime
    actual_start_time: Optional[datetime] = None
    actual_end_time: Optional[datetime] = None
    delay_minutes: float = 0.0
    confidence_score: float = 1.0
    notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'train_id': self.train_id,
            'section_id': self.section_id,
            'planned_start_time': self.planned_start_time.isoformat(),
            'planned_end_time': self.planned_end_time.isoformat(),
            'actual_start_time': self.actual_start_time.isoformat() if self.actual_start_time else None,
            'actual_end_time': self.actual_end_time.isoformat() if self.actual_end_time else None,
            'delay_minutes': self.delay_minutes,
            'confidence_score': self.confidence_score,
            'notes': self.notes
        }
    
    def is_completed(self) -> bool:
        """Check if schedule entry is completed"""
        return self.actual_end_time is not None
    
    def get_actual_duration(self) -> Optional[float]:
        """Get actual duration in minutes"""
        if self.actual_start_time and self.actual_end_time:
            return (self.actual_end_time - self.actual_start_time).total_seconds() / 60.0
        return None

# ==================== DATA MANAGEMENT LAYER ====================

class DataManager:
    """
    Comprehensive data management layer with database operations,
    caching, and real-time data synchronization capabilities
    """
    
    def __init__(self, db_path: str = "railway_control.db", enable_caching: bool = True):
        self.db_path = db_path
        self.enable_caching = enable_caching
        self.cache = {} if enable_caching else None
        self.cache_expiry = {}
        self.cache_timeout_minutes = 30
        
        # Thread safety
        self._db_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        
        # Database connection pool (simplified)
        self._connection = None
        
        # Performance metrics
        self.query_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize database
        self.init_database()
        
        logger.info(f"DataManager initialized with database: {db_path}")
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with thread safety"""
        if not self._connection:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        return self._connection
    
    def init_database(self):
        """Initialize database schema with comprehensive tables"""
        with self._db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                # Trains table with comprehensive fields
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trains (
                        train_id TEXT PRIMARY KEY,
                        train_type TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        scheduled_arrival TEXT NOT NULL,
                        scheduled_departure TEXT NOT NULL,
                        current_location TEXT NOT NULL,
                        destination TEXT NOT NULL,
                        delay_minutes INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'SCHEDULED',
                        speed_kmph REAL DEFAULT 80.0,
                        length_meters REAL DEFAULT 400.0,
                        passenger_load INTEGER DEFAULT 0,
                        route_sections TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT DEFAULT '{}'
                    )
                ''')
                
                # Sections table with infrastructure details
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sections (
                        section_id TEXT PRIMARY KEY,
                        start_station TEXT NOT NULL,
                        end_station TEXT NOT NULL,
                        length_km REAL NOT NULL,
                        max_speed_kmph REAL NOT NULL,
                        capacity_trains INTEGER NOT NULL,
                        has_loop BOOLEAN DEFAULT 0,
                        gradient_percent REAL DEFAULT 0.0,
                        signal_spacing_km REAL DEFAULT 2.0,
                        electrified BOOLEAN DEFAULT 1,
                        current_occupancy TEXT DEFAULT '[]',
                        maintenance_windows TEXT DEFAULT '[]',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT DEFAULT '{}'
                    )
                ''')
                
                # Decisions table for audit trail
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS decisions (
                        decision_id TEXT PRIMARY KEY,
                        section_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        optimization_objective TEXT,
                        trains_considered INTEGER,
                        response_time_ms REAL,
                        confidence_score REAL,
                        success BOOLEAN,
                        primary_solution TEXT,
                        alternatives TEXT DEFAULT '[]',
                        recommendations TEXT DEFAULT '[]',
                        action_required TEXT DEFAULT '[]',
                        user_preferences TEXT DEFAULT '{}',
                        emergency_mode BOOLEAN DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (section_id) REFERENCES sections (section_id)
                    )
                ''')
                
                # Performance metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL,
                        section_id TEXT,
                        train_id TEXT,
                        decision_id TEXT,
                        metadata TEXT DEFAULT '{}',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # System events and alerts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_events (
                        event_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        severity TEXT DEFAULT 'INFO',
                        component TEXT,
                        message TEXT,
                        details TEXT DEFAULT '{}',
                        resolved BOOLEAN DEFAULT 0,
                        resolved_at TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Constraints and disruptions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS constraints (
                        constraint_id TEXT PRIMARY KEY,
                        constraint_type TEXT NOT NULL,
                        description TEXT,
                        applies_to TEXT DEFAULT '[]',
                        start_time TEXT NOT NULL,
                        end_time TEXT NOT NULL,
                        severity INTEGER DEFAULT 1,
                        parameters TEXT DEFAULT '{}',
                        active BOOLEAN DEFAULT 1,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # User sessions and preferences table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        login_time TEXT NOT NULL,
                        last_activity TEXT NOT NULL,
                        preferences TEXT DEFAULT '{}',
                        role TEXT DEFAULT 'controller',
                        active BOOLEAN DEFAULT 1,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better query performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trains_status ON trains (status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trains_location ON trains (current_location)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_decisions_section ON decisions (section_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events (timestamp)')
                
                conn.commit()
                logger.info("Database schema initialized successfully")
                
            except Exception as e:
                logger.error(f"Database initialization failed: {e}")
                conn.rollback()
                raise
    
    def store_train(self, train: Train) -> bool:
        """Store or update train information"""
        try:
            with self._db_lock:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                # Prepare train data
                train_data = (
                    train.train_id,
                    train.train_type.value,
                    train.priority,
                    train.scheduled_arrival.isoformat(),
                    train.scheduled_departure.isoformat(),
                    train.current_location,
                    train.destination,
                    train.delay_minutes,
                    train.status.value,
                    train.speed_kmph,
                    train.length_meters,
                    train.passenger_load,
                    json.dumps(train.route_sections),
                    datetime.now().isoformat(),
                    json.dumps({})  # metadata placeholder
                )
                
                cursor.execute('''
                    INSERT OR REPLACE INTO trains 
                    (train_id, train_type, priority, scheduled_arrival, scheduled_departure,
                     current_location, destination, delay_minutes, status, speed_kmph,
                     length_meters, passenger_load, route_sections, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', train_data)
                
                conn.commit()
                self.query_count += 1
                
                # Clear related cache entries
                self._invalidate_cache(f"trains_*")
                
                logger.debug(f"Stored train {train.train_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store train {train.train_id}: {e}")
            return False
    
    def get_train(self, train_id: str) -> Optional[Train]:
        """Get single train by ID with caching"""
        cache_key = f"train_{train_id}"
        
        # Check cache first
        if self.enable_caching:
            cached_train = self._get_from_cache(cache_key)
            if cached_train:
                return cached_train
        
        try:
            with self._db_lock:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM trains WHERE train_id = ?', (train_id,))
                row = cursor.fetchone()
                self.query_count += 1
                
                if row:
                    train = self._row_to_train(row)
                    
                    # Cache the result
                    if self.enable_caching:
                        self._store_in_cache(cache_key, train)
                    
                    return train
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get train {train_id}: {e}")
            return None
    
    def get_trains(self, filters: Dict = None, limit: int = None, offset: int = 0) -> List[Train]:
        """Get trains with flexible filtering and pagination"""
        try:
            with self._db_lock:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                # Build query with filters
                query = "SELECT * FROM trains"
                params = []
                conditions = []
                
                if filters:
                    if 'status' in filters:
                        conditions.append("status = ?")
                        params.append(filters['status'])
                    
                    if 'train_type' in filters:
                        conditions.append("train_type = ?")
                        params.append(filters['train_type'])
                    
                    if 'priority' in filters:
                        conditions.append("priority = ?")
                        params.append(filters['priority'])
                    
                    if 'current_location' in filters:
                        conditions.append("current_location = ?")
                        params.append(filters['current_location'])
                    
                    if 'section_id' in filters:
                        conditions.append("route_sections LIKE ?")
                        params.append(f'%{filters["section_id"]}%')
                    
                    if 'delayed' in filters and filters['delayed']:
                        conditions.append("delay_minutes > 0")
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                # Add ordering
                query += " ORDER BY scheduled_arrival"
                
                # Add pagination
                if limit:
                    query += f" LIMIT {limit}"
                    if offset > 0:
                        query += f" OFFSET {offset}"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                self.query_count += 1
                
                trains = [self._row_to_train(row) for row in rows]
                return trains
                
        except Exception as e:
            logger.error(f"Failed to get trains with filters {filters}: {e}")
            return []
    
    def store_section(self, section: Section) -> bool:
        """Store or update section information"""
        try:
            with self._db_lock:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                section_data = (
                    section.section_id,
                    section.start_station,
                    section.end_station,
                    section.length_km,
                    section.max_speed_kmph,
                    section.capacity_trains,
                    section.has_loop,
                    section.gradient_percent,
                    section.signal_spacing_km,
                    section.electrified,
                    json.dumps(section.current_occupancy),
                    json.dumps(section.maintenance_windows),
                    datetime.now().isoformat(),
                    json.dumps({})
                )
                
                cursor.execute('''
                    INSERT OR REPLACE INTO sections 
                    (section_id, start_station, end_station, length_km, max_speed_kmph,
                     capacity_trains, has_loop, gradient_percent, signal_spacing_km,
                     electrified, current_occupancy, maintenance_windows, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', section_data)
                
                conn.commit()
                self.query_count += 1
                
                # Clear cache
                self._invalidate_cache(f"sections_*")
                
                logger.debug(f"Stored section {section.section_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store section {section.section_id}: {e}")
            return False
    
    def get_section(self, section_id: str) -> Optional[Section]:
        """Get single section by ID"""
        cache_key = f"section_{section_id}"
        
        if self.enable_caching:
            cached_section = self._get_from_cache(cache_key)
            if cached_section:
                return cached_section
        
        try:
            with self._db_lock:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM sections WHERE section_id = ?', (section_id,))
                row = cursor.fetchone()
                self.query_count += 1
                
                if row:
                    section = self._row_to_section(row)
                    
                    if self.enable_caching:
                        self._store_in_cache(cache_key, section)
                    
                    return section
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get section {section_id}: {e}")
            return None
    
    def get_sections(self, filters: Dict = None) -> List[Section]:
        """Get sections with filtering"""
        try:
            with self._db_lock:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                query = "SELECT * FROM sections"
                params = []
                conditions = []
                
                if filters:
                    if 'start_station' in filters:
                        conditions.append("start_station = ?")
                        params.append(filters['start_station'])
                    
                    if 'has_loop' in filters:
                        conditions.append("has_loop = ?")
                        params.append(filters['has_loop'])
                    
                    if 'electrified' in filters:
                        conditions.append("electrified = ?")
                        params.append(filters['electrified'])
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY section_id"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                self.query_count += 1
                
                sections = [self._row_to_section(row) for row in rows]
                return sections
                
        except Exception as e:
            logger.error(f"Failed to get sections with filters {filters}: {e}")
            return []
    
    def store_performance_metric(self, metric_name: str, metric_value: float, 
                                section_id: str = None, train_id: str = None,
                                decision_id: str = None, metadata: Dict = None) -> bool:
        """Store performance metric"""
        try:
            with self._db_lock:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                metric_data = (
                    datetime.now().isoformat(),
                    metric_name,
                    metric_value,
                    section_id,
                    train_id,
                    decision_id,
                    json.dumps(metadata or {})
                )
                
                cursor.execute('''
                    INSERT INTO performance_metrics 
                    (timestamp, metric_name, metric_value, section_id, train_id, decision_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', metric_data)
                
                conn.commit()
                self.query_count += 1
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store performance metric {metric_name}: {e}")
            return False
    
    def store_system_event(self, event_type: str, message: str, severity: str = 'INFO',
                          component: str = None, details: Dict = None) -> bool:
        """Store system event or alert"""
        try:
            with self._db_lock:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                event_data = (
                    str(uuid.uuid4()),
                    datetime.now().isoformat(),
                    event_type,
                    severity,
                    component,
                    message,
                    json.dumps(details or {})
                )
                
                cursor.execute('''
                    INSERT INTO system_events 
                    (event_id, timestamp, event_type, severity, component, message, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', event_data)
                
                conn.commit()
                self.query_count += 1
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to store system event: {e}")
            return False
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        stats = {
            'query_count': self.query_count,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            'cache_size': len(self.cache) if self.cache else 0,
            'table_counts': {},
            'database_size_mb': 0
        }
        
        try:
            with self._db_lock:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                # Get table row counts
                tables = ['trains', 'sections', 'decisions', 'performance_metrics', 
                         'system_events', 'constraints', 'user_sessions']
                
                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    count = cursor.fetchone()[0]
                    stats['table_counts'][table] = count
                
                # Get database file size
                if os.path.exists(self.db_path):
                    size_bytes = os.path.getsize(self.db_path)
                    stats['database_size_mb'] = size_bytes / (1024 * 1024)
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
        
        return stats
    
    def _row_to_train(self, row) -> Train:
        """Convert database row to Train object"""
        return Train(
            train_id=row['train_id'],
            train_type=TrainType(row['train_type']),
            priority=row['priority'],
            scheduled_arrival=datetime.fromisoformat(row['scheduled_arrival']),
            scheduled_departure=datetime.fromisoformat(row['scheduled_departure']),
            current_location=row['current_location'],
            destination=row['destination'],
            delay_minutes=row['delay_minutes'],
            status=TrainStatus(row['status']),
            speed_kmph=row['speed_kmph'],
            length_meters=row['length_meters'],
            passenger_load=row['passenger_load'],
            route_sections=json.loads(row['route_sections']) if row['route_sections'] else []
        )
    
    def _row_to_section(self, row) -> Section:
        """Convert database row to Section object"""
        return Section(
            section_id=row['section_id'],
            start_station=row['start_station'],
            end_station=row['end_station'],
            length_km=row['length_km'],
            max_speed_kmph=row['max_speed_kmph'],
            capacity_trains=row['capacity_trains'],
            has_loop=bool(row['has_loop']),
            gradient_percent=row['gradient_percent'],
            signal_spacing_km=row['signal_spacing_km'],
            electrified=bool(row['electrified']),
            current_occupancy=json.loads(row['current_occupancy']) if row['current_occupancy'] else [],
            maintenance_windows=json.loads(row['maintenance_windows']) if row['maintenance_windows'] else []
        )
    
    def _get_from_cache(self, key: str) -> Any:
        """Get item from cache with expiry check"""
        if not self.enable_caching:
            return None
        
        with self._cache_lock:
            if key in self.cache:
                # Check if expired
                if key in self.cache_expiry:
                    if datetime.now() > self.cache_expiry[key]:
                        del self.cache[key]
                        del self.cache_expiry[key]
                        self.cache_misses += 1
                        return None
                
                self.cache_hits += 1
                return self.cache[key]
            
            self.cache_misses += 1
            return None
    
    def _store_in_cache(self, key: str, value: Any):
        """Store item in cache with expiry"""
        if not self.enable_caching:
            return
        
        with self._cache_lock:
            self.cache[key] = value
            self.cache_expiry[key] = datetime.now() + timedelta(minutes=self.cache_timeout_minutes)
    
    def _invalidate_cache(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        if not self.enable_caching:
            return
        
        with self._cache_lock:
            keys_to_remove = [key for key in self.cache.keys() if pattern.replace('*', '') in key]
            for key in keys_to_remove:
                del self.cache[key]
                if key in self.cache_expiry:
                    del self.cache_expiry[key]
    
    def close(self):
        """Close database connection and cleanup"""
        if self._connection:
            self._connection.close()
            self._connection = None
        
        if self.cache:
            self.cache.clear()
            self.cache_expiry.clear()
        
        logger.info("DataManager closed")

# ==================== UTILITY FUNCTIONS ====================

def create_sample_train(train_id: str, train_type: TrainType, base_time: datetime, 
                       delay: int = 0, route: List[str] = None) -> Train:
    """Create a sample train for testing"""
    if route is None:
        route = []
    
    priorities = {
        TrainType.EXPRESS: 1,
        TrainType.LOCAL: 3,
        TrainType.FREIGHT: 4,
        TrainType.MAINTENANCE: 5,
        TrainType.SPECIAL: 2
    }
    
    speeds = {
        TrainType.EXPRESS: 120.0,
        TrainType.LOCAL: 80.0,
        TrainType.FREIGHT: 45.0,
        TrainType.MAINTENANCE: 30.0,
        TrainType.SPECIAL: 100.0
    }
    
    return Train(
        train_id=train_id,
        train_type=train_type,
        priority=priorities[train_type],
        scheduled_arrival=base_time,
        scheduled_departure=base_time + timedelta(minutes=10),
        current_location="Station_A",
        destination="Station_B",
        delay_minutes=delay,
        speed_kmph=speeds[train_type],
        route_sections=route
    )

def create_sample_section(section_id: str, start: str, end: str, 
                         length: float, capacity: int = 3) -> Section:
    """Create a sample section for testing"""
    return Section(
        section_id=section_id,
        start_station=start,
        end_station=end,
        length_km=length,
        max_speed_kmph=100.0,
        capacity_trains=capacity,
        has_loop=capacity > 2,
        electrified=True
    )

def validate_train(train: Train) -> List[str]:
    """Validate train data and return list of errors"""
    errors = []
    
    if not train.train_id:
        errors.append("Train ID cannot be empty")
    
    if train.priority < 1 or train.priority > 5:
        errors.append("Priority must be between 1 and 5")
    
    if train.speed_kmph <= 0:
        errors.append("Speed must be positive")
        
    if train.length_meters <= 0:
        errors.append("Length must be positive")
        
    if train.scheduled_departure < train.scheduled_arrival:
        errors.append("Departure time cannot be before arrival time")
    
    return errors

def validate_section(section: Section) -> List[str]:
    """Validate section data and return list of errors"""
    errors = []
    
    if not section.section_id:
        errors.append("Section ID cannot be empty")
        
    if section.length_km <= 0:
        errors.append("Length must be positive")
        
    if section.max_speed_kmph <= 0:
        errors.append("Max speed must be positive")
        
    if section.capacity_trains <= 0:
        errors.append("Capacity must be positive")
        
    if len(section.current_occupancy) > section.capacity_trains:
        errors.append("Current occupancy exceeds capacity")
    
    return errors

if __name__ == "__main__":
    # Test the core data layer
    print("Testing Core Data Layer...")
    
    # Create data manager
    dm = DataManager(db_path=":memory:", enable_caching=True)
    
    # Test train operations
    base_time = datetime.now()
    test_train = create_sample_train("TEST001", TrainType.EXPRESS, base_time)
    
    success = dm.store_train(test_train)
    print(f"Store train success: {success}")
    
    retrieved_train = dm.get_train("TEST001")
    print(f"Retrieved train: {retrieved_train.train_id if retrieved_train else 'None'}")
    
    # Test section operations
    test_section = create_sample_section("SEC001", "StationA", "StationB", 50.0)
    success = dm.store_section(test_section)
    print(f"Store section success: {success}")
    
    # Get database statistics
    stats = dm.get_database_stats()
    print(f"Database stats: {stats}")
    
    dm.close()
    print("Core Data Layer test completed!")
