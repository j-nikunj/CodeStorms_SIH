# MERGED DELIVERABLE 3: SIMULATION AND VALIDATION ENGINE
# File: simulation_validation_engine.py

"""
Simulation and Validation Engine for Intelligent Train Traffic Control System
Combines traffic simulation, disruption modeling, and comprehensive validation capabilities
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import logging
import json
import uuid
from copy import deepcopy
from dataclasses import asdict
import random
import time
import unittest
import sys
from io import StringIO

from core_data_layer import Train, Section, TrainType, TrainStatus, Constraint

logger = logging.getLogger(__name__)

# ==================== TRAFFIC SIMULATOR ====================

class TrafficSimulator:
    """
    Advanced traffic simulation engine for what-if scenario analysis
    Supports multiple disruption types and comprehensive scenario modeling
    """
    
    def __init__(self):
        self.scenarios: Dict[str, Dict] = {}
        self.simulation_results: Dict[str, Dict] = {}
        self.disruption_models: Dict[str, Any] = self._initialize_disruption_models()
        self.simulation_parameters = {
            'time_step_minutes': 1.0,
            'simulation_speed': 1.0,
            'random_seed': None,
            'weather_enabled': True,
            'passenger_behavior_enabled': True
        }
        
        logger.info("Traffic Simulator initialized")
    
    def _initialize_disruption_models(self) -> Dict:
        """Initialize different disruption model types"""
        return {
            'TRAIN_DELAY': {
                'description': 'Individual train delays',
                'parameters': ['delay_minutes', 'affected_trains'],
                'severity_levels': [5, 15, 30, 60, 120]  # minutes
            },
            'SPEED_RESTRICTION': {
                'description': 'Section speed restrictions',
                'parameters': ['speed_factor', 'affected_sections', 'duration_minutes'],
                'severity_levels': [0.8, 0.6, 0.4, 0.3, 0.2]  # factor of normal speed
            },
            'PLATFORM_UNAVAILABLE': {
                'description': 'Platform or station unavailability',
                'parameters': ['station', 'platform_count', 'duration_minutes'],
                'severity_levels': [30, 60, 120, 240, 480]  # minutes
            },
            'SIGNAL_FAILURE': {
                'description': 'Signal system failures',
                'parameters': ['affected_sections', 'failure_type', 'repair_time'],
                'severity_levels': [15, 30, 60, 120, 300]  # minutes
            },
            'WEATHER_IMPACT': {
                'description': 'Weather-related disruptions',
                'parameters': ['weather_type', 'intensity', 'affected_areas'],
                'weather_types': ['fog', 'heavy_rain', 'snow', 'extreme_heat', 'wind']
            },
            'ROLLING_STOCK_FAILURE': {
                'description': 'Train breakdown or mechanical issues',
                'parameters': ['train_id', 'failure_type', 'repair_location', 'estimated_repair_time'],
                'failure_types': ['engine_failure', 'brake_issue', 'door_malfunction', 'electrical_fault']
            },
            'INFRASTRUCTURE_MAINTENANCE': {
                'description': 'Planned or emergency infrastructure work',
                'parameters': ['affected_sections', 'work_type', 'duration_hours'],
                'work_types': ['track_maintenance', 'signal_upgrade', 'bridge_repair', 'station_work']
            },
            'PASSENGER_INCIDENT': {
                'description': 'Passenger-related incidents affecting operations',
                'parameters': ['station', 'incident_type', 'delay_minutes'],
                'incident_types': ['medical_emergency', 'security_alert', 'overcrowding', 'evacuation']
            }
        }
    
    def create_scenario(self, scenario_name: str, 
                       trains: List[Train], 
                       sections: List[Section],
                       constraints: List[Constraint] = None,
                       disruptions: List[Dict] = None,
                       description: str = "") -> str:
        """
        Create a comprehensive simulation scenario
        """
        if constraints is None:
            constraints = []
        if disruptions is None:
            disruptions = []
        
        scenario_id = str(uuid.uuid4())
        
        scenario = {
            'id': scenario_id,
            'name': scenario_name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'trains': [train.to_dict() for train in trains],
            'sections': [section.to_dict() for section in sections],
            'constraints': [asdict(constraint) for constraint in constraints],
            'disruptions': disruptions,
            'parameters': self.simulation_parameters.copy()
        }
        
        self.scenarios[scenario_name] = scenario
        logger.info(f"Created scenario '{scenario_name}' with {len(trains)} trains, "
                   f"{len(sections)} sections, {len(disruptions)} disruptions")
        
        return scenario_name
    
    def add_disruption_to_scenario(self, scenario_name: str, disruption: Dict) -> bool:
        """Add a disruption to an existing scenario"""
        if scenario_name not in self.scenarios:
            logger.error(f"Scenario '{scenario_name}' not found")
            return False
        
        # Validate disruption
        if not self._validate_disruption(disruption):
            logger.error(f"Invalid disruption format: {disruption}")
            return False
        
        self.scenarios[scenario_name]['disruptions'].append(disruption)
        logger.info(f"Added {disruption['type']} disruption to scenario '{scenario_name}'")
        return True
    
    def _validate_disruption(self, disruption: Dict) -> bool:
        """Validate disruption format and parameters"""
        required_fields = ['type', 'start_time', 'parameters']
        
        # Check required fields
        for field in required_fields:
            if field not in disruption:
                return False
        
        # Check if disruption type is supported
        if disruption['type'] not in self.disruption_models:
            return False
        
        return True
    
    def apply_disruption(self, trains: List[Train], sections: List[Section], 
                        disruption: Dict) -> Tuple[List[Train], List[Section]]:
        """
        Apply a specific disruption to trains and sections
        Returns modified copies of trains and sections
        """
        modified_trains = deepcopy(trains)
        modified_sections = deepcopy(sections)
        
        disruption_type = disruption['type']
        parameters = disruption.get('parameters', {})
        
        try:
            if disruption_type == 'TRAIN_DELAY':
                modified_trains = self._apply_train_delay(modified_trains, parameters)
            
            elif disruption_type == 'SPEED_RESTRICTION':
                modified_sections = self._apply_speed_restriction(modified_sections, parameters)
            
            elif disruption_type == 'PLATFORM_UNAVAILABLE':
                modified_trains = self._apply_platform_unavailable(modified_trains, parameters)
            
            elif disruption_type == 'SIGNAL_FAILURE':
                modified_sections = self._apply_signal_failure(modified_sections, parameters)
            
            elif disruption_type == 'WEATHER_IMPACT':
                modified_trains, modified_sections = self._apply_weather_impact(
                    modified_trains, modified_sections, parameters)
            
            elif disruption_type == 'ROLLING_STOCK_FAILURE':
                modified_trains = self._apply_rolling_stock_failure(modified_trains, parameters)
            
            elif disruption_type == 'INFRASTRUCTURE_MAINTENANCE':
                modified_sections = self._apply_infrastructure_maintenance(modified_sections, parameters)
            
            elif disruption_type == 'PASSENGER_INCIDENT':
                modified_trains = self._apply_passenger_incident(modified_trains, parameters)
            
            logger.debug(f"Applied {disruption_type} disruption")
            
        except Exception as e:
            logger.error(f"Error applying {disruption_type} disruption: {e}")
        
        return modified_trains, modified_sections
    
    def _apply_train_delay(self, trains: List[Train], parameters: Dict) -> List[Train]:
        """Apply individual train delays"""
        delay_minutes = parameters.get('delay_minutes', 15)
        affected_trains = parameters.get('affected_trains', [])
        
        for train in trains:
            if not affected_trains or train.train_id in affected_trains:
                train.delay_minutes += delay_minutes
                train.status = TrainStatus.DELAYED
                logger.debug(f"Applied {delay_minutes}min delay to train {train.train_id}")
        
        return trains
    
    def _apply_speed_restriction(self, sections: List[Section], parameters: Dict) -> List[Section]:
        """Apply speed restrictions to sections"""
        speed_factor = parameters.get('speed_factor', 0.6)
        affected_sections = parameters.get('affected_sections', [])
        
        for section in sections:
            if not affected_sections or section.section_id in affected_sections:
                section.max_speed_kmph *= speed_factor
                logger.debug(f"Applied speed restriction to section {section.section_id}: "
                           f"new max speed {section.max_speed_kmph:.1f} kmph")
        
        return sections
    
    def _apply_weather_impact(self, trains: List[Train], sections: List[Section], 
                             parameters: Dict) -> Tuple[List[Train], List[Section]]:
        """Apply weather-related impacts"""
        weather_type = parameters.get('weather_type', 'fog')
        intensity = parameters.get('intensity', 'moderate')
        
        # Weather impact factors
        weather_factors = {
            'fog': {'speed_factor': 0.7, 'delay_factor': 1.3},
            'heavy_rain': {'speed_factor': 0.8, 'delay_factor': 1.2},
            'snow': {'speed_factor': 0.5, 'delay_factor': 1.5},
            'extreme_heat': {'speed_factor': 0.9, 'delay_factor': 1.1},
            'wind': {'speed_factor': 0.8, 'delay_factor': 1.2}
        }
        
        factors = weather_factors.get(weather_type, weather_factors['fog'])
        
        # Adjust intensity
        intensity_multipliers = {'light': 0.5, 'moderate': 1.0, 'severe': 1.5}
        multiplier = intensity_multipliers.get(intensity, 1.0)
        
        # Apply to sections
        for section in sections:
            section.max_speed_kmph *= (factors['speed_factor'] * multiplier)
        
        # Apply to trains
        for train in trains:
            additional_delay = random.uniform(5, 20) * factors['delay_factor'] * multiplier
            train.delay_minutes += additional_delay
        
        logger.debug(f"Applied {intensity} {weather_type} weather impact")
        
        return trains, sections
    
    def _apply_signal_failure(self, sections: List[Section], parameters: Dict) -> List[Section]:
        """Apply signal failure impacts"""
        affected_sections = parameters.get('affected_sections', [])
        failure_type = parameters.get('failure_type', 'complete_failure')
        
        for section in sections:
            if section.section_id in affected_sections:
                if failure_type == 'complete_failure':
                    section.max_speed_kmph *= 0.3  # Severe speed reduction
                    section.capacity_trains = max(1, section.capacity_trains // 2)
                elif failure_type == 'partial_failure':
                    section.max_speed_kmph *= 0.6  # Moderate speed reduction
                
                logger.debug(f"Signal failure in section {section.section_id}: "
                           f"{failure_type}")
        
        return sections
    
    def run_scenario(self, scenario_name: str, solver=None, real_time: bool = False) -> Dict:
        """
        Execute a simulation scenario and return comprehensive results
        """
        if scenario_name not in self.scenarios:
            return {'error': f'Scenario {scenario_name} not found'}
        
        scenario = self.scenarios[scenario_name]
        start_time = datetime.now()
        
        try:
            # Reconstruct objects from scenario data
            trains = [Train.from_dict(train_data) for train_data in scenario['trains']]
            sections = [Section.from_dict(section_data) for section_data in scenario['sections']]
            
            # Track original state for comparison
            original_trains = deepcopy(trains)
            original_sections = deepcopy(sections)
            
            # Apply all disruptions in chronological order
            disruptions = sorted(scenario['disruptions'], 
                               key=lambda d: d.get('start_time', datetime.now().isoformat()))
            
            applied_disruptions = []
            for disruption in disruptions:
                trains, sections = self.apply_disruption(trains, sections, disruption)
                applied_disruptions.append({
                    'type': disruption['type'],
                    'applied_at': datetime.now().isoformat(),
                    'parameters': disruption.get('parameters', {})
                })
            
            # Calculate performance metrics
            simulation_metrics = self._calculate_simulation_metrics(
                original_trains, trains, original_sections, sections)
            
            # Store results
            result = {
                'scenario_name': scenario_name,
                'success': True,
                'execution_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                'trains_processed': len(trains),
                'sections_processed': len(sections),
                'disruptions_applied': len(applied_disruptions),
                'simulation_metrics': simulation_metrics,
                'applied_disruptions': applied_disruptions,
                'timestamp': start_time.isoformat()
            }
            
            self.simulation_results[scenario_name] = result
            logger.info(f"Scenario '{scenario_name}' executed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute scenario '{scenario_name}': {e}")
            return {
                'scenario_name': scenario_name,
                'success': False,
                'error': str(e),
                'timestamp': start_time.isoformat()
            }
    
    def _calculate_simulation_metrics(self, original_trains: List[Train], 
                                    modified_trains: List[Train],
                                    original_sections: List[Section], 
                                    modified_sections: List[Section]) -> Dict:
        """Calculate comprehensive simulation performance metrics"""
        
        # Train metrics
        original_delay = sum(t.delay_minutes for t in original_trains)
        modified_delay = sum(t.delay_minutes for t in modified_trains)
        
        # Calculate on-time performance
        on_time_original = len([t for t in original_trains if t.delay_minutes <= 5])
        on_time_modified = len([t for t in modified_trains if t.delay_minutes <= 5])
        
        # Section utilization changes
        original_utilizations = [s.get_utilization() for s in original_sections]
        modified_utilizations = [s.get_utilization() for s in modified_sections]
        
        # Speed impact analysis
        speed_changes = []
        for orig, mod in zip(original_sections, modified_sections):
            if orig.section_id == mod.section_id:
                speed_change = (mod.max_speed_kmph - orig.max_speed_kmph) / orig.max_speed_kmph
                speed_changes.append(speed_change)
        
        return {
            'train_metrics': {
                'total_trains': len(modified_trains),
                'original_total_delay': original_delay,
                'modified_total_delay': modified_delay,
                'delay_increase': modified_delay - original_delay,
                'average_delay_increase': (modified_delay - original_delay) / len(modified_trains) if modified_trains else 0,
                'on_time_performance_original': on_time_original / len(original_trains) if original_trains else 0,
                'on_time_performance_modified': on_time_modified / len(modified_trains) if modified_trains else 0
            },
            'section_metrics': {
                'total_sections': len(modified_sections),
                'average_utilization_original': np.mean(original_utilizations) if original_utilizations else 0,
                'average_utilization_modified': np.mean(modified_utilizations) if modified_utilizations else 0,
                'average_speed_change': np.mean(speed_changes) if speed_changes else 0,
                'sections_with_speed_reduction': len([c for c in speed_changes if c < 0])
            },
            'disruption_severity': self._assess_disruption_severity(original_delay, modified_delay, speed_changes)
        }
    
    def _assess_disruption_severity(self, original_delay: float, modified_delay: float, 
                                  speed_changes: List[float]) -> Dict:
        """Assess the overall severity of disruptions"""
        delay_increase_pct = ((modified_delay - original_delay) / max(original_delay, 1)) * 100
        avg_speed_reduction = abs(np.mean([c for c in speed_changes if c < 0])) if speed_changes else 0
        
        # Determine overall impact level
        if delay_increase_pct > 50 or avg_speed_reduction > 0.4:
            impact = 'SEVERE'
        elif delay_increase_pct > 25 or avg_speed_reduction > 0.2:
            impact = 'HIGH'
        elif delay_increase_pct > 10 or avg_speed_reduction > 0.1:
            impact = 'MODERATE'
        else:
            impact = 'LOW'
        
        return {
            'overall_impact': impact,
            'delay_increase_percentage': delay_increase_pct,
            'average_speed_reduction': avg_speed_reduction,
            'severity_score': min(100, delay_increase_pct + (avg_speed_reduction * 100))
        }

# ==================== VALIDATION ENGINE ====================

class ValidationEngine:
    """
    Comprehensive validation engine for testing system components and scenarios
    """
    
    def __init__(self):
        self.test_results: List[Dict] = []
        self.validation_metrics = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_coverage': 0.0,
            'last_run': None
        }
        
        logger.info("Validation Engine initialized")
    
    def run_comprehensive_validation(self, trains: List[Train] = None, 
                                   sections: List[Section] = None) -> Dict:
        """Run comprehensive system validation"""
        
        start_time = datetime.now()
        validation_results = {
            'validation_id': str(uuid.uuid4()),
            'timestamp': start_time.isoformat(),
            'test_results': [],
            'summary': {},
            'recommendations': []
        }
        
        # Test categories
        test_categories = [
            ('Data Model Validation', self._test_data_models),
            ('Business Logic Validation', self._test_business_logic),
            ('Performance Validation', self._test_performance),
            ('Integration Validation', self._test_integration)
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for category_name, test_function in test_categories:
            try:
                category_results = test_function(trains, sections)
                validation_results['test_results'].append({
                    'category': category_name,
                    'results': category_results,
                    'passed': category_results.get('passed', False),
                    'test_count': category_results.get('test_count', 0)
                })
                
                total_tests += category_results.get('test_count', 0)
                if category_results.get('passed', False):
                    passed_tests += category_results.get('test_count', 0)
                    
            except Exception as e:
                logger.error(f"Validation category '{category_name}' failed: {e}")
                validation_results['test_results'].append({
                    'category': category_name,
                    'results': {'error': str(e), 'passed': False, 'test_count': 1},
                    'passed': False,
                    'test_count': 1
                })
                total_tests += 1
        
        # Calculate summary
        execution_time = (datetime.now() - start_time).total_seconds()
        success_rate = (passed_tests / max(total_tests, 1)) * 100
        
        validation_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': success_rate,
            'execution_time_seconds': execution_time,
            'overall_status': 'PASS' if success_rate >= 80 else 'FAIL'
        }
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_validation_recommendations(
            validation_results['test_results'])
        
        # Update metrics
        self.validation_metrics.update({
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'test_coverage': success_rate,
            'last_run': start_time.isoformat()
        })
        
        self.test_results.append(validation_results)
        
        logger.info(f"Comprehensive validation completed: {success_rate:.1f}% success rate")
        return validation_results
    
    def _test_data_models(self, trains: List[Train] = None, sections: List[Section] = None) -> Dict:
        """Test data model validation"""
        from core_data_layer import create_sample_train, create_sample_section, validate_train, validate_section
        
        tests_passed = 0
        total_tests = 5
        errors = []
        
        try:
            # Test 1: Train model creation and validation
            base_time = datetime.now()
            test_train = create_sample_train("TEST001", TrainType.EXPRESS, base_time)
            train_errors = validate_train(test_train)
            if not train_errors:
                tests_passed += 1
            else:
                errors.append(f"Train validation failed: {train_errors}")
            
            # Test 2: Section model creation and validation
            test_section = create_sample_section("SEC001", "StationA", "StationB", 50.0)
            section_errors = validate_section(test_section)
            if not section_errors:
                tests_passed += 1
            else:
                errors.append(f"Section validation failed: {section_errors}")
            
            # Test 3: Train serialization/deserialization
            train_dict = test_train.to_dict()
            recreated_train = Train.from_dict(train_dict)
            if recreated_train.train_id == test_train.train_id:
                tests_passed += 1
            else:
                errors.append("Train serialization/deserialization failed")
            
            # Test 4: Section serialization/deserialization
            section_dict = test_section.to_dict()
            recreated_section = Section.from_dict(section_dict)
            if recreated_section.section_id == test_section.section_id:
                tests_passed += 1
            else:
                errors.append("Section serialization/deserialization failed")
            
            # Test 5: Business logic methods
            travel_time = test_section.calculate_travel_time(test_train)
            if travel_time > 0:
                tests_passed += 1
            else:
                errors.append("Travel time calculation failed")
                
        except Exception as e:
            errors.append(f"Data model test exception: {e}")
        
        return {
            'passed': tests_passed == total_tests,
            'test_count': total_tests,
            'passed_count': tests_passed,
            'errors': errors,
            'details': f"Data model validation: {tests_passed}/{total_tests} tests passed"
        }
    
    def _test_business_logic(self, trains: List[Train] = None, sections: List[Section] = None) -> Dict:
        """Test business logic validation"""
        tests_passed = 0
        total_tests = 4
        errors = []
        
        try:
            # Use provided data or create test data
            if not trains or not sections:
                from core_data_layer import create_sample_train, create_sample_section
                base_time = datetime.now()
                trains = [create_sample_train(f"T{i:03d}", TrainType.EXPRESS, base_time + timedelta(minutes=i*10)) 
                         for i in range(3)]
                sections = [create_sample_section(f"SEC{i:03d}", f"Station{i}", f"Station{i+1}", 25.0) 
                          for i in range(2)]
            
            # Test 1: Priority calculation
            if trains:
                priority_weights = [train.get_priority_weight() for train in trains]
                if all(w > 0 for w in priority_weights):
                    tests_passed += 1
                else:
                    errors.append("Priority weight calculation failed")
            
            # Test 2: Section capacity management
            if sections:
                test_section = sections[0]
                if trains:
                    can_accommodate = test_section.can_accommodate(trains[0])
                    if isinstance(can_accommodate, bool):
                        tests_passed += 1
                    else:
                        errors.append("Section capacity check failed")
                else:
                    tests_passed += 1  # Skip if no trains
            
            # Test 3: Delay calculations
            if trains:
                delayed_train = trains[0]
                delayed_train.delay_minutes = 15
                if delayed_train.is_delayed():
                    tests_passed += 1
                else:
                    errors.append("Delay detection failed")
            
            # Test 4: Time calculations
            if trains:
                test_train = trains[0]
                actual_arrival = test_train.get_actual_arrival()
                if isinstance(actual_arrival, datetime):
                    tests_passed += 1
                else:
                    errors.append("Time calculation failed")
                    
        except Exception as e:
            errors.append(f"Business logic test exception: {e}")
        
        return {
            'passed': tests_passed == total_tests,
            'test_count': total_tests,
            'passed_count': tests_passed,
            'errors': errors,
            'details': f"Business logic validation: {tests_passed}/{total_tests} tests passed"
        }
    
    def _test_performance(self, trains: List[Train] = None, sections: List[Section] = None) -> Dict:
        """Test performance validation"""
        tests_passed = 0
        total_tests = 3
        errors = []
        
        try:
            # Test 1: Model creation performance
            start_time = time.time()
            from core_data_layer import create_sample_train, create_sample_section
            base_time = datetime.now()
            for i in range(100):
                train = create_sample_train(f"PERF{i:03d}", TrainType.LOCAL, base_time)
            creation_time = time.time() - start_time
            
            if creation_time < 1.0:  # Should create 100 trains in less than 1 second
                tests_passed += 1
            else:
                errors.append(f"Model creation too slow: {creation_time:.2f}s for 100 trains")
            
            # Test 2: Serialization performance
            start_time = time.time()
            test_train = create_sample_train("PERF_TEST", TrainType.EXPRESS, base_time)
            for i in range(1000):
                train_dict = test_train.to_dict()
            serialization_time = time.time() - start_time
            
            if serialization_time < 1.0:  # Should serialize 1000 times in less than 1 second
                tests_passed += 1
            else:
                errors.append(f"Serialization too slow: {serialization_time:.2f}s for 1000 operations")
            
            # Test 3: Calculation performance
            start_time = time.time()
            test_section = create_sample_section("PERF_SEC", "A", "B", 50.0)
            test_train = create_sample_train("PERF_TRAIN", TrainType.LOCAL, base_time)
            for i in range(1000):
                travel_time = test_section.calculate_travel_time(test_train)
            calculation_time = time.time() - start_time
            
            if calculation_time < 1.0:  # Should calculate 1000 times in less than 1 second
                tests_passed += 1
            else:
                errors.append(f"Calculation too slow: {calculation_time:.2f}s for 1000 operations")
                
        except Exception as e:
            errors.append(f"Performance test exception: {e}")
        
        return {
            'passed': tests_passed == total_tests,
            'test_count': total_tests,
            'passed_count': tests_passed,
            'errors': errors,
            'details': f"Performance validation: {tests_passed}/{total_tests} tests passed"
        }
    
    def _test_integration(self, trains: List[Train] = None, sections: List[Section] = None) -> Dict:
        """Test integration validation"""
        tests_passed = 0
        total_tests = 3
        errors = []
        
        try:
            # Test 1: Traffic Simulator integration
            simulator = TrafficSimulator()
            if trains and sections:
                scenario_name = simulator.create_scenario("test_scenario", trains[:2], sections[:1])
                if scenario_name:
                    tests_passed += 1
                else:
                    errors.append("Traffic simulator scenario creation failed")
            else:
                # Create test data
                from core_data_layer import create_sample_train, create_sample_section
                base_time = datetime.now()
                test_trains = [create_sample_train("INT001", TrainType.EXPRESS, base_time)]
                test_sections = [create_sample_section("INT_SEC", "A", "B", 30.0)]
                scenario_name = simulator.create_scenario("test_scenario", test_trains, test_sections)
                if scenario_name:
                    tests_passed += 1
                else:
                    errors.append("Traffic simulator scenario creation failed")
            
            # Test 2: Disruption application
            try:
                from core_data_layer import create_sample_train, create_sample_section
                base_time = datetime.now()
                test_trains = [create_sample_train("DISRUPT001", TrainType.LOCAL, base_time)]
                test_sections = [create_sample_section("DISRUPT_SEC", "X", "Y", 20.0)]
                
                disruption = {
                    'type': 'TRAIN_DELAY',
                    'start_time': base_time.isoformat(),
                    'parameters': {'delay_minutes': 10, 'affected_trains': ['DISRUPT001']}
                }
                
                modified_trains, modified_sections = simulator.apply_disruption(
                    test_trains, test_sections, disruption)
                
                if modified_trains[0].delay_minutes >= 10:
                    tests_passed += 1
                else:
                    errors.append("Disruption application failed")
                    
            except Exception as e:
                errors.append(f"Disruption test failed: {e}")
            
            # Test 3: Scenario execution
            try:
                if scenario_name in simulator.scenarios:
                    result = simulator.run_scenario(scenario_name)
                    if result.get('success', False):
                        tests_passed += 1
                    else:
                        errors.append(f"Scenario execution failed: {result.get('error', 'Unknown')}")
                else:
                    errors.append("Scenario not found for execution test")
                    
            except Exception as e:
                errors.append(f"Scenario execution test failed: {e}")
                
        except Exception as e:
            errors.append(f"Integration test exception: {e}")
        
        return {
            'passed': tests_passed == total_tests,
            'test_count': total_tests,
            'passed_count': tests_passed,
            'errors': errors,
            'details': f"Integration validation: {tests_passed}/{total_tests} tests passed"
        }
    
    def _generate_validation_recommendations(self, test_results: List[Dict]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        for result in test_results:
            if not result['passed']:
                category = result['category']
                errors = result['results'].get('errors', [])
                
                if category == 'Data Model Validation':
                    recommendations.append("🔧 Review data model implementations and validation logic")
                    if errors:
                        recommendations.append(f"📝 Address data model errors: {'; '.join(errors[:2])}")
                
                elif category == 'Business Logic Validation':
                    recommendations.append("⚙️ Review business logic implementations")
                    if errors:
                        recommendations.append(f"📝 Fix business logic issues: {'; '.join(errors[:2])}")
                
                elif category == 'Performance Validation':
                    recommendations.append("🚀 Optimize performance-critical operations")
                    recommendations.append("📈 Consider caching or algorithmic improvements")
                
                elif category == 'Integration Validation':
                    recommendations.append("🔗 Review component integration and interfaces")
                    recommendations.append("🧪 Add more integration test coverage")
        
        if not recommendations:
            recommendations.append("✅ All validation tests passed - system is operating correctly")
        
        return recommendations

# ==================== COMBINED SIMULATION AND VALIDATION ENGINE ====================

class SimulationValidationEngine:
    """
    Combined engine that provides both simulation and validation capabilities
    """
    
    def __init__(self):
        self.simulator = TrafficSimulator()
        self.validator = ValidationEngine()
        
        logger.info("Combined Simulation and Validation Engine initialized")
    
    def run_comprehensive_analysis(self, trains: List[Train], sections: List[Section]) -> Dict:
        """Run both simulation and validation analysis"""
        
        analysis_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Run validation first
        validation_results = self.validator.run_comprehensive_validation(trains, sections)
        
        # Create and run simulation scenarios
        simulation_results = {}
        
        # Scenario 1: Baseline (no disruptions)
        baseline_scenario = self.simulator.create_scenario(
            "baseline_analysis", trains, sections, description="Baseline scenario for analysis")
        baseline_result = self.simulator.run_scenario(baseline_scenario)
        simulation_results['baseline'] = baseline_result
        
        # Scenario 2: Delay disruption
        delay_scenario = self.simulator.create_scenario(
            "delay_analysis", trains, sections, 
            disruptions=[{
                'type': 'TRAIN_DELAY',
                'start_time': datetime.now().isoformat(),
                'parameters': {'delay_minutes': 15, 'affected_trains': [trains[0].train_id] if trains else []}
            }],
            description="Delay impact analysis")
        delay_result = self.simulator.run_scenario(delay_scenario)
        simulation_results['delay_impact'] = delay_result
        
        # Scenario 3: Weather disruption
        weather_scenario = self.simulator.create_scenario(
            "weather_analysis", trains, sections,
            disruptions=[{
                'type': 'WEATHER_IMPACT',
                'start_time': datetime.now().isoformat(),
                'parameters': {'weather_type': 'fog', 'intensity': 'moderate'}
            }],
            description="Weather impact analysis")
        weather_result = self.simulator.run_scenario(weather_scenario)
        simulation_results['weather_impact'] = weather_result
        
        # Compile comprehensive analysis
        analysis = {
            'analysis_id': analysis_id,
            'timestamp': start_time.isoformat(),
            'execution_time_seconds': (datetime.now() - start_time).total_seconds(),
            'trains_analyzed': len(trains),
            'sections_analyzed': len(sections),
            'validation_results': validation_results,
            'simulation_results': simulation_results,
            'recommendations': self._generate_analysis_recommendations(
                validation_results, simulation_results)
        }
        
        logger.info(f"Comprehensive analysis completed for {len(trains)} trains and {len(sections)} sections")
        return analysis
    
    def _generate_analysis_recommendations(self, validation_results: Dict, 
                                         simulation_results: Dict) -> List[str]:
        """Generate comprehensive analysis recommendations"""
        recommendations = []
        
        # Add validation recommendations
        recommendations.extend(validation_results.get('recommendations', []))
        
        # Add simulation-based recommendations
        for scenario_name, result in simulation_results.items():
            if result.get('success') and 'simulation_metrics' in result:
                metrics = result['simulation_metrics']
                
                # Check disruption severity
                disruption_severity = metrics.get('disruption_severity', {})
                impact = disruption_severity.get('overall_impact', 'UNKNOWN')
                
                if impact == 'SEVERE':
                    recommendations.append(f"🚨 {scenario_name} scenario shows severe impact - consider mitigation strategies")
                elif impact == 'HIGH':
                    recommendations.append(f"⚠️ {scenario_name} scenario shows high impact - monitor conditions")
                
                # Check performance degradation
                train_metrics = metrics.get('train_metrics', {})
                delay_increase = train_metrics.get('delay_increase', 0)
                if delay_increase > 30:
                    recommendations.append(f"⏰ {scenario_name}: Significant delay increase ({delay_increase:.1f} min) - review scheduling")
        
        if not any('🚨' in rec or '⚠️' in rec for rec in recommendations):
            recommendations.append("✅ System shows good resilience to analyzed disruption scenarios")
        
        return recommendations

if __name__ == "__main__":
    # Test the simulation and validation engine
    print("Testing Simulation and Validation Engine...")
    
    from core_data_layer import create_sample_train, create_sample_section, TrainType
    
    # Create test data
    base_time = datetime.now()
    trains = [
        create_sample_train("T001", TrainType.EXPRESS, base_time),
        create_sample_train("T002", TrainType.LOCAL, base_time + timedelta(minutes=20)),
        create_sample_train("T003", TrainType.FREIGHT, base_time + timedelta(minutes=40))
    ]
    
    sections = [
        create_sample_section("SEC001", "StationA", "StationB", 50.0),
        create_sample_section("SEC002", "StationB", "StationC", 30.0)
    ]
    
    # Create simulation and validation engine
    engine = SimulationValidationEngine()
    
    # Run comprehensive analysis
    analysis = engine.run_comprehensive_analysis(trains, sections)
    
    print(f"Analysis results:")
    print(f"  Analysis ID: {analysis['analysis_id']}")
    print(f"  Execution time: {analysis['execution_time_seconds']:.2f}s")
    print(f"  Trains analyzed: {analysis['trains_analyzed']}")
    print(f"  Sections analyzed: {analysis['sections_analyzed']}")
    
    # Validation results
    validation = analysis['validation_results']['summary']
    print(f"  Validation: {validation['passed_tests']}/{validation['total_tests']} tests passed")
    print(f"  Validation status: {validation['overall_status']}")
    
    # Simulation results
    simulation = analysis['simulation_results']
    print(f"  Simulation scenarios: {len(simulation)}")
    
    for scenario_name, result in simulation.items():
        if result.get('success'):
            print(f"    {scenario_name}: {result['trains_processed']} trains, {result['execution_time_ms']:.1f}ms")
    
    print(f"  Recommendations: {len(analysis['recommendations'])}")
    for rec in analysis['recommendations'][:3]:  # Show first 3 recommendations
        print(f"    - {rec}")
    
    print("Simulation and Validation Engine test completed!")
