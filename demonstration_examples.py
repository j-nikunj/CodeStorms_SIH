# MERGED DELIVERABLE 6: DEMONSTRATION AND EXAMPLES
# File: demonstration_examples.py

"""
Demonstration and Examples Module for Intelligent Train Traffic Control System
Combines all demo scripts and example scenarios into a comprehensive demonstration suite
"""

import json
import time
import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import threading

# Import our main components
from core_data_layer import Train, Section, TrainType, TrainStatus, create_sample_train, create_sample_section
from ai_decision_engine import RealTimeDecisionEngine
from simulation_validation_engine import SimulationValidationEngine
from web_api_services import WebServices
from operations_utilities import OperationsOrchestrator

logger = logging.getLogger(__name__)

class ScenarioRunner:
    """Runs various demonstration scenarios"""
    
    def __init__(self):
        self.decision_engine = RealTimeDecisionEngine(use_rl=True, use_simulation=False)
        self.sim_val_engine = SimulationValidationEngine()
        self.web_services = WebServices()
        self.operations = OperationsOrchestrator()
        
        self.scenario_results = []
        
        logger.info("Scenario Runner initialized")
    
    def run_basic_scheduling_demo(self) -> Dict[str, Any]:
        """Demonstrate basic train scheduling functionality"""
        print("🚄 Running Basic Scheduling Demonstration...")
        
        scenario_start = datetime.now()
        results = {
            'scenario': 'Basic Scheduling Demo',
            'start_time': scenario_start.isoformat(),
            'status': 'RUNNING',
            'details': {}
        }
        
        try:
            # Step 1: Create sample data
            print("  Step 1: Setting up railway network...")
            
            base_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
            
            # Create trains
            trains = [
                Train("EXPRESS001", TrainType.EXPRESS, 1, base_time, base_time + timedelta(minutes=10),
                     "Mumbai Central", "New Delhi", 0, TrainStatus.SCHEDULED, 120.0, 400.0, 1200, ["SEC001", "SEC002"]),
                Train("LOCAL001", TrainType.LOCAL, 3, base_time + timedelta(minutes=5), 
                     base_time + timedelta(minutes=7), "Mumbai Central", "Borivali", 2, TrainStatus.DELAYED, 80.0, 300.0, 1500, ["SEC001"]),
                Train("FREIGHT001", TrainType.FREIGHT, 4, base_time + timedelta(minutes=30), 
                     base_time + timedelta(minutes=35), "Borivali", "Virar", 0, TrainStatus.SCHEDULED, 60.0, 800.0, 0, ["SEC002"])
            ]
            
            # Create sections
            sections = [
                Section("SEC001", "Mumbai Central", "Borivali", 25.0, 100.0, 4, True, 0.5, 2.0, True),
                Section("SEC002", "Borivali", "Virar", 18.0, 80.0, 2, False, 1.2, 1.5, True)
            ]
            
            # Add to decision engine
            for section in sections:
                self.decision_engine.cp_solver.add_section(section)
            
            for train in trains:
                self.decision_engine.cp_solver.add_train(train)
            
            print(f"    ✅ Added {len(trains)} trains and {len(sections)} sections")
            results['details']['trains_count'] = len(trains)
            results['details']['sections_count'] = len(sections)
            
            # Step 2: Make scheduling decisions
            print("  Step 2: Making scheduling decisions...")
            
            decisions = []
            for section in sections:
                print(f"    Processing section: {section.section_id}")
                
                decision = self.decision_engine.make_decision(
                    section_id=section.section_id,
                    trains=trains,
                    sections=[section],
                    optimization_objective='minimize_delay'
                )
                
                decisions.append(decision)
                print(f"    Decision confidence: {decision.get('confidence', 0.0):.2f}")
            
            results['details']['decisions'] = len(decisions)
            results['details']['avg_confidence'] = sum(d.get('confidence', 0.0) for d in decisions) / len(decisions)
            
            # Step 3: System status check
            print("  Step 3: Checking system status...")
            status = self.decision_engine.get_system_status()
            print(f"    System health: {status['system_health']}")
            print(f"    Total decisions made: {status['performance_metrics']['total_decisions']}")
            
            results['details']['system_status'] = status
            results['status'] = 'COMPLETED'
            results['end_time'] = datetime.now().isoformat()
            results['duration_seconds'] = (datetime.now() - scenario_start).total_seconds()
            
            print("  ✅ Basic Scheduling Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Basic scheduling demo failed: {e}")
            results['status'] = 'FAILED'
            results['error'] = str(e)
            results['end_time'] = datetime.now().isoformat()
            print(f"  ❌ Demo failed: {e}")
        
        return results
    
    def run_disruption_handling_demo(self) -> Dict[str, Any]:
        """Demonstrate handling of various disruptions"""
        print("🚨 Running Disruption Handling Demonstration...")
        
        scenario_start = datetime.now()
        results = {
            'scenario': 'Disruption Handling Demo',
            'start_time': scenario_start.isoformat(),
            'status': 'RUNNING',
            'details': {}
        }
        
        try:
            # Step 1: Set up normal operations
            print("  Step 1: Setting up normal operations...")
            
            base_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
            trains = [
                create_sample_train("NORMAL001", base_time),
                create_sample_train("NORMAL002", base_time + timedelta(minutes=10)),
                create_sample_train("NORMAL003", base_time + timedelta(minutes=20))
            ]
            
            sections = [
                create_sample_section("MAIN001"),
                create_sample_section("MAIN002")
            ]
            
            # Add to system
            for section in sections:
                self.decision_engine.cp_solver.add_section(section)
            
            for train in trains:
                self.decision_engine.cp_solver.add_train(train)
            
            print("    ✅ Normal operations established")
            
            # Step 2: Create and simulate disruptions
            print("  Step 2: Simulating various disruptions...")
            
            disruption_scenarios = [
                {
                    'type': 'DELAY',
                    'description': 'Train delay due to signal failure',
                    'affected_train': trains[0].train_id,
                    'delay_minutes': 15
                },
                {
                    'type': 'SPEED_RESTRICTION',
                    'description': 'Speed restriction on track section',
                    'affected_section': sections[0].section_id,
                    'reduced_speed': 40.0
                },
                {
                    'type': 'PLATFORM_UNAVAILABLE',
                    'description': 'Platform maintenance',
                    'affected_platform': 2
                }
            ]
            
            disruption_results = []
            
            for i, disruption in enumerate(disruption_scenarios, 1):
                print(f"    Disruption {i}: {disruption['description']}")
                
                # Apply disruption
                if disruption['type'] == 'DELAY':
                    train = next((t for t in trains if t.train_id == disruption['affected_train']), None)
                    if train:
                        train.delay_minutes = disruption['delay_minutes']
                        train.status = TrainStatus.DELAYED
                        print(f"      Applied {disruption['delay_minutes']} min delay to {train.train_id}")
                
                elif disruption['type'] == 'SPEED_RESTRICTION':
                    section = next((s for s in sections if s.section_id == disruption['affected_section']), None)
                    if section:
                        original_speed = section.max_speed
                        section.max_speed = disruption['reduced_speed']
                        print(f"      Reduced speed from {original_speed} to {disruption['reduced_speed']} km/h")
                
                # Make decision with disruption
                decision = self.decision_engine.make_decision(
                    section_id=sections[0].section_id,
                    trains=trains,
                    sections=sections,
                    emergency_mode=(disruption['type'] == 'DELAY' and disruption['delay_minutes'] > 10)
                )
                
                disruption_results.append({
                    'disruption': disruption,
                    'decision': decision,
                    'handled': decision.get('success', False)
                })
                
                print(f"      Decision confidence: {decision.get('confidence', 0.0):.2f}")
                print(f"      Emergency mode: {decision.get('emergency_mode', False)}")
            
            results['details']['disruptions'] = disruption_results
            results['details']['disruptions_handled'] = sum(1 for dr in disruption_results if dr['handled'])
            
            # Step 3: Recovery assessment
            print("  Step 3: Assessing system recovery...")
            
            recovery_status = self.decision_engine.get_system_status()
            recovery_health = recovery_status['system_health']
            
            print(f"    System health after disruptions: {recovery_health}")
            
            results['details']['recovery_status'] = recovery_status
            results['status'] = 'COMPLETED'
            results['end_time'] = datetime.now().isoformat()
            results['duration_seconds'] = (datetime.now() - scenario_start).total_seconds()
            
            print("  ✅ Disruption Handling Demo completed!")
            
        except Exception as e:
            logger.error(f"Disruption handling demo failed: {e}")
            results['status'] = 'FAILED'
            results['error'] = str(e)
            results['end_time'] = datetime.now().isoformat()
            print(f"  ❌ Demo failed: {e}")
        
        return results
    
    def run_ai_learning_demo(self) -> Dict[str, Any]:
        """Demonstrate AI learning and adaptation"""
        print("🧠 Running AI Learning Demonstration...")
        
        scenario_start = datetime.now()
        results = {
            'scenario': 'AI Learning Demo',
            'start_time': scenario_start.isoformat(),
            'status': 'RUNNING',
            'details': {}
        }
        
        try:
            # Step 1: Initialize learning environment
            print("  Step 1: Setting up learning environment...")
            
            # Create a series of scenarios for the AI to learn from
            learning_scenarios = []
            
            for i in range(5):
                base_time = datetime.now().replace(hour=8+i, minute=0, second=0, microsecond=0)
                
                scenario_trains = [
                    Train(f"LEARN{i:02d}A", TrainType.EXPRESS, 1, base_time, 
                         base_time + timedelta(minutes=10), "Station A", "Station B", 
                         random.randint(0, 20), TrainStatus.RUNNING, 120.0, 400.0, 1200, ["LEARN_SEC"]),
                    Train(f"LEARN{i:02d}B", TrainType.LOCAL, 2, base_time + timedelta(minutes=5), 
                         base_time + timedelta(minutes=8), "Station A", "Station C", 
                         random.randint(0, 10), TrainStatus.SCHEDULED, 80.0, 300.0, 1500, ["LEARN_SEC"])
                ]
                
                learning_scenarios.append({
                    'scenario_id': i,
                    'trains': scenario_trains,
                    'complexity': 'low' if i < 2 else 'medium' if i < 4 else 'high'
                })
            
            print(f"    ✅ Created {len(learning_scenarios)} learning scenarios")
            
            # Step 2: Train the AI system
            print("  Step 2: Training AI decision system...")
            
            learning_results = []
            initial_performance = None
            
            for scenario in learning_scenarios:
                print(f"    Training on scenario {scenario['scenario_id']} ({scenario['complexity']} complexity)")
                
                # Add scenario data to system
                section = Section("LEARN_SEC", "Station A", "Station B", 20.0, 100.0, 2, True, 0.0, 1.0, True)
                self.decision_engine.cp_solver.add_section(section)
                
                for train in scenario['trains']:
                    self.decision_engine.cp_solver.add_train(train)
                
                # Make multiple decisions to simulate learning
                scenario_decisions = []
                for j in range(3):  # 3 decision cycles per scenario
                    decision = self.decision_engine.make_decision(
                        section_id="LEARN_SEC",
                        trains=scenario['trains'],
                        sections=[section],
                        optimization_objective='minimize_delay'
                    )
                    scenario_decisions.append(decision)
                
                # Calculate performance metrics
                avg_confidence = sum(d.get('confidence', 0.0) for d in scenario_decisions) / len(scenario_decisions)
                success_rate = sum(1 for d in scenario_decisions if d.get('success', False)) / len(scenario_decisions)
                
                learning_result = {
                    'scenario_id': scenario['scenario_id'],
                    'avg_confidence': avg_confidence,
                    'success_rate': success_rate,
                    'decisions_count': len(scenario_decisions)
                }
                
                learning_results.append(learning_result)
                
                if initial_performance is None:
                    initial_performance = avg_confidence
                
                print(f"      Performance: {avg_confidence:.2f} confidence, {success_rate:.2f} success rate")
            
            # Step 3: Analyze learning progress
            print("  Step 3: Analyzing learning progress...")
            
            final_performance = learning_results[-1]['avg_confidence'] if learning_results else 0.0
            improvement = final_performance - initial_performance if initial_performance else 0.0
            
            print(f"    Initial performance: {initial_performance:.2f}")
            print(f"    Final performance: {final_performance:.2f}")
            print(f"    Improvement: {improvement:.2f}")
            
            results['details']['learning_scenarios'] = len(learning_scenarios)
            results['details']['learning_results'] = learning_results
            results['details']['initial_performance'] = initial_performance
            results['details']['final_performance'] = final_performance
            results['details']['improvement'] = improvement
            
            # Step 4: Test learned behavior
            print("  Step 4: Testing learned behavior...")
            
            test_scenario = {
                'trains': [
                    Train("TEST001", TrainType.EXPRESS, 1, datetime.now(), 
                         datetime.now() + timedelta(minutes=10), "Test A", "Test B", 
                         25, TrainStatus.DELAYED, 120.0, 400.0, 1200, ["TEST_SEC"])
                ],
                'section': Section("TEST_SEC", "Test A", "Test B", 30.0, 120.0, 3, True, 0.0, 1.0, True)
            }
            
            # Add test data
            self.decision_engine.cp_solver.add_section(test_scenario['section'])
            for train in test_scenario['trains']:
                self.decision_engine.cp_solver.add_train(train)
            
            test_decision = self.decision_engine.make_decision(
                section_id="TEST_SEC",
                trains=test_scenario['trains'],
                sections=[test_scenario['section']],
                optimization_objective='minimize_delay'
            )
            
            print(f"    Test decision confidence: {test_decision.get('confidence', 0.0):.2f}")
            print(f"    Test decision success: {test_decision.get('success', False)}")
            
            results['details']['test_decision'] = test_decision
            results['status'] = 'COMPLETED'
            results['end_time'] = datetime.now().isoformat()
            results['duration_seconds'] = (datetime.now() - scenario_start).total_seconds()
            
            print("  ✅ AI Learning Demo completed!")
            
        except Exception as e:
            logger.error(f"AI learning demo failed: {e}")
            results['status'] = 'FAILED'
            results['error'] = str(e)
            results['end_time'] = datetime.now().isoformat()
            print(f"  ❌ Demo failed: {e}")
        
        return results
    
    def run_real_time_monitoring_demo(self) -> Dict[str, Any]:
        """Demonstrate real-time monitoring and dashboard functionality"""
        print("📊 Running Real-Time Monitoring Demonstration...")
        
        scenario_start = datetime.now()
        results = {
            'scenario': 'Real-Time Monitoring Demo',
            'start_time': scenario_start.isoformat(),
            'status': 'RUNNING',
            'details': {}
        }
        
        try:
            # Step 1: Set up monitoring environment
            print("  Step 1: Setting up monitoring environment...")
            
            # Create active trains
            base_time = datetime.now()
            active_trains = []
            
            for i in range(8):  # Create 8 active trains
                train = Train(
                    f"MONITOR{i:02d}",
                    random.choice([TrainType.EXPRESS, TrainType.LOCAL, TrainType.FREIGHT]),
                    random.randint(1, 6),
                    base_time + timedelta(minutes=i*5),
                    base_time + timedelta(minutes=i*5 + 10),
                    f"Station {chr(65+i)}",  # Station A, B, C, etc.
                    f"Station {chr(90-i)}",  # Station Z, Y, X, etc.
                    random.randint(0, 30),
                    random.choice([TrainStatus.RUNNING, TrainStatus.DELAYED, TrainStatus.SCHEDULED]),
                    random.uniform(60, 120),
                    random.uniform(300, 500),
                    random.randint(0, 2000),
                    [f"MON_SEC{i%3}"]
                )
                active_trains.append(train)
            
            # Create monitoring sections
            monitoring_sections = [
                Section(f"MON_SEC{i}", f"Hub {i}", f"Hub {i+1}", 
                       random.uniform(15, 35), random.uniform(80, 120), 
                       random.randint(2, 4), True, random.uniform(0, 2), 
                       random.uniform(1, 3), True) 
                for i in range(3)
            ]
            
            # Add to system
            for section in monitoring_sections:
                self.decision_engine.cp_solver.add_section(section)
            
            for train in active_trains:
                self.decision_engine.cp_solver.add_train(train)
            
            print(f"    ✅ Monitoring {len(active_trains)} trains across {len(monitoring_sections)} sections")
            
            # Step 2: Start real-time monitoring
            print("  Step 2: Starting real-time monitoring (10 second simulation)...")
            
            monitoring_data = []
            monitoring_duration = 10  # seconds
            monitoring_interval = 2  # seconds
            
            for cycle in range(monitoring_duration // monitoring_interval):
                cycle_start = time.time()
                print(f"    Monitoring cycle {cycle + 1}...")
                
                # Get dashboard data
                dashboard_response = self.web_services.api.get_dashboard_data()
                
                if dashboard_response['success']:
                    dashboard_data = dashboard_response['data']
                    
                    cycle_data = {
                        'cycle': cycle + 1,
                        'timestamp': datetime.now().isoformat(),
                        'key_metrics': dashboard_data['key_metrics'],
                        'system_status': dashboard_data['system_status']['system_health']
                    }
                    
                    monitoring_data.append(cycle_data)
                    
                    # Display key metrics
                    metrics = dashboard_data['key_metrics']
                    print(f"      Total trains: {metrics['total_trains']}")
                    print(f"      Delayed trains: {metrics['delayed_trains']}")
                    print(f"      On-time performance: {metrics['on_time_percentage']:.1f}%")
                    print(f"      Average delay: {metrics['average_delay']:.1f} minutes")
                
                # Simulate some dynamic changes
                if cycle == 2:  # Introduce a delay in cycle 3
                    if active_trains:
                        train_to_delay = random.choice(active_trains)
                        train_to_delay.delay_minutes += 15
                        train_to_delay.status = TrainStatus.DELAYED
                        print(f"      🚨 Introduced delay to {train_to_delay.train_id}")
                
                # Wait for next cycle
                cycle_time = time.time() - cycle_start
                if cycle_time < monitoring_interval:
                    time.sleep(monitoring_interval - cycle_time)
            
            print("    ✅ Real-time monitoring simulation completed")
            
            # Step 3: System health check
            print("  Step 3: Performing comprehensive system health check...")
            
            health_check = self.operations.run_full_system_check()
            print(f"    Overall system status: {health_check['overall_status']}")
            
            for component, status in health_check['components'].items():
                print(f"      {component}: {status.get('status', 'UNKNOWN')}")
            
            # Step 4: Generate monitoring report
            print("  Step 4: Generating monitoring report...")
            
            if monitoring_data:
                total_cycles = len(monitoring_data)
                avg_on_time = sum(cycle['key_metrics']['on_time_percentage'] for cycle in monitoring_data) / total_cycles
                avg_delay = sum(cycle['key_metrics']['average_delay'] for cycle in monitoring_data) / total_cycles
                
                monitoring_report = {
                    'monitoring_duration_seconds': monitoring_duration,
                    'monitoring_cycles': total_cycles,
                    'average_on_time_performance': avg_on_time,
                    'average_delay_minutes': avg_delay,
                    'system_health_checks': [cycle['system_status'] for cycle in monitoring_data]
                }
                
                print(f"    📈 Monitoring Report:")
                print(f"      Duration: {monitoring_duration} seconds")
                print(f"      Cycles: {total_cycles}")
                print(f"      Avg on-time performance: {avg_on_time:.1f}%")
                print(f"      Avg delay: {avg_delay:.1f} minutes")
                
                results['details']['monitoring_data'] = monitoring_data
                results['details']['monitoring_report'] = monitoring_report
            
            results['details']['health_check'] = health_check
            results['status'] = 'COMPLETED'
            results['end_time'] = datetime.now().isoformat()
            results['duration_seconds'] = (datetime.now() - scenario_start).total_seconds()
            
            print("  ✅ Real-Time Monitoring Demo completed!")
            
        except Exception as e:
            logger.error(f"Real-time monitoring demo failed: {e}")
            results['status'] = 'FAILED'
            results['error'] = str(e)
            results['end_time'] = datetime.now().isoformat()
            print(f"  ❌ Demo failed: {e}")
        
        return results
    
    def run_comprehensive_system_demo(self) -> Dict[str, Any]:
        """Run a comprehensive demonstration of all system capabilities"""
        print("🌟 Running Comprehensive System Demonstration...")
        
        scenario_start = datetime.now()
        results = {
            'scenario': 'Comprehensive System Demo',
            'start_time': scenario_start.isoformat(),
            'status': 'RUNNING',
            'details': {},
            'sub_demos': []
        }
        
        try:
            print("  This demo will run all individual demonstrations in sequence")
            print("  Total estimated time: ~2-3 minutes")
            print()
            
            # Run all individual demos
            demo_methods = [
                self.run_basic_scheduling_demo,
                self.run_disruption_handling_demo,
                self.run_ai_learning_demo,
                self.run_real_time_monitoring_demo
            ]
            
            successful_demos = 0
            
            for i, demo_method in enumerate(demo_methods, 1):
                print(f"Running sub-demo {i}/{len(demo_methods)}...")
                
                try:
                    demo_result = demo_method()
                    results['sub_demos'].append(demo_result)
                    
                    if demo_result['status'] == 'COMPLETED':
                        successful_demos += 1
                        print(f"  ✅ Sub-demo {i} completed successfully")
                    else:
                        print(f"  ⚠️ Sub-demo {i} completed with issues")
                    
                    # Brief pause between demos
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"  ❌ Sub-demo {i} failed: {e}")
                    results['sub_demos'].append({
                        'scenario': f'Sub-demo {i}',
                        'status': 'FAILED',
                        'error': str(e)
                    })
                
                print()
            
            # Final system validation
            print("  Running final system validation...")
            
            final_validation = self.sim_val_engine.run_comprehensive_analysis(
                self.decision_engine.cp_solver.trains,
                self.decision_engine.cp_solver.sections
            )
            
            print(f"    Validation successful: {final_validation.get('validation_success', False)}")
            print(f"    Simulation successful: {final_validation.get('simulation_success', False)}")
            
            # Summary
            results['details']['total_sub_demos'] = len(demo_methods)
            results['details']['successful_sub_demos'] = successful_demos
            results['details']['success_rate'] = successful_demos / len(demo_methods) * 100
            results['details']['final_validation'] = final_validation
            
            results['status'] = 'COMPLETED' if successful_demos == len(demo_methods) else 'PARTIAL'
            results['end_time'] = datetime.now().isoformat()
            results['duration_seconds'] = (datetime.now() - scenario_start).total_seconds()
            
            print(f"🎯 Comprehensive Demo Summary:")
            print(f"   Total sub-demos: {len(demo_methods)}")
            print(f"   Successful: {successful_demos}")
            print(f"   Success rate: {results['details']['success_rate']:.1f}%")
            print(f"   Total duration: {results['duration_seconds']:.1f} seconds")
            print()
            print("✅ Comprehensive System Demonstration completed!")
            
        except Exception as e:
            logger.error(f"Comprehensive system demo failed: {e}")
            results['status'] = 'FAILED'
            results['error'] = str(e)
            results['end_time'] = datetime.now().isoformat()
            print(f"  ❌ Comprehensive demo failed: {e}")
        
        return results

class InteractiveDemoRunner:
    """Interactive demonstration runner with user choices"""
    
    def __init__(self):
        self.scenario_runner = ScenarioRunner()
        self.available_demos = {
            '1': ('Basic Scheduling Demo', self.scenario_runner.run_basic_scheduling_demo),
            '2': ('Disruption Handling Demo', self.scenario_runner.run_disruption_handling_demo),
            '3': ('AI Learning Demo', self.scenario_runner.run_ai_learning_demo),
            '4': ('Real-Time Monitoring Demo', self.scenario_runner.run_real_time_monitoring_demo),
            '5': ('Comprehensive System Demo', self.scenario_runner.run_comprehensive_system_demo)
        }
    
    def show_menu(self):
        """Display the interactive menu"""
        print("\n" + "="*60)
        print("🚄 INTELLIGENT TRAIN TRAFFIC CONTROL SYSTEM")
        print("   Interactive Demonstration Suite")
        print("="*60)
        print()
        print("Available Demonstrations:")
        
        for key, (name, _) in self.available_demos.items():
            print(f"  {key}. {name}")
        
        print("  0. Exit")
        print()
    
    def run_interactive(self):
        """Run interactive demonstration"""
        print("Welcome to the Interactive Demonstration Suite!")
        
        while True:
            self.show_menu()
            
            try:
                choice = input("Select a demonstration (0-5): ").strip()
                
                if choice == '0':
                    print("\nThank you for using the demonstration suite!")
                    break
                
                if choice in self.available_demos:
                    demo_name, demo_method = self.available_demos[choice]
                    print(f"\nStarting: {demo_name}")
                    print("-" * (len(demo_name) + 10))
                    
                    result = demo_method()
                    
                    print(f"\nDemo Result: {result['status']}")
                    if result['status'] == 'COMPLETED':
                        print("✅ Demo completed successfully!")
                    else:
                        print("⚠️ Demo completed with issues or failed")
                    
                    input("\nPress Enter to continue...")
                    
                else:
                    print("Invalid choice. Please select 0-5.")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n\nDemo interrupted by user.")
                break
            except Exception as e:
                print(f"\nError running demo: {e}")
                input("Press Enter to continue...")

class BenchmarkRunner:
    """Performance benchmarking and stress testing"""
    
    def __init__(self):
        self.decision_engine = RealTimeDecisionEngine(use_rl=True, use_simulation=False)
    
    def run_performance_benchmark(self, num_trains: int = 50, num_sections: int = 10) -> Dict[str, Any]:
        """Run performance benchmark with specified scale"""
        print(f"🏁 Running Performance Benchmark ({num_trains} trains, {num_sections} sections)...")
        
        benchmark_start = datetime.now()
        results = {
            'benchmark': 'Performance Test',
            'scale': {'trains': num_trains, 'sections': num_sections},
            'start_time': benchmark_start.isoformat(),
            'status': 'RUNNING'
        }
        
        try:
            # Step 1: Generate test data
            print("  Generating test data...")
            
            base_time = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
            
            # Generate trains
            trains = []
            for i in range(num_trains):
                train = Train(
                    f"BENCH{i:04d}",
                    random.choice([TrainType.EXPRESS, TrainType.LOCAL, TrainType.FREIGHT]),
                    random.randint(1, 8),
                    base_time + timedelta(minutes=i*2),
                    base_time + timedelta(minutes=i*2 + random.randint(5, 15)),
                    f"Station {i%20}",
                    f"Destination {(i+10)%20}",
                    random.randint(0, 45),
                    random.choice(list(TrainStatus)),
                    random.uniform(40, 130),
                    random.uniform(200, 500),
                    random.randint(0, 2500),
                    [f"SEC{j}" for j in random.sample(range(num_sections), random.randint(1, 3))]
                )
                trains.append(train)
            
            # Generate sections
            sections = []
            for i in range(num_sections):
                section = Section(
                    f"SEC{i}",
                    f"Hub {i}",
                    f"Hub {(i+1)%num_sections}",
                    random.uniform(10, 50),
                    random.uniform(60, 140),
                    random.randint(2, 6),
                    random.choice([True, False]),
                    random.uniform(0, 3),
                    random.uniform(0.5, 3.0),
                    True
                )
                sections.append(section)
            
            # Add to system
            for section in sections:
                self.decision_engine.cp_solver.add_section(section)
            
            for train in trains:
                self.decision_engine.cp_solver.add_train(train)
            
            print(f"    ✅ Generated {len(trains)} trains and {len(sections)} sections")
            
            # Step 2: Performance testing
            print("  Running performance tests...")
            
            decision_times = []
            successful_decisions = 0
            
            test_sections = random.sample(sections, min(5, len(sections)))  # Test 5 random sections
            
            for i, section in enumerate(test_sections):
                print(f"    Testing section {i+1}/{len(test_sections)}: {section.section_id}")
                
                decision_start = time.time()
                
                decision = self.decision_engine.make_decision(
                    section_id=section.section_id,
                    trains=trains,
                    sections=[section],
                    optimization_objective='balanced'
                )
                
                decision_time = (time.time() - decision_start) * 1000  # Convert to milliseconds
                decision_times.append(decision_time)
                
                if decision.get('success', False):
                    successful_decisions += 1
                
                print(f"      Decision time: {decision_time:.2f}ms")
                print(f"      Confidence: {decision.get('confidence', 0.0):.2f}")
            
            # Calculate performance metrics
            avg_decision_time = sum(decision_times) / len(decision_times) if decision_times else 0
            max_decision_time = max(decision_times) if decision_times else 0
            min_decision_time = min(decision_times) if decision_times else 0
            success_rate = successful_decisions / len(test_sections) * 100
            
            benchmark_duration = (datetime.now() - benchmark_start).total_seconds()
            
            results.update({
                'status': 'COMPLETED',
                'end_time': datetime.now().isoformat(),
                'duration_seconds': benchmark_duration,
                'performance_metrics': {
                    'avg_decision_time_ms': avg_decision_time,
                    'max_decision_time_ms': max_decision_time,
                    'min_decision_time_ms': min_decision_time,
                    'success_rate_percent': success_rate,
                    'total_decisions': len(test_sections),
                    'successful_decisions': successful_decisions,
                    'throughput_decisions_per_second': len(test_sections) / benchmark_duration
                }
            })
            
            print(f"  📊 Benchmark Results:")
            print(f"    Average decision time: {avg_decision_time:.2f}ms")
            print(f"    Max decision time: {max_decision_time:.2f}ms")
            print(f"    Success rate: {success_rate:.1f}%")
            print(f"    Throughput: {results['performance_metrics']['throughput_decisions_per_second']:.2f} decisions/second")
            
            print("  ✅ Performance Benchmark completed!")
            
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            results['status'] = 'FAILED'
            results['error'] = str(e)
            results['end_time'] = datetime.now().isoformat()
            print(f"  ❌ Benchmark failed: {e}")
        
        return results

def main():
    """Main demonstration entry point"""
    print("🚄 Intelligent Train Traffic Control System")
    print("   Demonstration and Examples Suite")
    print("="*50)
    
    # Check if user wants interactive mode
    try:
        mode = input("\nChoose demonstration mode:\n1. Interactive (choose demos)\n2. Automated (run all demos)\n3. Benchmark (performance testing)\nEnter choice (1-3): ").strip()
        
        if mode == '1':
            # Interactive mode
            interactive_runner = InteractiveDemoRunner()
            interactive_runner.run_interactive()
            
        elif mode == '2':
            # Automated mode - run all demos
            scenario_runner = ScenarioRunner()
            comprehensive_result = scenario_runner.run_comprehensive_system_demo()
            
            print(f"\n🎯 Final Result: {comprehensive_result['status']}")
            if 'success_rate' in comprehensive_result['details']:
                print(f"   Success Rate: {comprehensive_result['details']['success_rate']:.1f}%")
            
        elif mode == '3':
            # Benchmark mode
            print("\nPerformance Benchmark Options:")
            scale = input("Choose scale (1=Small: 20/5, 2=Medium: 50/10, 3=Large: 100/20): ").strip()
            
            if scale == '1':
                trains, sections = 20, 5
            elif scale == '2':
                trains, sections = 50, 10
            elif scale == '3':
                trains, sections = 100, 20
            else:
                trains, sections = 50, 10
            
            benchmark_runner = BenchmarkRunner()
            benchmark_result = benchmark_runner.run_performance_benchmark(trains, sections)
            
            print(f"\n🏁 Benchmark Result: {benchmark_result['status']}")
            if 'performance_metrics' in benchmark_result:
                metrics = benchmark_result['performance_metrics']
                print(f"   Average Response Time: {metrics['avg_decision_time_ms']:.2f}ms")
                print(f"   Throughput: {metrics['throughput_decisions_per_second']:.2f} decisions/second")
        
        else:
            print("Invalid choice. Running automated demo...")
            scenario_runner = ScenarioRunner()
            scenario_runner.run_comprehensive_system_demo()
    
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nDemo system error: {e}")
        print("Please check the system setup and try again.")

if __name__ == "__main__":
    main()
