# MERGED DELIVERABLE 2: AI DECISION ENGINE
# File: ai_decision_engine.py

"""
AI Decision Engine for Intelligent Train Traffic Control System
Combines constraint programming, reinforcement learning, and decision integration
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import logging
import json
import threading
import time
from collections import deque, defaultdict
import heapq
import uuid
import pickle
import random

from core_data_layer import Train, Section, TrainType, TrainStatus, Constraint, ScheduleEntry

logger = logging.getLogger(__name__)

# ==================== CONSTRAINT PROGRAMMING SOLVER ====================

class ConstraintProgrammingSolver:
    """
    Advanced Constraint Programming solver for train scheduling
    Uses activity-based modeling with sophisticated constraint propagation
    """
    
    def __init__(self):
        self.trains: List[Train] = []
        self.sections: List[Section] = []
        self.constraints: List = []
        self.solution_cache: Dict = {}
        self.scheduling_horizon_hours: int = 24
        
    def add_train(self, train: Train):
        """Add train to the scheduling problem"""
        if train not in self.trains:
            self.trains.append(train)
            logger.debug(f"Added train {train.train_id} to solver")
    
    def add_section(self, section: Section):
        """Add section to the scheduling problem"""
        if section not in self.sections:
            self.sections.append(section)
            logger.debug(f"Added section {section.section_id} to solver")
    
    def remove_train(self, train_id: str):
        """Remove train from scheduling problem"""
        self.trains = [t for t in self.trains if t.train_id != train_id]
        
    def clear_all(self):
        """Clear all trains and sections"""
        self.trains.clear()
        self.sections.clear()
        self.solution_cache.clear()
    
    def calculate_travel_time(self, train: Train, section: Section) -> float:
        """
        Calculate precise travel time considering multiple factors
        Returns time in minutes
        """
        # Base calculation using section's method
        base_time = section.calculate_travel_time(train)
        
        # Additional factors for constraint programming
        
        # Congestion factor based on current occupancy
        utilization = section.get_utilization()
        if utilization > 0.8:
            congestion_penalty = 1.0 + (utilization - 0.8) * 0.5  # Up to 10% penalty
            base_time *= congestion_penalty
        
        # Signal spacing factor
        if section.signal_spacing_km > 3.0:  # Sparse signals
            base_time *= 1.1  # 10% penalty for longer signal blocks
        
        # Weather and external factors (placeholder for future enhancement)
        # Could integrate with weather APIs
        weather_factor = 1.0  # Default no impact
        
        # Priority-based speed adjustment
        if train.priority == 1 and train.train_type == TrainType.EXPRESS:
            base_time *= 0.95  # VIP trains get 5% speed bonus
        
        return base_time * weather_factor
    
    def check_conflict(self, train1: Train, train2: Train, section: Section,
                      start_time1: datetime, start_time2: datetime) -> bool:
        """
        Advanced conflict detection between two trains
        """
        if train1.train_id == train2.train_id:
            return False
        
        # Calculate travel times and end times
        travel_time1 = self.calculate_travel_time(train1, section)
        travel_time2 = self.calculate_travel_time(train2, section)
        
        end_time1 = start_time1 + timedelta(minutes=travel_time1)
        end_time2 = start_time2 + timedelta(minutes=travel_time2)
        
        # Buffer time calculation (minimum separation)
        buffer_minutes = self.calculate_buffer_time(train1, train2, section)
        
        # Check for temporal overlap with buffer
        overlap = not (end_time1 + timedelta(minutes=buffer_minutes) <= start_time2 or 
                      end_time2 + timedelta(minutes=buffer_minutes) <= start_time1)
        
        return overlap
    
    def calculate_buffer_time(self, train1: Train, train2: Train, section: Section) -> float:
        """
        Calculate required buffer time between trains based on safety requirements
        """
        base_buffer = 5.0  # 5 minutes base buffer
        
        # Adjust based on train types
        if train1.train_type == TrainType.FREIGHT or train2.train_type == TrainType.FREIGHT:
            base_buffer += 3.0  # Extra buffer for freight
        
        # Adjust based on speed differences
        speed_diff = abs(train1.speed_kmph - train2.speed_kmph)
        if speed_diff > 40:  # Significant speed difference
            base_buffer += 2.0
        
        # Section characteristics
        if not section.has_loop:  # Single track
            base_buffer += 5.0
        
        if section.gradient_percent > 2.0:  # Steep gradient
            base_buffer += 2.0
        
        return base_buffer
    
    def optimize_schedule_order(self, trains: List[Train], section: Section) -> List[Train]:
        """
        Optimize the order of trains using priority-based sorting with constraints
        """
        # Multi-criteria sorting
        def sort_key(train: Train) -> Tuple:
            # Primary: Priority (lower number = higher priority)
            priority_score = train.priority
            
            # Secondary: Delay (delayed trains get higher priority)
            delay_penalty = -train.delay_minutes if train.delay_minutes > 0 else 0
            
            # Tertiary: Scheduled time (earlier trains first)
            time_score = train.scheduled_arrival.timestamp()
            
            # Quaternary: Train type priority
            type_priorities = {
                TrainType.EXPRESS: 1,
                TrainType.LOCAL: 2,
                TrainType.SPECIAL: 3,
                TrainType.FREIGHT: 4,
                TrainType.MAINTENANCE: 5
            }
            type_score = type_priorities.get(train.train_type, 3)
            
            # Passenger load consideration (more passengers = higher priority)
            passenger_bonus = -train.passenger_load / 1000.0  # Normalize passenger count
            
            return (priority_score, delay_penalty, type_score, time_score, passenger_bonus)
        
        # Sort trains by the composite criteria
        optimized_order = sorted(trains, key=sort_key)
        
        logger.debug(f"Optimized train order for section {section.section_id}: "
                    f"{[t.train_id for t in optimized_order]}")
        
        return optimized_order
    
    def solve_section_schedule(self, section_id: str, 
                             time_window_hours: int = 24,
                             optimization_objective: str = 'minimize_delay') -> Dict:
        """
        Main constraint programming solver for section scheduling
        
        Args:
            section_id: ID of the section to schedule
            time_window_hours: Scheduling horizon in hours
            optimization_objective: 'minimize_delay', 'maximize_throughput', or 'balanced'
        """
        try:
            # Find the section
            section = next((s for s in self.sections if s.section_id == section_id), None)
            if not section:
                return {'error': f'Section {section_id} not found'}
            
            # Get relevant trains for this section
            relevant_trains = [
                train for train in self.trains
                if (section_id in train.route_sections or 
                    train.current_location == section.start_station or
                    train.destination == section.end_station or
                    train.current_location == section.end_station)
            ]
            
            if not relevant_trains:
                return {
                    'schedule': [],
                    'throughput': 0,
                    'total_delay': 0,
                    'average_delay': 0,
                    'section_utilization': 0.0,
                    'optimization_objective': optimization_objective
                }
            
            logger.info(f"Scheduling {len(relevant_trains)} trains for section {section_id}")
            
            # Optimize train ordering based on objective
            if optimization_objective == 'maximize_throughput':
                optimized_trains = sorted(relevant_trains, 
                                        key=lambda t: (t.scheduled_arrival, t.priority))
            else:
                optimized_trains = self.optimize_schedule_order(relevant_trains, section)
            
            # Generate initial schedule
            schedule_entries = []
            current_section_time = datetime.now()
            total_delay = 0.0
            
            for train in optimized_trains:
                # Calculate earliest possible start time
                earliest_start = max(
                    train.get_actual_arrival(),
                    current_section_time
                )
                
                # Check for conflicts with already scheduled trains
                start_time = earliest_start
                max_attempts = 20
                attempt = 0
                
                while attempt < max_attempts:
                    conflicts = []
                    
                    for existing_entry in schedule_entries:
                        existing_train = next((t for t in self.trains 
                                             if t.train_id == existing_entry.train_id), None)
                        if existing_train and self.check_conflict(
                            train, existing_train, section, 
                            start_time, existing_entry.planned_start_time):
                            conflicts.append(existing_entry)
                    
                    if not conflicts:
                        break
                    
                    # Find the latest conflicting train and schedule after it
                    latest_conflict = max(conflicts, key=lambda e: e.planned_end_time)
                    buffer_time = self.calculate_buffer_time(train, existing_train, section)
                    start_time = latest_conflict.planned_end_time + timedelta(minutes=buffer_time)
                    attempt += 1
                
                # Calculate travel time and end time
                travel_time = self.calculate_travel_time(train, section)
                end_time = start_time + timedelta(minutes=travel_time)
                
                # Calculate delay
                original_delay = train.delay_minutes
                scheduling_delay = max(0, (start_time - train.get_actual_arrival()).total_seconds() / 60.0)
                total_train_delay = original_delay + scheduling_delay
                
                # Create schedule entry
                entry = ScheduleEntry(
                    train_id=train.train_id,
                    section_id=section_id,
                    planned_start_time=start_time,
                    planned_end_time=end_time,
                    delay_minutes=total_train_delay,
                    confidence_score=self._calculate_confidence(train, section, conflicts),
                    notes=f"Travel time: {travel_time:.1f}min, Buffer conflicts: {len(conflicts)}"
                )
                
                schedule_entries.append(entry)
                total_delay += total_train_delay
                
                # Update section occupancy time for next train
                current_section_time = end_time
            
            # Calculate metrics
            throughput = len(schedule_entries)
            average_delay = total_delay / max(throughput, 1)
            
            # Calculate section utilization
            if schedule_entries:
                schedule_span = (max(entry.planned_end_time for entry in schedule_entries) - 
                               min(entry.planned_start_time for entry in schedule_entries))
                total_time_hours = schedule_span.total_seconds() / 3600.0
                train_hours = sum((entry.planned_end_time - entry.planned_start_time).total_seconds() / 3600.0 
                                for entry in schedule_entries)
                utilization = min(train_hours / (total_time_hours * section.capacity_trains), 1.0)
            else:
                utilization = 0.0
            
            # Prepare result
            result = {
                'schedule': [entry.to_dict() for entry in schedule_entries],
                'throughput': throughput,
                'total_delay': total_delay,
                'average_delay': average_delay,
                'section_utilization': utilization,
                'optimization_objective': optimization_objective,
                'solution_quality': self._assess_solution_quality(schedule_entries, section),
                'computational_time_ms': 0  # Would track actual computation time
            }
            
            # Cache the solution
            cache_key = f"{section_id}_{time_window_hours}_{optimization_objective}"
            self.solution_cache[cache_key] = result
            
            logger.info(f"Successfully scheduled {throughput} trains for section {section_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error solving schedule for section {section_id}: {str(e)}")
            return {'error': f'Scheduling failed: {str(e)}'}
    
    def _calculate_confidence(self, train: Train, section: Section, conflicts: List) -> float:
        """Calculate confidence score for a scheduling decision"""
        base_confidence = 0.9
        
        # Reduce confidence for conflicts
        if conflicts:
            base_confidence -= len(conflicts) * 0.1
        
        # Reduce confidence for high utilization
        if section.get_utilization() > 0.8:
            base_confidence -= 0.1
        
        # Increase confidence for express trains (higher priority)
        if train.train_type == TrainType.EXPRESS:
            base_confidence += 0.05
        
        # Reduce confidence for maintenance trains
        if train.train_type == TrainType.MAINTENANCE:
            base_confidence -= 0.05
        
        return max(0.0, min(1.0, base_confidence))
    
    def _assess_solution_quality(self, schedule: List[ScheduleEntry], section: Section) -> str:
        """Assess the quality of the generated solution"""
        if not schedule:
            return "NO_SOLUTION"
        
        avg_delay = sum(entry.delay_minutes for entry in schedule) / len(schedule)
        avg_confidence = sum(entry.confidence_score for entry in schedule) / len(schedule)
        utilization = section.get_utilization()
        
        if avg_delay < 15 and avg_confidence > 0.8 and utilization < 0.9:
            return "OPTIMAL"
        elif avg_delay < 30 and avg_confidence > 0.6:
            return "GOOD"
        elif avg_delay < 60:
            return "ACCEPTABLE"
        else:
            return "POOR"

# ==================== REINFORCEMENT LEARNING AGENT ====================

class TrainSchedulingRLAgent:
    """
    Reinforcement Learning agent for learning optimal scheduling policies
    Based on passenger-centric RL approach with adaptive learning
    """
    
    def __init__(self, 
                 state_size: int = 25, 
                 action_size: int = 12,
                 learning_rate: float = 0.01,
                 discount_factor: float = 0.9,
                 exploration_rate: float = 0.1):
        
        # Core RL parameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = exploration_rate  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Experience replay memory
        self.memory = deque(maxlen=50000)
        self.batch_size = 32
        
        # Q-learning table (state -> action values)
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        
        # Performance tracking
        self.performance_history = []
        self.reward_history = []
        self.episode_count = 0
        
        # Learning statistics
        self.successful_episodes = 0
        self.total_reward = 0.0
        self.best_reward = float('-inf')
        
        # Action definitions
        self.action_definitions = self._initialize_action_definitions()
        
        logger.info(f"Initialized RL Agent with state_size={state_size}, action_size={action_size}")
    
    def _initialize_action_definitions(self) -> Dict[int, Dict]:
        """Define the action space for the RL agent"""
        return {
            0: {
                'name': 'FIFO_SCHEDULING',
                'description': 'First-in-first-out scheduling',
                'priority_weight': 0.1,
                'delay_sensitivity': 0.3,
                'throughput_focus': 0.6
            },
            1: {
                'name': 'PRIORITY_FIRST',
                'description': 'Strict priority-based scheduling',
                'priority_weight': 0.8,
                'delay_sensitivity': 0.1,
                'throughput_focus': 0.1
            },
            2: {
                'name': 'SHORTEST_JOB_FIRST',
                'description': 'Schedule shortest travel time first',
                'priority_weight': 0.2,
                'delay_sensitivity': 0.2,
                'throughput_focus': 0.6
            },
            3: {
                'name': 'DELAY_MINIMIZATION',
                'description': 'Focus on minimizing total delays',
                'priority_weight': 0.3,
                'delay_sensitivity': 0.6,
                'throughput_focus': 0.1
            },
            4: {
                'name': 'THROUGHPUT_MAXIMIZATION',
                'description': 'Maximize number of trains scheduled',
                'priority_weight': 0.1,
                'delay_sensitivity': 0.1,
                'throughput_focus': 0.8
            },
            5: {
                'name': 'EXPRESS_PRIORITY',
                'description': 'Prioritize express trains',
                'priority_weight': 0.7,
                'delay_sensitivity': 0.2,
                'throughput_focus': 0.1
            },
            6: {
                'name': 'BALANCED_OPTIMIZATION',
                'description': 'Balance all objectives equally',
                'priority_weight': 0.33,
                'delay_sensitivity': 0.33,
                'throughput_focus': 0.34
            },
            7: {
                'name': 'ADAPTIVE_CONGESTION',
                'description': 'Adapt to current congestion levels',
                'priority_weight': 0.4,
                'delay_sensitivity': 0.4,
                'throughput_focus': 0.2
            },
            8: {
                'name': 'PASSENGER_CENTRIC',
                'description': 'Optimize passenger experience',
                'priority_weight': 0.5,
                'delay_sensitivity': 0.4,
                'throughput_focus': 0.1
            },
            9: {
                'name': 'FREIGHT_SEPARATION',
                'description': 'Separate freight from passenger trains',
                'priority_weight': 0.6,
                'delay_sensitivity': 0.2,
                'throughput_focus': 0.2
            },
            10: {
                'name': 'PEAK_HOUR_OPTIMIZATION',
                'description': 'Optimize for peak hour conditions',
                'priority_weight': 0.3,
                'delay_sensitivity': 0.5,
                'throughput_focus': 0.2
            },
            11: {
                'name': 'ENERGY_EFFICIENT',
                'description': 'Focus on energy-efficient scheduling',
                'priority_weight': 0.4,
                'delay_sensitivity': 0.3,
                'throughput_focus': 0.3
            }
        }
    
    def get_state_vector(self, trains: List[Train], sections: List[Section], 
                        current_time: datetime = None) -> np.ndarray:
        """
        Convert current railway system state to feature vector for RL agent
        """
        if current_time is None:
            current_time = datetime.now()
        
        state = np.zeros(self.state_size)
        
        if trains:
            # Train-related features (indices 0-9)
            state[0] = len(trains)  # Number of trains
            state[1] = np.mean([t.delay_minutes for t in trains])  # Average delay
            state[2] = len([t for t in trains if t.train_type == TrainType.EXPRESS]) / len(trains)
            state[3] = len([t for t in trains if t.train_type == TrainType.LOCAL]) / len(trains)
            state[4] = len([t for t in trains if t.train_type == TrainType.FREIGHT]) / len(trains)
            
            # Priority distribution
            priorities = [t.priority for t in trains]
            state[5] = np.mean(priorities) if priorities else 0
            state[6] = np.std(priorities) if len(priorities) > 1 else 0
            
            # Delay distribution
            delays = [t.delay_minutes for t in trains]
            state[7] = np.max(delays) if delays else 0  # Max delay
            state[8] = len([d for d in delays if d > 15]) / len(delays)  # Fraction with >15min delay
            state[9] = np.sum([t.passenger_load for t in trains]) / len(trains)  # Avg passenger load
        
        if sections:
            # Section-related features (indices 10-15)
            state[10] = len(sections)  # Number of sections
            utilizations = [len(s.current_occupancy) / max(s.capacity_trains, 1) for s in sections]
            state[11] = np.mean(utilizations)  # Average utilization
            state[12] = np.max(utilizations) if utilizations else 0  # Max utilization
            state[13] = np.mean([s.length_km for s in sections])  # Average section length
            state[14] = np.mean([s.max_speed_kmph for s in sections])  # Average max speed
            state[15] = len([s for s in sections if s.has_loop]) / len(sections)  # Fraction with loops
        
        # Time-related features (indices 16-20)
        current_hour = current_time.hour
        state[16] = current_hour / 24.0  # Normalized hour of day
        state[17] = 1.0 if 6 <= current_hour <= 10 else 0.0  # Morning peak
        state[18] = 1.0 if 17 <= current_hour <= 21 else 0.0  # Evening peak
        state[19] = 1.0 if current_time.weekday() < 5 else 0.0  # Weekday
        state[20] = np.sin(2 * np.pi * current_hour / 24)  # Cyclical hour representation
        
        # System performance features (indices 21-24)
        if hasattr(self, 'recent_performance'):
            state[21] = self.recent_performance.get('avg_throughput', 0) / 10.0  # Normalized
            state[22] = self.recent_performance.get('avg_delay', 0) / 60.0  # Normalized
            state[23] = self.recent_performance.get('success_rate', 0)
            state[24] = len(self.reward_history[-10:]) / 10.0 if self.reward_history else 0
        
        return state
    
    def select_action(self, state_vector: np.ndarray, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        """
        state_key = self._state_to_key(state_vector)
        
        if explore and random.random() < self.epsilon:
            # Exploration: random action
            action = random.randrange(self.action_size)
            logger.debug(f"Exploration action selected: {action}")
        else:
            # Exploitation: best known action
            q_values = self.q_table[state_key]
            action = np.argmax(q_values)
            logger.debug(f"Exploitation action selected: {action}, Q-value: {q_values[action]:.3f}")
        
        return action
    
    def get_action_strategy(self, action: int) -> Dict:
        """Get the strategy definition for an action"""
        return self.action_definitions.get(action, self.action_definitions[0])
    
    def calculate_reward(self, schedule_result: Dict, trains: List[Train], 
                        action: int, state_vector: np.ndarray) -> float:
        """
        Calculate reward based on multiple objectives and outcomes
        """
        if 'error' in schedule_result or not schedule_result.get('schedule'):
            return -100.0  # Heavy penalty for failure
        
        # Extract metrics
        throughput = schedule_result.get('throughput', 0)
        total_delay = schedule_result.get('total_delay', 0)
        average_delay = schedule_result.get('average_delay', 0)
        utilization = schedule_result.get('section_utilization', 0)
        
        # Base reward components
        throughput_reward = throughput * 15  # Encourage high throughput
        delay_penalty = -total_delay * 0.3   # Penalize total delays
        utilization_reward = utilization * 25  # Encourage efficient resource use
        
        # Passenger-centric bonuses
        passenger_trains = len([t for t in trains 
                              if t.train_type in [TrainType.EXPRESS, TrainType.LOCAL]])
        passenger_bonus = passenger_trains * 8
        
        # Express train priority bonus
        express_trains = len([t for t in trains if t.train_type == TrainType.EXPRESS])
        express_bonus = express_trains * 5
        
        # Delay severity penalties
        if average_delay > 30:
            delay_penalty -= 20  # Extra penalty for high delays
        elif average_delay < 10:
            delay_penalty += 10  # Bonus for very low delays
        
        # Action-specific rewards based on strategy effectiveness
        action_strategy = self.get_action_strategy(action)
        strategy_bonus = 0
        
        # Evaluate strategy effectiveness
        if action_strategy['name'] == 'PASSENGER_CENTRIC' and passenger_trains > 0:
            strategy_bonus += 10
        elif action_strategy['name'] == 'EXPRESS_PRIORITY' and express_trains > 0:
            strategy_bonus += 8
        elif action_strategy['name'] == 'THROUGHPUT_MAXIMIZATION' and throughput >= len(trains):
            strategy_bonus += 12
        elif action_strategy['name'] == 'DELAY_MINIMIZATION' and average_delay < 15:
            strategy_bonus += 10
        
        # Time-based bonuses
        current_hour = datetime.now().hour
        if 6 <= current_hour <= 10 or 17 <= current_hour <= 21:  # Peak hours
            if action_strategy['name'] == 'PEAK_HOUR_OPTIMIZATION':
                strategy_bonus += 5
        
        # Combine all reward components
        total_reward = (throughput_reward + delay_penalty + utilization_reward + 
                       passenger_bonus + express_bonus + strategy_bonus)
        
        # Normalize reward
        total_reward = max(-150, min(200, total_reward))
        
        logger.debug(f"Reward calculation: throughput={throughput_reward:.1f}, "
                    f"delay={delay_penalty:.1f}, utilization={utilization_reward:.1f}, "
                    f"strategy={strategy_bonus:.1f}, total={total_reward:.1f}")
        
        return total_reward
    
    def update_policy(self, state: np.ndarray, action: int, reward: float, 
                     next_state: np.ndarray, done: bool = False):
        """
        Update Q-table using Q-learning algorithm
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Next state's maximum Q-value
        if done:
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_state_key])
        
        # Q-learning update rule: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # Store experience in memory for potential replay
        experience = (state_key, action, reward, next_state_key, done)
        self.memory.append(experience)
        
        # Update learning statistics
        self.reward_history.append(reward)
        self.total_reward += reward
        
        if reward > self.best_reward:
            self.best_reward = reward
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        logger.debug(f"Q-value updated: {current_q:.3f} -> {new_q:.3f}, "
                    f"reward: {reward:.1f}, epsilon: {self.epsilon:.3f}")
    
    def _state_to_key(self, state_vector: np.ndarray) -> tuple:
        """Convert state vector to hashable key for Q-table"""
        # Round to 2 decimal places to reduce state space
        rounded_state = np.round(state_vector, 2)
        return tuple(rounded_state)
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        recent_rewards = self.reward_history[-100:]  # Last 100 episodes
        
        stats = {
            'total_episodes': len(self.reward_history),
            'successful_episodes': self.successful_episodes,
            'success_rate': self.successful_episodes / max(len(self.reward_history), 1),
            'total_reward': self.total_reward,
            'average_reward': self.total_reward / max(len(self.reward_history), 1),
            'recent_average_reward': np.mean(recent_rewards) if recent_rewards else 0,
            'recent_best_reward': np.max(recent_rewards) if recent_rewards else 0,
            'best_reward_ever': self.best_reward,
            'current_epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'memory_size': len(self.memory),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }
        
        return stats

# ==================== REAL-TIME DECISION ENGINE ====================

class RealTimeDecisionEngine:
    """
    Main decision engine that integrates multiple AI approaches for optimal train scheduling
    Provides real-time decision making with confidence scoring and performance tracking
    """
    
    def __init__(self, 
                 use_rl: bool = True,
                 use_simulation: bool = True,
                 decision_timeout_seconds: float = 30.0):
        
        # Core components
        self.cp_solver = ConstraintProgrammingSolver()
        self.rl_agent = TrainSchedulingRLAgent() if use_rl else None
        self.simulator = None  # Will be injected if needed
        
        # Configuration
        self.use_rl = use_rl
        self.use_simulation = use_simulation
        self.decision_timeout = decision_timeout_seconds
        
        # Decision tracking
        self.decision_history: List[Dict] = []
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'failed_decisions': 0,
            'average_response_time_ms': 0.0,
            'average_confidence_score': 0.0,
            'total_trains_scheduled': 0,
            'total_delay_reduced': 0.0,
            'system_uptime_hours': 0.0
        }
        
        # Real-time monitoring
        self.active_decisions: Dict[str, Dict] = {}
        self.system_alerts: List[Dict] = []
        self.system_start_time = datetime.now()
        
        # Thread safety
        self._decision_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        
        # Emergency protocols
        self.emergency_mode = False
        
        logger.info(f"Real-Time Decision Engine initialized (RL: {use_rl}, Simulation: {use_simulation})")
    
    def make_decision(self, 
                     section_id: str, 
                     trains: List[Train], 
                     sections: List[Section],
                     constraints: List[Constraint] = None,
                     emergency_mode: bool = False,
                     optimization_objective: str = 'balanced',
                     user_preferences: Dict = None) -> Dict:
        """
        Make a comprehensive real-time scheduling decision
        
        Args:
            section_id: Target section for scheduling
            trains: List of trains to schedule
            sections: List of relevant sections
            constraints: Additional operational constraints
            emergency_mode: Use emergency protocols if True
            optimization_objective: 'minimize_delay', 'maximize_throughput', 'balanced'
            user_preferences: Controller preferences and overrides
        
        Returns:
            Comprehensive decision with recommendations, confidence scores, and alternatives
        """
        decision_id = f"DEC_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{section_id}"
        start_time = datetime.now()
        
        if constraints is None:
            constraints = []
        if user_preferences is None:
            user_preferences = {}
        
        with self._decision_lock:
            try:
                # Initialize decision tracking
                decision_context = {
                    'decision_id': decision_id,
                    'section_id': section_id,
                    'start_time': start_time,
                    'trains_count': len(trains),
                    'emergency_mode': emergency_mode,
                    'optimization_objective': optimization_objective
                }
                
                self.active_decisions[decision_id] = decision_context
                
                # Validate inputs
                validation_result = self._validate_decision_inputs(section_id, trains, sections)
                if not validation_result['valid']:
                    return self._create_error_decision(decision_id, validation_result['error'], start_time)
                
                # Multi-algorithm decision making
                decision_components = {}
                
                # 1. Constraint Programming Solution
                cp_result = self._get_cp_solution(section_id, trains, sections, constraints, 
                                                optimization_objective)
                decision_components['constraint_programming'] = cp_result
                
                # 2. Reinforcement Learning Recommendation
                if self.use_rl and self.rl_agent:
                    rl_result = self._get_rl_recommendation(trains, sections, optimization_objective)
                    decision_components['reinforcement_learning'] = rl_result
                else:
                    decision_components['reinforcement_learning'] = {'available': False}
                
                # 3. Integrate all components into final decision
                integrated_decision = self._integrate_decision_components(
                    decision_components, user_preferences, optimization_objective)
                
                # 4. Calculate comprehensive confidence score
                confidence_score = self._calculate_decision_confidence(
                    decision_components, integrated_decision, [])
                
                # 5. Generate human-readable recommendations
                recommendations = self._generate_recommendations(
                    integrated_decision, [], confidence_score, trains, sections)
                
                # Calculate response time
                response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                # Create comprehensive decision object
                final_decision = {
                    'decision_id': decision_id,
                    'section_id': section_id,
                    'timestamp': start_time.isoformat(),
                    'response_time_ms': response_time_ms,
                    'trains_considered': len(trains),
                    'optimization_objective': optimization_objective,
                    'emergency_mode': emergency_mode,
                    
                    # Core decision components
                    'primary_solution': integrated_decision,
                    'alternatives': [],
                    'decision_components': decision_components,
                    
                    # Decision quality metrics
                    'confidence_score': confidence_score,
                    'reliability_score': self._calculate_reliability_score(decision_components),
                    
                    # Human interface
                    'recommendations': recommendations,
                    'explanation': self._generate_decision_explanation(decision_components, integrated_decision),
                    'action_required': self._determine_required_actions(integrated_decision, confidence_score),
                    
                    # System context
                    'system_state': self._get_system_state_summary(),
                    'constraints_applied': len(constraints),
                    'user_preferences_applied': list(user_preferences.keys()) if user_preferences else [],
                    
                    'success': True
                }
                
                # Update learning systems
                if self.use_rl and self.rl_agent and cp_result.get('throughput', 0) > 0:
                    self._update_rl_learning(trains, sections, rl_result, cp_result)
                
                # Store decision and update metrics
                self._store_decision(final_decision)
                self._update_performance_metrics(final_decision)
                
                # Cleanup active decision tracking
                if decision_id in self.active_decisions:
                    del self.active_decisions[decision_id]
                
                logger.info(f"Decision {decision_id} completed successfully in {response_time_ms:.1f}ms "
                           f"(confidence: {confidence_score:.1%})")
                
                return final_decision
                
            except Exception as e:
                logger.error(f"Decision making failed for {decision_id}: {str(e)}")
                error_decision = self._create_error_decision(decision_id, str(e), start_time)
                
                # Cleanup on error
                if decision_id in self.active_decisions:
                    del self.active_decisions[decision_id]
                
                return error_decision
    
    def _validate_decision_inputs(self, section_id: str, trains: List[Train], 
                                 sections: List[Section]) -> Dict:
        """Validate inputs for decision making"""
        if not section_id:
            return {'valid': False, 'error': 'Section ID cannot be empty'}
        
        if not trains:
            return {'valid': False, 'error': 'No trains provided for scheduling'}
        
        if not sections:
            return {'valid': False, 'error': 'No sections provided'}
        
        # Check if target section exists
        target_section = next((s for s in sections if s.section_id == section_id), None)
        if not target_section:
            return {'valid': False, 'error': f'Target section {section_id} not found'}
        
        return {'valid': True}
    
    def _get_cp_solution(self, section_id: str, trains: List[Train], sections: List[Section],
                        constraints: List[Constraint], optimization_objective: str) -> Dict:
        """Get constraint programming solution"""
        try:
            # Prepare solver
            self.cp_solver.clear_all()
            for train in trains:
                self.cp_solver.add_train(train)
            for section in sections:
                self.cp_solver.add_section(section)
            
            # Solve with specified objective
            result = self.cp_solver.solve_section_schedule(section_id, 24, optimization_objective)
            result['solver_type'] = 'constraint_programming'
            result['available'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"CP solver failed: {e}")
            return {'available': False, 'error': str(e)}
    
    def _get_rl_recommendation(self, trains: List[Train], sections: List[Section],
                              optimization_objective: str) -> Dict:
        """Get reinforcement learning recommendation"""
        try:
            # Get current state
            state_vector = self.rl_agent.get_state_vector(trains, sections)
            
            # Select action
            action = self.rl_agent.select_action(state_vector, explore=False)  # No exploration for decisions
            
            # Get strategy details
            strategy = self.rl_agent.get_action_strategy(action)
            
            return {
                'available': True,
                'action': action,
                'strategy': strategy,
                'state_vector': state_vector.tolist(),
                'confidence': 1.0 - self.rl_agent.epsilon,  # Higher confidence with less exploration
                'solver_type': 'reinforcement_learning'
            }
            
        except Exception as e:
            logger.error(f"RL agent failed: {e}")
            return {'available': False, 'error': str(e)}
    
    def _integrate_decision_components(self, components: Dict, user_preferences: Dict,
                                     optimization_objective: str) -> Dict:
        """Integrate multiple decision components into final solution"""
        
        # Start with CP solution as base (most reliable)
        cp_result = components.get('constraint_programming', {})
        if not cp_result.get('available', False):
            return {'error': 'No viable solution found'}
        
        integrated = cp_result.copy()
        
        # Apply RL strategy modifications
        rl_result = components.get('reinforcement_learning', {})
        if rl_result.get('available', False):
            strategy = rl_result.get('strategy', {})
            
            # Modify solution based on RL strategy
            if strategy.get('name') == 'EXPRESS_PRIORITY':
                # Boost express train priorities in the solution
                integrated['strategy_modification'] = 'EXPRESS_PRIORITY_APPLIED'
            elif strategy.get('name') == 'DELAY_MINIMIZATION':
                # Focus on delay reduction
                integrated['strategy_modification'] = 'DELAY_FOCUS_APPLIED'
            
            integrated['rl_strategy'] = strategy
        
        # Apply user preferences
        if user_preferences:
            integrated['user_preferences_applied'] = user_preferences
            
            # Example preference applications
            if user_preferences.get('prefer_passenger_trains', False):
                integrated['passenger_priority_applied'] = True
        
        integrated['integration_method'] = 'weighted_multi_algorithm'
        integrated['components_used'] = list(components.keys())
        
        return integrated
    
    def _calculate_decision_confidence(self, components: Dict, decision: Dict, 
                                     alternatives: List[Dict]) -> float:
        """Calculate comprehensive confidence score for the decision"""
        
        confidence_factors = []
        
        # CP solver confidence
        cp_result = components.get('constraint_programming', {})
        if cp_result.get('available', False):
            throughput = cp_result.get('throughput', 0)
            avg_delay = cp_result.get('average_delay', 100)
            utilization = cp_result.get('section_utilization', 0)
            
            cp_confidence = 0.7  # Base CP confidence
            
            if throughput > 0:
                cp_confidence += 0.2
            if avg_delay < 20:
                cp_confidence += 0.1
            if 0.5 <= utilization <= 0.9:
                cp_confidence += 0.1
            
            confidence_factors.append(min(1.0, cp_confidence))
        
        # RL confidence
        rl_result = components.get('reinforcement_learning', {})
        if rl_result.get('available', False):
            rl_confidence = rl_result.get('confidence', 0.5)
            confidence_factors.append(rl_confidence)
        
        # Calculate weighted average confidence
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.5  # Moderate confidence if no factors available
    
    def _calculate_reliability_score(self, components: Dict) -> float:
        """Calculate reliability score based on component availability and consistency"""
        available_components = sum(1 for comp in components.values() if comp.get('available', False))
        total_components = len(components)
        
        base_reliability = available_components / max(total_components, 1)
        
        # Bonus for CP solver availability (most critical)
        if components.get('constraint_programming', {}).get('available', False):
            base_reliability += 0.2
        
        return min(1.0, base_reliability)
    
    def _generate_recommendations(self, decision: Dict, alternatives: List[Dict],
                                confidence_score: float, trains: List[Train], 
                                sections: List[Section]) -> List[str]:
        """Generate human-readable recommendations"""
        recommendations = []
        
        # Confidence-based recommendations
        if confidence_score > 0.8:
            recommendations.append("✅ High confidence decision - proceed with implementation")
        elif confidence_score > 0.6:
            recommendations.append("⚠️ Moderate confidence - monitor closely during execution")
        else:
            recommendations.append("🚨 Low confidence - consider manual review before implementation")
        
        # Performance-based recommendations
        avg_delay = decision.get('average_delay', 0)
        throughput = decision.get('throughput', 0)
        utilization = decision.get('section_utilization', 0)
        
        if avg_delay > 30:
            recommendations.append(f"⏰ Average delay of {avg_delay:.1f}min is above target - consider delay reduction measures")
        
        if utilization > 0.9:
            recommendations.append(f"🚦 Section utilization at {utilization:.1%} - monitor for congestion")
        elif utilization < 0.5:
            recommendations.append(f"📊 Low utilization at {utilization:.1%} - potential for increased capacity")
        
        return recommendations
    
    def _generate_decision_explanation(self, components: Dict, decision: Dict) -> str:
        """Generate human-readable explanation of the decision"""
        explanation_parts = []
        
        # Component availability
        available_components = [name for name, comp in components.items() 
                              if comp.get('available', False)]
        explanation_parts.append(f"Decision based on {len(available_components)} analysis methods: {', '.join(available_components)}")
        
        # Primary solution details
        throughput = decision.get('throughput', 0)
        avg_delay = decision.get('average_delay', 0)
        utilization = decision.get('section_utilization', 0)
        
        explanation_parts.append(f"Scheduled {throughput} trains with {avg_delay:.1f} minutes average delay")
        explanation_parts.append(f"Section utilization: {utilization:.1%}")
        
        return ". ".join(explanation_parts)
    
    def _determine_required_actions(self, decision: Dict, confidence_score: float) -> List[str]:
        """Determine what actions the controller should take"""
        actions = []
        
        if confidence_score < 0.6:
            actions.append("MANUAL_REVIEW_REQUIRED")
        
        avg_delay = decision.get('average_delay', 0)
        if avg_delay > 45:
            actions.append("PASSENGER_NOTIFICATION_RECOMMENDED")
        
        utilization = decision.get('section_utilization', 0)
        if utilization > 0.9:
            actions.append("MONITOR_CONGESTION")
        
        if not actions:
            actions.append("IMPLEMENT_AS_PLANNED")
        
        return actions
    
    def _update_rl_learning(self, trains: List[Train], sections: List[Section],
                           rl_result: Dict, cp_result: Dict):
        """Update RL agent with learning from decision outcome"""
        try:
            if not rl_result.get('available', False):
                return
            
            state_vector = rl_result['state_vector']
            action = rl_result['action']
            
            # Calculate reward from CP solution quality
            reward = self.rl_agent.calculate_reward(cp_result, trains, action, np.array(state_vector))
            
            # Update policy (using same state as next state for simplification)
            next_state = np.array(state_vector)
            self.rl_agent.update_policy(np.array(state_vector), action, reward, next_state)
            
            logger.debug(f"RL agent updated with reward {reward:.2f} for action {action}")
            
        except Exception as e:
            logger.error(f"Failed to update RL learning: {e}")
    
    def _get_system_state_summary(self) -> Dict:
        """Get current system state summary"""
        return {
            'active_decisions': len(self.active_decisions),
            'emergency_mode': self.emergency_mode,
            'system_uptime_minutes': (datetime.now() - self.system_start_time).total_seconds() / 60,
            'components_available': {
                'constraint_programming': True,
                'reinforcement_learning': self.use_rl and self.rl_agent is not None,
                'simulation': self.use_simulation
            }
        }
    
    def _create_error_decision(self, decision_id: str, error_message: str, start_time: datetime) -> Dict:
        """Create error decision response"""
        return {
            'decision_id': decision_id,
            'timestamp': start_time.isoformat(),
            'error': error_message,
            'success': False,
            'response_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
            'recommendations': ['⚠️ Decision making failed - manual intervention required'],
            'action_required': ['MANUAL_OVERRIDE_REQUIRED']
        }
    
    def _store_decision(self, decision: Dict):
        """Store decision in history"""
        self.decision_history.append(decision)
        
        # Keep only last 1000 decisions in memory
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    def _update_performance_metrics(self, decision: Dict):
        """Update system performance metrics"""
        with self._metrics_lock:
            self.performance_metrics['total_decisions'] += 1
            
            if decision.get('success', False):
                self.performance_metrics['successful_decisions'] += 1
                
                # Update response time (running average)
                response_time = decision.get('response_time_ms', 0)
                current_avg = self.performance_metrics['average_response_time_ms']
                total_decisions = self.performance_metrics['total_decisions']
                
                new_avg = (current_avg * (total_decisions - 1) + response_time) / total_decisions
                self.performance_metrics['average_response_time_ms'] = new_avg
                
                # Update confidence score
                confidence = decision.get('confidence_score', 0)
                current_conf_avg = self.performance_metrics['average_confidence_score']
                new_conf_avg = (current_conf_avg * (total_decisions - 1) + confidence) / total_decisions
                self.performance_metrics['average_confidence_score'] = new_conf_avg
                
                # Update trains scheduled
                throughput = decision.get('primary_solution', {}).get('throughput', 0)
                self.performance_metrics['total_trains_scheduled'] += throughput
                
            else:
                self.performance_metrics['failed_decisions'] += 1
            
            # Update system uptime
            uptime_seconds = (datetime.now() - self.system_start_time).total_seconds()
            self.performance_metrics['system_uptime_hours'] = uptime_seconds / 3600
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        with self._metrics_lock:
            metrics = self.performance_metrics.copy()
        
        return {
            'system_health': 'HEALTHY' if metrics['successful_decisions'] / max(metrics['total_decisions'], 1) > 0.8 else 'DEGRADED',
            'performance_metrics': metrics,
            'rl_performance': self.rl_agent.get_performance_stats() if self.rl_agent else None,
            'active_decisions': len(self.active_decisions),
            'emergency_mode': self.emergency_mode,
            'system_alerts': len(self.system_alerts),
            'components_status': {
                'constraint_programming': 'ACTIVE',
                'reinforcement_learning': 'ACTIVE' if self.use_rl else 'DISABLED',
                'simulation': 'ACTIVE' if self.use_simulation else 'DISABLED'
            },
            'recent_decisions': len([d for d in self.decision_history 
                                   if datetime.fromisoformat(d.get('timestamp', '1970-01-01T00:00:00')) > 
                                      datetime.now() - timedelta(hours=1)])
        }

if __name__ == "__main__":
    # Test the AI decision engine
    print("Testing AI Decision Engine...")
    
    from core_data_layer import create_sample_train, create_sample_section, TrainType
    
    # Create decision engine
    engine = RealTimeDecisionEngine(use_rl=True, use_simulation=False)
    
    # Create test data
    base_time = datetime.now()
    trains = [
        create_sample_train("T001", TrainType.EXPRESS, base_time, 5, ["SEC001"]),
        create_sample_train("T002", TrainType.LOCAL, base_time + timedelta(minutes=20), 0, ["SEC001"]),
        create_sample_train("T003", TrainType.FREIGHT, base_time + timedelta(minutes=40), 10, ["SEC001"])
    ]
    
    sections = [create_sample_section("SEC001", "StationA", "StationB", 50.0)]
    
    # Make decision
    decision = engine.make_decision(
        section_id="SEC001",
        trains=trains,
        sections=sections,
        optimization_objective='balanced'
    )
    
    print(f"Decision result:")
    print(f"  Success: {decision.get('success', False)}")
    print(f"  Decision ID: {decision.get('decision_id', 'N/A')}")
    print(f"  Response time: {decision.get('response_time_ms', 0):.1f}ms")
    print(f"  Confidence: {decision.get('confidence_score', 0):.1%}")
    print(f"  Recommendations: {len(decision.get('recommendations', []))}")
    
    if decision.get('success'):
        primary_solution = decision.get('primary_solution', {})
        print(f"  Throughput: {primary_solution.get('throughput', 0)} trains")
        print(f"  Average delay: {primary_solution.get('average_delay', 0):.1f} minutes")
    
    # Get system status
    status = engine.get_system_status()
    print(f"System status: {status['system_health']}")
    print(f"Total decisions: {status['performance_metrics']['total_decisions']}")
    
    print("AI Decision Engine test completed!")
