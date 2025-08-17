import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class DynamicConfigSystem:
    """
    Central configuration system that adapts parameters based on context and learning
    """
    
    def __init__(self):
        self.learned_parameters = {}
        self.context_adaptations = {}
        self.performance_history = []
        
    def get_adaptive_threshold(self, context: Dict, analysis_type: str, default: float = 0.3) -> float:
        """Get adaptive similarity threshold based on context and performance"""
        context_key = f"{analysis_type}_{context.get('alert_type', 'default')}"
        
        # Check if we have learned parameters for this context
        if context_key in self.learned_parameters:
            learned_threshold = self.learned_parameters[context_key].get('similarity_threshold', default)
            
            # Adapt based on recent performance
            if self.performance_history:
                recent_performance = self._get_recent_performance(context_key)
                if recent_performance['accuracy'] < 0.7:  # Poor performance
                    return max(learned_threshold + 0.1, 0.9)  # Increase threshold
                elif recent_performance['accuracy'] > 0.9:  # Excellent performance
                    return max(learned_threshold - 0.05, 0.1)  # Decrease threshold
            
            return learned_threshold
        
        # Adapt default based on alert characteristics
        if context.get('severity') == 'critical':
            return min(default + 0.1, 0.9)  # Higher threshold for critical alerts
        elif context.get('severity') == 'low':
            return max(default - 0.1, 0.1)  # Lower threshold for low severity
        
        return default
    
    def get_dynamic_risk_thresholds(self, context: Dict, agent_type: str) -> Dict[str, int]:
        """Get dynamic risk level thresholds based on context"""
        base_thresholds = {
            'CRITICAL': 80,
            'HIGH': 60,
            'MEDIUM': 40,
            'LOW': 20,
            'NORMAL': 0
        }
        
        # Adapt based on alert context
        severity_multiplier = self._get_severity_multiplier(context.get('severity', 'medium'))
        environment_factor = self._get_environment_factor(context)
        
        adapted_thresholds = {}
        for level, threshold in base_thresholds.items():
            adapted_value = int(threshold * severity_multiplier * environment_factor)
            adapted_thresholds[level] = max(min(adapted_value, 100), 0)
        
        return adapted_thresholds
    
    def get_dynamic_agent_weights(self, context: Dict, available_agents: List[str]) -> Dict[str, float]:
        """Calculate dynamic weights for agents based on context and performance"""
        base_weights = {
            'user_context': 0.25,
            'ip_agent': 0.25,
            'timeline': 0.20,
            'alert_history': 0.30
        }
        
        # Adapt weights based on alert type
        alert_type = context.get('alert_type', '').lower()
        threat_category = context.get('threat_category', '').lower()
        
        adapted_weights = base_weights.copy()
        
        # Increase user_context weight for authentication-related alerts
        if any(term in alert_type for term in ['auth', 'login', 'credential', 'account']):
            adapted_weights['user_context'] = min(adapted_weights['user_context'] + 0.15, 0.5)
        
        # Increase ip_agent weight for network-related alerts
        if any(term in alert_type for term in ['network', 'connection', 'traffic', 'firewall']):
            adapted_weights['ip_agent'] = min(adapted_weights['ip_agent'] + 0.15, 0.5)
        
        # Increase timeline weight for multi-stage attacks
        if any(term in threat_category for term in ['lateral', 'persistence', 'exfiltration']):
            adapted_weights['timeline'] = min(adapted_weights['timeline'] + 0.1, 0.4)
        
        # Adjust for novel vs known threats
        if context.get('historical_matches', 0) == 0:  # Novel threat
            adapted_weights['alert_history'] = max(adapted_weights['alert_history'] - 0.1, 0.1)
            adapted_weights['timeline'] = min(adapted_weights['timeline'] + 0.1, 0.4)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(adapted_weights[agent] for agent in available_agents if agent in adapted_weights)
        if total_weight > 0:
            for agent in available_agents:
                if agent in adapted_weights:
                    adapted_weights[agent] = adapted_weights[agent] / total_weight
        
        return adapted_weights
    
    def get_adaptive_confidence_params(self, context: Dict) -> Dict[str, Any]:
        """Get adaptive confidence calculation parameters"""
        return {
            'base_confidence': self._calculate_base_confidence(context),
            'consistency_weight': self._get_consistency_weight(context),
            'evidence_weight': self._get_evidence_weight(context),
            'contradiction_penalty': self._get_contradiction_penalty(context)
        }
    
    def _get_severity_multiplier(self, severity: str) -> float:
        """Get multiplier based on alert severity"""
        multipliers = {
            'critical': 1.2,
            'high': 1.1,
            'medium': 1.0,
            'low': 0.9,
            'info': 0.8
        }
        return multipliers.get(severity.lower(), 1.0)
    
    def _get_environment_factor(self, context: Dict) -> float:
        """Get environment-specific adjustment factor"""
        # Production environments should have higher thresholds
        if context.get('environment') == 'production':
            return 1.1
        elif context.get('environment') == 'development':
            return 0.9
        
        if context.get('timestamp'):
            try:
                hour = datetime.fromtimestamp(context['timestamp']).hour
                if hour < 6 or hour > 22:  # Off-hours
                    return 1.1
            except:
                pass
        
        return 1.0
    
    def _calculate_base_confidence(self, context: Dict) -> float:
        """Calculate adaptive base confidence"""
        base = 50.0
        
        # Adjust based on data quality
        if context.get('data_completeness', 1.0) < 0.8:
            base -= 10
        
        # Adjust based on alert age
        if context.get('alert_age_hours', 0) > 24:
            base -= 5
        
        # Adjust based on system load/performance
        if context.get('system_load', 0.5) > 0.8:
            base -= 5
        
        return max(base, 20.0)
    
    def _get_consistency_weight(self, context: Dict) -> float:
        """Get weight for consistency findings"""
        base_weight = 15.0
        
        # Higher weight when more agents agree
        agent_count = context.get('agent_count', 1)
        if agent_count > 3:
            base_weight += 5.0
        
        return base_weight
    
    def _get_evidence_weight(self, context: Dict) -> float:
        """Get weight for supporting evidence"""
        base_weight = 10.0
        
        # Higher weight for high-confidence evidence
        if context.get('evidence_quality', 'medium') == 'high':
            base_weight += 5.0
        
        return base_weight
    
    def _get_contradiction_penalty(self, context: Dict) -> float:
        """Get penalty for contradictory findings"""
        base_penalty = 20.0
        
        # Higher penalty in critical situations
        if context.get('severity') == 'critical':
            base_penalty += 10.0
        
        return base_penalty
    
    def _get_recent_performance(self, context_key: str) -> Dict[str, float]:
        """Get recent performance metrics for learning"""
        # Mock implementation - in real system would track actual performance
        return {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.90,
            'false_positive_rate': 0.15
        }
    
    def learn_from_feedback(self, context: Dict, analysis_type: str, 
                          parameters_used: Dict, outcome: Dict):
        """Learn and adapt parameters based on feedback"""
        context_key = f"{analysis_type}_{context.get('alert_type', 'default')}"
        
        if context_key not in self.learned_parameters:
            self.learned_parameters[context_key] = {}
        
        # Store performance feedback
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'parameters': parameters_used,
            'outcome': outcome,
            'context': context
        }
        
        self.performance_history.append(feedback)
        
        # Update learned parameters based on success/failure
        if outcome.get('success', False):
            # Successful analysis - reinforce these parameters
            for param, value in parameters_used.items():
                if param not in self.learned_parameters[context_key]:
                    self.learned_parameters[context_key][param] = value
                else:
                    # Moving average toward successful parameters
                    current = self.learned_parameters[context_key][param]
                    self.learned_parameters[context_key][param] = (current * 0.8) + (value * 0.2)
        
        # Keep only recent history (last 1000 entries)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_time_window_thresholds(self, context: Dict) -> Dict[str, int]:
        """Get adaptive time window thresholds"""
        base_windows = {
            'rapid_succession': 300,    # 5 minutes
            'short_term': 3600,         # 1 hour  
            'medium_term': 86400,       # 1 day
            'long_term': 604800         # 1 week
        }
        
        # Adapt based on alert type
        alert_type = context.get('alert_type', '').lower()
        
        if 'brute' in alert_type or 'flood' in alert_type:
            # Faster thresholds for brute force attacks
            return {k: int(v * 0.5) for k, v in base_windows.items()}
        elif 'persistence' in alert_type or 'apt' in alert_type:
            # Longer thresholds for APT activities
            return {k: int(v * 2.0) for k, v in base_windows.items()}
        
        return base_windows
    
    def get_query_complexity_params(self, context: Dict, prompt: str) -> Dict[str, Any]:
        """Get adaptive parameters for query complexity"""
        prompt_complexity = len(prompt.split())
        
        return {
            'max_events': min(10 + prompt_complexity, 50),
            'search_depth': min(5 + (prompt_complexity // 5), 20),
            'correlation_window': 3600 * (1 + prompt_complexity // 10),
            'pattern_sensitivity': max(0.1, 0.5 - (prompt_complexity * 0.02))
        }

# Global configuration instance
config_system = DynamicConfigSystem()

def get_config() -> DynamicConfigSystem:
    """Get the global configuration system instance"""
    return config_system