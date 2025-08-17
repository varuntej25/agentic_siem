import json
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime, timedelta
import requests
from gpu_utils import to_cuda, move_to_device, amp_autocast

class TimelineAgent:
    _event_catalogue = None

    @staticmethod
    def _load_event_catalogue(path="Spharaka-Windows_Events_Filtered.json"):
        if TimelineAgent._event_catalogue is None:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    TimelineAgent._event_catalogue = {int(e['event_id']): e for e in data if 'event_id' in e}
            except Exception:
                TimelineAgent._event_catalogue = {}
    def __init__(self, es_query_agent_url: str = "http://10.9.56.10:30200",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.es_query_agent_url = es_query_agent_url
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for semantic analysis, ensuring device consistency."""
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            # Move all tensors to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with amp_autocast():
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                    embeddings.append(embedding)
        return np.array(embeddings)
    
    def extract_temporal_context(self, alert: Dict, data_entry: Dict) -> Dict[str, Any]:
        """Extract time-related context from alert and data"""
        source = data_entry.get('_source', {})
        context = {}
        
        # Extract timestamp fields dynamically
        timestamp_fields = ['timestamp', 'eventDate', 'createdAt', 'logTime', 'occurredAt', 'time']
        for field in timestamp_fields:
            if source.get(field):
                context['primary_timestamp'] = source[field]
                break
        
        # Extract alert timestamp
        alert_timestamp_fields = ['createdAt', 'timestamp', 'alertTime', 'triggeredAt']
        for field in alert_timestamp_fields:
            if alert.get(field):
                context['alert_timestamp'] = alert[field]
                break
        
        # Extract context for timeline building
        context_mapping = {
            'event_id': ['eventID', 'event_id', 'eventCode'],
            'user': ['targetUsername', 'userName', 'user', 'accountName'],
            'host': ['hostName', 'computerName', 'host'],
            'ip': ['srcIP', 'sourceIP', 'clientIP'],
            'process': ['processName', 'process', 'executable'],
            'action': ['action', 'eventType', 'activity', 'operation'],
            'severity': ['severity', 'level', 'priority']
        }
        
        for context_key, possible_fields in context_mapping.items():
            for field in possible_fields:
                if source.get(field):
                    context[context_key] = source[field]
                    break
        
        # Add alert context
        context['alert_info'] = {
            'alert_id': alert.get('alertId'),
            'alert_name': alert.get('alertName'),
            'alert_severity': alert.get('severity')
        }
        
        return context
    
    def analyze_timeline_intent(self, analysis_prompt: str) -> Dict[str, Any]:
        """Analyze prompt to determine timeline analysis requirements"""
        prompt_embedding = self._get_embeddings([analysis_prompt])
        
        # Define timeline analysis types
        timeline_categories = {
            'sequence_analysis': 'sequence chronological order events progression timeline',
            'pattern_analysis': 'patterns recurring cycles frequency temporal patterns',
            'correlation_analysis': 'correlation relationships causation dependencies linked',
            'anomaly_analysis': 'anomalies unusual timing irregular schedule temporal anomalies',
            'progression_analysis': 'progression escalation development evolution attack chain',
            'duration_analysis': 'duration time spans intervals session length timing',
            'frequency_analysis': 'frequency rate occurrence repetition temporal frequency'
        }
        
        # Get similarities
        category_descriptions = list(timeline_categories.values())
        category_embeddings = self._get_embeddings(category_descriptions)
        similarities = cosine_similarity(prompt_embedding, category_embeddings)[0]
        
        # Determine timeline scope and focus
        threshold = 0.3
        relevant_categories = []
        category_keys = list(timeline_categories.keys())
        
        for i, similarity in enumerate(similarities):
            if similarity > threshold:
                relevant_categories.append(category_keys[i])
        
        # Default analysis if no specific intent
        if not relevant_categories:
            relevant_categories = ['sequence_analysis', 'correlation_analysis']
        
        # Determine time window based on prompt
        time_window = self._extract_time_window(analysis_prompt)
        
        return {
            'categories': relevant_categories,
            'time_window': time_window,
            'focus': 'comprehensive' if len(relevant_categories) > 2 else relevant_categories[0]
        }
    
    def _extract_time_window(self, analysis_prompt: str) -> str:
        """Extract time window from analysis prompt"""
        prompt_lower = analysis_prompt.lower()
        
        # Time window patterns
        if any(term in prompt_lower for term in ['hour', '60 minutes']):
            return 'last_hour'
        elif any(term in prompt_lower for term in ['day', '24 hours', 'today']):
            return 'last_day'
        elif any(term in prompt_lower for term in ['week', '7 days']):
            return 'last_week'
        elif any(term in prompt_lower for term in ['month', '30 days']):
            return 'last_month'
        else:
            return 'contextual'  # Use alert timestamp as reference
    
    def generate_timeline_queries(self, temporal_context: Dict, timeline_intent: Dict, other_agent_data: List[Dict]) -> List[Dict]:
        """Generate timeline-focused queries"""
        queries = []
        
        # Base context from alert
        base_timestamp = temporal_context.get('primary_timestamp') or temporal_context.get('alert_timestamp')
        user = temporal_context.get('user')
        host = temporal_context.get('host')
        ip = temporal_context.get('ip')
        
        # Generate queries based on timeline categories
        for category in timeline_intent['categories']:
            
            if category == 'sequence_analysis':
                if user:
                    queries.append({
                        'name': f'sequence_events_{user}',
                        'prompt': f"Find chronological sequence of events for user {user} around the alert timeframe",
                        'focus': 'event sequence'
                    })
                if ip:
                    queries.append({
                        'name': f'sequence_network_{ip}',
                        'prompt': f"Get chronological network events from IP {ip} for timeline analysis",
                        'focus': 'network sequence'
                    })
            
            elif category == 'pattern_analysis':
                queries.append({
                    'name': 'temporal_patterns',
                    'prompt': f"Analyze temporal patterns and recurring activities in the specified time window",
                    'focus': 'pattern detection'
                })
            
            elif category == 'correlation_analysis':
                if user and ip:
                    queries.append({
                        'name': f'correlation_{user}_{ip}',
                        'prompt': f"Find correlated events between user {user} and IP {ip} for timeline correlation",
                        'focus': 'event correlation'
                    })
            
            elif category == 'anomaly_analysis':
                queries.append({
                    'name': 'temporal_anomalies',
                    'prompt': f"Identify temporal anomalies and unusual timing patterns around the alert",
                    'focus': 'timing anomalies'
                })
            
            elif category == 'progression_analysis':
                queries.append({
                    'name': 'attack_progression',
                    'prompt': f"Analyze event progression and potential attack chain development",
                    'focus': 'progression tracking'
                })
        
        return queries
    
    def request_es_query(self, query_request: Dict) -> Dict:
        """Request ES query from Query Agent"""
        try:
            response = requests.post(
                f"{self.es_query_agent_url}/generate_query",
                json=query_request,
                timeout=30
            )
            return response.json()
        except:
            return {'elasticsearch_query': {'query': {'match_all': {}}}}
    
    def execute_es_query(self, es_query: Dict) -> Dict:
        """Execute ES query and get results"""
        try:
            response = requests.post(
                f"{self.es_query_agent_url}/execute_query",
                json={'query': es_query},
                timeout=60
            )
            return response.json()
        except:
            return {'hits': {'total': {'value': 0}, 'hits': []}}
    
    def process_timeline_data(self, query_results: Dict, other_agent_data: List[Dict]) -> List[Dict]:
        """Process and combine timeline data from all sources"""
        timeline_events = []
        
        # Process ES query results
        for query_name, results in query_results.items():
            hits = results.get('hits', {}).get('hits', [])
            for hit in hits:
                source = hit.get('_source', {})
                event = {
                    'timestamp': source.get('timestamp') or source.get('eventDate'),
                    'event_id': source.get('eventID'),
                    'description': source.get('eventDescription') or source.get('message'),
                    'user': source.get('targetUsername'),
                    'host': source.get('hostName'),
                    'ip': source.get('srcIP'),
                    'severity': source.get('severity'),
                    'source': f'es_query_{query_name}',
                    'raw_data': source
                }
                if event['timestamp']:
                    timeline_events.append(event)
        
        # Process other agent data
        for agent_data in other_agent_data:
            if agent_data.get('agent_type') == 'user_context':
                # Extract timeline-relevant data from user context agent
                user_context = agent_data.get('user_context', {})
                timeline_events.append({
                    'timestamp': user_context.get('alert_info', {}).get('timestamp'),
                    'event_type': 'user_analysis',
                    'description': f"User analysis completed: {agent_data.get('risk_level')}",
                    'user': user_context.get('username'),
                    'source': 'user_context_agent',
                    'analysis_data': agent_data
                })
            
            elif agent_data.get('agent_type') == 'ip_agent':
                # Extract timeline-relevant data from IP agent
                ip_context = agent_data.get('ip_context', {})
                timeline_events.append({
                    'timestamp': ip_context.get('alert_info', {}).get('timestamp'),
                    'event_type': 'ip_analysis',
                    'description': f"IP analysis completed: {agent_data.get('risk_level')}",
                    'ip': ip_context.get('primary_ip'),
                    'source': 'ip_agent',
                    'analysis_data': agent_data
                })
        
        # Sort by timestamp
        timeline_events.sort(key=lambda x: x.get('timestamp', 0) if x.get('timestamp') else 0)
        
        return timeline_events
    
    def analyze_timeline_patterns(self, query_results: Dict, other_agent_data: List[Dict], timeline_intent: Dict) -> Dict[str, Any]:
        """Analyze timeline for patterns, correlations, and anomalies using all sources."""
        self._load_event_catalogue()
        # Build timeline_events from both query_results and other_agent_data
        timeline_events = self.process_timeline_data(query_results, other_agent_data)
        # Calculate all required analysis fields
        patterns = self._analyze_sequence(timeline_events)
        correlations = self._find_correlations(timeline_events)
        anomalies = self._detect_temporal_anomalies(timeline_events)
        progression = self._analyze_progression(timeline_events)
        time_span = self._calculate_time_span(timeline_events)
        total_events = len(timeline_events)
        analysis = {
            'patterns': patterns,
            'correlations': correlations,
            'anomalies': anomalies,
            'progression': progression,
            'time_span': time_span,
            'total_events': total_events
        }
        risk_score = self._calculate_timeline_risk(analysis)
        analysis['risk_score'] = risk_score
        return analysis
        try:
            start_time = min(timestamps)
            end_time = max(timestamps)
            duration = end_time - start_time
            
            if duration < 3600:  # Less than 1 hour
                return f"{duration // 60} minutes"
            elif duration < 86400:  # Less than 1 day
                return f"{duration // 3600} hours"
            else:
                return f"{duration // 86400} days"
        except:
            return "Unknown timespan"
    
    def _analyze_sequence(self, events: List[Dict]) -> List[str]:
        """Analyze event sequence patterns"""
        patterns = []
        
        if len(events) > 5:
            patterns.append(f"Complex event sequence with {len(events)} events")
        elif len(events) > 2:
            patterns.append(f"Event sequence shows {len(events)} related activities")
        
        # Analyze rapid succession
        rapid_events = 0
        for i in range(1, len(events)):
            if events[i].get('timestamp') and events[i-1].get('timestamp'):
                if events[i]['timestamp'] - events[i-1]['timestamp'] < 300:  # 5 minutes
                    rapid_events += 1
        
        if rapid_events > 3:
            patterns.append("Rapid event succession detected")
        
        return patterns
    
    def _find_correlations(self, events: List[Dict]) -> List[str]:
        """Find correlations between events"""
        correlations = []
        
        # User correlations
        users = set(event.get('user') for event in events if event.get('user'))
        if len(users) == 1 and len(events) > 3:
            correlations.append(f"All events correlated to single user: {list(users)[0]}")
        
        # IP correlations
        ips = set(event.get('ip') for event in events if event.get('ip'))
        if len(ips) == 1 and len(events) > 3:
            correlations.append(f"All events correlated to single IP: {list(ips)[0]}")
        
        # Host correlations
        hosts = set(event.get('host') for event in events if event.get('host'))
        if len(hosts) == 1 and len(events) > 3:
            correlations.append(f"All events correlated to single host: {list(hosts)[0]}")
        
        return correlations
    
    def _detect_temporal_anomalies(self, events: List[Dict]) -> List[str]:
        """Detect temporal anomalies"""
        anomalies = []
        
        # Off-hours detection (mock implementation)
        off_hours_count = 0
        for event in events:
            if event.get('timestamp'):
                # Mock: consider events outside 9-17 as off-hours
                try:
                    hour = datetime.fromtimestamp(event['timestamp']).hour
                    if hour < 9 or hour > 17:
                        off_hours_count += 1
                except:
                    pass
        
        if off_hours_count > len(events) * 0.5:
            anomalies.append("Significant off-hours activity detected")
        
        return anomalies
    
    def _analyze_progression(self, events: List[Dict]) -> List[str]:
        """Analyze attack progression patterns"""
        progression = []
        
        if len(events) > 4:
            progression.append("Multi-stage activity progression detected")
        
        # Check for escalating severity
        severities = [event.get('severity') for event in events if event.get('severity')]
        if severities and len(set(severities)) > 1:
            progression.append("Escalating severity pattern observed")
        
        return progression
    
    def _calculate_timeline_risk(self, analysis: Dict) -> int:
        """Calculate timeline-based risk score"""
        risk_score = 0
        
        risk_score += len(analysis['anomalies']) * 25
        risk_score += len(analysis['correlations']) * 15
        risk_score += len(analysis['progression']) * 30
        
        if analysis['total_events'] > 10:
            risk_score += 20
        
        return min(risk_score, 100)
    
    def generate_timeline_report(self, temporal_context: Dict, timeline_analysis: Dict, timeline_events: List[Dict]) -> str:
        """Generate comprehensive timeline report"""
        
        report_sections = [
            f"TIMELINE ANALYSIS REPORT",
            f"Alert: {temporal_context.get('alert_info', {}).get('alert_name', 'Unknown')}",
            f"Time Span: {timeline_analysis['time_span']}",
            f"Total Events: {timeline_analysis['total_events']}",
            f"Timeline Risk Score: {timeline_analysis['risk_score']}/100",
            "",
            "SEQUENCE PATTERNS:" if timeline_analysis['patterns'] else "",
            *[f"• {pattern}" for pattern in timeline_analysis['patterns']],
            "",
            "EVENT CORRELATIONS:" if timeline_analysis['correlations'] else "",
            *[f"• {correlation}" for correlation in timeline_analysis['correlations']],
            "",
            "TEMPORAL ANOMALIES:" if timeline_analysis['anomalies'] else "",
            *[f"• {anomaly}" for anomaly in timeline_analysis['anomalies']],
            "",
            "PROGRESSION ANALYSIS:" if timeline_analysis['progression'] else "",
            *[f"• {progression}" for progression in timeline_analysis['progression']],
            "",
            "EVENT TIMELINE:",
            *[f"{i+1}. {event.get('description', 'Unknown event')} [{event.get('source', 'Unknown')}]" 
              for i, event in enumerate(timeline_events[:10])]  # Show first 10 events
        ]
        
        return '\n'.join([section for section in report_sections if section])
    
    def analyze_timeline(self, alert: Dict, data_entry: Dict, analysis_prompt: str, other_agent_data: List[Dict] = None) -> Dict[str, Any]:
        """Main function: Complete timeline analysis, robust to device/torch errors."""
        try:
            if other_agent_data is None:
                other_agent_data = []
            self.model, _ = to_cuda(self.model)
            temporal_context = self.extract_temporal_context(alert, data_entry)
            timeline_intent = self.analyze_timeline_intent(analysis_prompt)
            queries = self.generate_timeline_queries(temporal_context, timeline_intent, other_agent_data)
            query_results = {}
            for query in queries:
                es_response = self.request_es_query(query)
                results = self.execute_es_query(es_response.get('elasticsearch_query', {}))
                query_results[query['name']] = results
            timeline_analysis = self.analyze_timeline_patterns(query_results, other_agent_data, timeline_intent)
            report = self.generate_timeline_report(temporal_context, timeline_analysis, None)
            risk_level = timeline_analysis.get('risk_level', 'NORMAL')
            confidence_score = timeline_analysis.get('confidence_score', 0)
            result = {
                'agent_type': 'timeline',
                'report': report,
                'risk_level': risk_level,
                'confidence_score': confidence_score,
                'queries_executed': len(queries)
            }
            if 'es_context' in timeline_analysis:
                result['es_context'] = timeline_analysis['es_context']
            return result
        except Exception as e:
            return {
                'agent_type': 'timeline',
                'report': 'Timeline analysis could not be completed.',
                'risk_level': 'NORMAL',
                'confidence_score': 0,
                'queries_executed': 0
            }

# Usage
if __name__ == "__main__":
    print("=== Timeline Agent (Dynamic) ===")
    
    sample_alert = {
        'alertId': 'test-alert',
        'alertName': 'Security Incident',
        'severity': 'high',
        'createdAt': datetime.now().timestamp()
    }
    
    sample_data = {
        '_source': {
            'timestamp': datetime.now().timestamp(),
            'targetUsername': 'testuser',
            'srcIP': '192.168.1.100',
            'hostName': 'test-host'
        }
    }
    
    agent = TimelineAgent()
    result = agent.analyze_timeline(sample_alert, sample_data, "analyze chronological sequence of events")
    
    print(f"Timeline Categories: {result['timeline_intent']['categories']}")
    print(f"Total Events: {result['timeline_analysis']['total_events']}")
    print(f"Risk Score: {result['timeline_analysis']['risk_score']}")
    print("\nReport:")
    print(result['report'])