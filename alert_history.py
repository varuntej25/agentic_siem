import json
import logging
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime, timedelta
import requests
from gpu_utils import get_device, move_to_device, to_cuda, amp_autocast, get_optimal_batch_size, clear_cache

class AlertHistoryAgent:
    def __init__(self, es_query_agent_url: str = "http://10.9.56.10:30200",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.es_query_agent_url = es_query_agent_url
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.optimal_batch_size = get_optimal_batch_size(base_batch_size=16)
        # Setup logger
        self.logger = logging.getLogger("AlertHistoryAgent")
        if not self.logger.hasHandlers():
            logging.basicConfig(level=logging.INFO)
        self.logger.info(f"Using device for embedding: {self.device}")

        # Load event catalogue for event descriptions
        self.event_catalogue = {}
        try:
            import os
            catalogue_path = os.path.join(os.path.dirname(__file__), "Spharaka-Windows_Events_Filtered.json")
            with open(catalogue_path, "r", encoding="utf-8") as f:
                import json
                events = json.load(f)
                for entry in events:
                    eid = entry.get("event_id")
                    if eid is not None:
                        self.event_catalogue[int(eid)] = {
                            "event_name": entry.get("event_name", ""),
                            "description": entry.get("description", "")
                        }
        except Exception as e:
            self.logger.warning(f"Could not load event catalogue: {e}")
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for semantic analysis with GPU acceleration"""
        embeddings = []
        
        # Process in optimal batches
        batch_size = min(self.optimal_batch_size, len(texts))
        
        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(batch_texts, return_tensors='pt', truncation=True, 
                                      padding=True, max_length=512)
                
                # Move all tensors to the correct device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with amp_autocast():
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        embeddings.extend(batch_embeddings)
            
            # Clear cache periodically
            if len(embeddings) > 100:
                clear_cache()
                
        except Exception as e:
            print(f"Batch processing failed: {e}, falling back to single processing")
            # Fallback to original method
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                    embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def extract_alert_context(self, alert: Dict, data_entry: Dict) -> Dict[str, Any]:
        """Extract context for historical analysis"""
        source = data_entry.get('_source', {})
        context = {}
        
        # Extract alert characteristics
        context['current_alert'] = {
            'alert_id': alert.get('alertId'),
            'alert_name': alert.get('alertName'),
            'severity': alert.get('severity'),
            'alert_type': alert.get('alertType'),
            'threat_category': alert.get('threatCategory'),
            'technique_id': alert.get('techniqueId'),
            'tactic_id': alert.get('tacticId'),
            'description': alert.get('description'),
            'timestamp': alert.get('createdAt')
        }
        
        # Extract event context for pattern matching
        event_mapping = {
            'event_id': ['eventID', 'event_id', 'eventCode'],
            'user': ['targetUsername', 'userName', 'user', 'accountName'],
            'host': ['hostName', 'computerName', 'host'],
            'ip': ['srcIP', 'sourceIP', 'clientIP'],
            'process': ['processName', 'process', 'executable'],
            'domain': ['targetDomain', 'domain'],
            'logon_type': ['logonType', 'loginType'],
            'source_system': ['source', 'sourceSystem']
        }
        
        for context_key, possible_fields in event_mapping.items():
            for field in possible_fields:
                if source.get(field):
                    context[context_key] = source[field]
                    break
        
        return context
    
    def analyze_history_intent(self, analysis_prompt: str) -> Dict[str, Any]:
        """Analyze prompt to determine what type of historical analysis is needed"""
        prompt_embedding = self._get_embeddings([analysis_prompt])
        
        # Define historical analysis categories
        history_categories = {
            'pattern_matching': 'patterns similar previous alerts recurring incidents',
            'trend_analysis': 'trends frequency increase decrease historical trends',
            'attack_correlation': 'attack campaigns related incidents threat actors',
            'user_history': 'user historical behavior previous activities account history',
            'ip_reputation': 'IP reputation historical malicious previous incidents',
            'technique_analysis': 'techniques tactics procedures TTPs methodology',
            'seasonal_analysis': 'seasonal timing patterns cyclical historical timing',
            'escalation_patterns': 'escalation patterns progression severity increases'
        }
        
        # Get similarities
        category_descriptions = list(history_categories.values())
        category_embeddings = self._get_embeddings(category_descriptions)
        similarities = cosine_similarity(prompt_embedding, category_embeddings)[0]
        
        # Determine relevant categories
        threshold = 0.3
        relevant_categories = []
        category_keys = list(history_categories.keys())
        
        for i, similarity in enumerate(similarities):
            if similarity > threshold:
                relevant_categories.append(category_keys[i])
        
        # Default analysis if no specific intent
        if not relevant_categories:
            relevant_categories = ['pattern_matching', 'trend_analysis']
        
        # Determine lookback period
        lookback_period = self._extract_lookback_period(analysis_prompt)
        
        return {
            'categories': relevant_categories,
            'lookback_period': lookback_period,
            'focus': 'comprehensive' if len(relevant_categories) > 2 else relevant_categories[0]
        }
    
    def _extract_lookback_period(self, analysis_prompt: str) -> str:
        """Extract how far back to look in history"""
        prompt_lower = analysis_prompt.lower()
        
        if any(term in prompt_lower for term in ['week', '7 days']):
            return 'last_week'
        elif any(term in prompt_lower for term in ['month', '30 days']):
            return 'last_month'
        elif any(term in prompt_lower for term in ['quarter', '3 months', '90 days']):
            return 'last_quarter'
        elif any(term in prompt_lower for term in ['year', '12 months', '365 days']):
            return 'last_year'
        else:
            return 'last_6_months'  # Default lookback
    
    def generate_history_queries(self, alert_context: Dict, history_intent: Dict, analysis_prompt: str) -> List[Dict]:
        """Generate historical analysis queries"""
        queries = []
        current_alert = alert_context['current_alert']

        # Generate queries based on analysis categories
        for category in history_intent['categories']:
            if category == 'pattern_matching':
                # Similar alert patterns
                queries.append({
                    'name': 'similar_alerts',
                    'prompt': f"Find historical alerts similar to {current_alert['alert_name']} with same characteristics",
                    'focus': 'alert pattern matching'
                })

                # Same event ID patterns, with event description if available
                if alert_context.get('event_id'):
                    eid = alert_context['event_id']
                    event_desc = ""
                    if isinstance(eid, str) and eid.isdigit():
                        eid_int = int(eid)
                    else:
                        eid_int = eid if isinstance(eid, int) else None
                    if eid_int and eid_int in self.event_catalogue:
                        event_info = self.event_catalogue[eid_int]
                        event_desc = f" ({event_info['event_name']})"
                        if event_info['description'] and event_info['description'] != event_info['event_name']:
                            event_desc += f" - {event_info['description']}"
                    queries.append({
                        'name': f"event_history_{eid}",
                        'prompt': f"Find historical occurrences of event ID {eid}{event_desc} for pattern analysis",
                        'focus': 'event pattern history'
                    })
            elif category == 'trend_analysis':
                # Alert frequency trends
                queries.append({
                    'name': 'alert_frequency_trends',
                    'prompt': f"Analyze frequency trends for {current_alert['alert_type']} alerts over time",
                    'focus': 'frequency analysis'
                })
            elif category == 'attack_correlation':
                # Related attack patterns
                if current_alert.get('technique_id'):
                    queries.append({
                        'name': f"technique_history_{current_alert['technique_id']}",
                        'prompt': f"Find historical attacks using technique {current_alert['technique_id']}",
                        'focus': 'attack technique correlation'
                    })
            elif category == 'user_history':
                # User-specific history
                if alert_context.get('user'):
                    queries.append({
                        'name': f"user_alert_history_{alert_context['user']}",
                        'prompt': f"Find historical alerts and incidents involving user {alert_context['user']}",
                        'focus': 'user historical analysis'
                    })
            elif category == 'ip_reputation':
                # IP historical analysis
                if alert_context.get('ip'):
                    queries.append({
                        'name': f"ip_history_{alert_context['ip']}",
                        'prompt': f"Find historical incidents and reputation data for IP {alert_context['ip']}",
                        'focus': 'IP reputation history'
                    })
            elif category == 'technique_analysis':
                # TTPs analysis
                queries.append({
                    'name': 'ttp_analysis',
                    'prompt': f"Analyze historical usage of similar tactics, techniques, and procedures",
                    'focus': 'TTP historical analysis'
                })
            
            elif category == 'seasonal_analysis':
                # Seasonal patterns
                queries.append({
                    'name': 'seasonal_patterns',
                    'prompt': f"Analyze seasonal and timing patterns for similar alerts",
                    'focus': 'seasonal pattern analysis'
                })
            
            elif category == 'escalation_patterns':
                # Escalation analysis
                queries.append({
                    'name': 'escalation_history',
                    'prompt': f"Find historical escalation patterns and progression sequences",
                    'focus': 'escalation pattern analysis'
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
    
    def analyze_historical_patterns(self, query_results: Dict, alert_context: Dict, history_intent: Dict) -> Dict[str, Any]:
        """Analyze historical data for patterns and trends"""
        analysis = {
            'pattern_findings': [],
            'trend_analysis': [],
            'correlations': [],
            'risk_indicators': [],
            'historical_context': [],
            'threat_assessment': 'UNKNOWN',
            'confidence_score': 0
        }
        
        total_historical_incidents = 0
        
        # Analyze each query result
        for query_name, results in query_results.items():
            hits = results.get('hits', {}).get('hits', [])
            hit_count = results.get('hits', {}).get('total', {}).get('value', 0)
            total_historical_incidents += hit_count
            
            # Analyze based on query type
            if 'similar_alerts' in query_name:
                if hit_count > 10:
                    analysis['pattern_findings'].append(f"High frequency of similar alerts detected ({hit_count} historical incidents)")
                    analysis['confidence_score'] += 30
                elif hit_count > 3:
                    analysis['pattern_findings'].append(f"Recurring pattern identified ({hit_count} similar incidents)")
                    analysis['confidence_score'] += 20
                elif hit_count == 0:
                    analysis['historical_context'].append("No similar historical alerts found - potentially new threat")
                    analysis['confidence_score'] += 10
            
            elif 'event_history' in query_name:
                if hit_count > 50:
                    analysis['trend_analysis'].append(f"Event shows high historical frequency ({hit_count} occurrences)")
                    analysis['confidence_score'] += 25
                elif hit_count > 10:
                    analysis['trend_analysis'].append(f"Event has moderate historical activity ({hit_count} occurrences)")
                    analysis['confidence_score'] += 15
            
            elif 'user_alert_history' in query_name:
                if hit_count > 5:
                    analysis['risk_indicators'].append(f"User has significant alert history ({hit_count} previous alerts)")
                    analysis['confidence_score'] += 35
                elif hit_count > 0:
                    analysis['correlations'].append(f"User has some previous alert activity ({hit_count} alerts)")
                    analysis['confidence_score'] += 15
            
            elif 'ip_history' in query_name:
                if hit_count > 0:
                    analysis['risk_indicators'].append(f"IP has malicious history ({hit_count} previous incidents)")
                    analysis['confidence_score'] += 40
            
            elif 'technique_history' in query_name:
                if hit_count > 20:
                    analysis['correlations'].append(f"Attack technique shows high usage pattern ({hit_count} instances)")
                    analysis['confidence_score'] += 30
            
            elif 'frequency_trends' in query_name:
                if hit_count > 100:
                    analysis['trend_analysis'].append(f"Alert type shows increasing trend ({hit_count} recent incidents)")
                    analysis['confidence_score'] += 25
        
        # Determine threat assessment
        analysis['threat_assessment'] = self._assess_historical_threat(analysis, total_historical_incidents)
        
        return analysis
    
    def _assess_historical_threat(self, analysis: Dict, total_incidents: int) -> str:
        """Assess threat level based on historical analysis"""
        confidence = analysis['confidence_score']
        risk_indicators = len(analysis['risk_indicators'])
        
        if confidence >= 80 and risk_indicators >= 2:
            return "CONFIRMED_THREAT"
        elif confidence >= 60 and (risk_indicators >= 1 or total_incidents > 50):
            return "LIKELY_THREAT"
        elif confidence >= 40 or total_incidents > 20:
            return "POSSIBLE_THREAT"
        elif total_incidents == 0:
            return "NOVEL_ACTIVITY"
        else:
            return "BENIGN_PATTERN"
    
    def calculate_recurrence_risk(self, analysis: Dict, alert_context: Dict) -> Dict[str, Any]:
        """Calculate risk of recurrence based on historical patterns"""
        risk_factors = {
            'recurrence_probability': 'LOW',
            'risk_factors': [],
            'mitigation_urgency': 'NORMAL'
        }
        
        # Analyze recurrence indicators
        pattern_count = len(analysis['pattern_findings'])
        risk_indicator_count = len(analysis['risk_indicators'])
        
        if pattern_count >= 2 and risk_indicator_count >= 2:
            risk_factors['recurrence_probability'] = 'HIGH'
            risk_factors['mitigation_urgency'] = 'URGENT'
            risk_factors['risk_factors'].append('Multiple historical patterns detected')
        elif pattern_count >= 1 or risk_indicator_count >= 1:
            risk_factors['recurrence_probability'] = 'MEDIUM'
            risk_factors['mitigation_urgency'] = 'ELEVATED'
        
        # Add specific risk factors
        if any('user has' in finding.lower() for finding in analysis['risk_indicators']):
            risk_factors['risk_factors'].append('User with previous incident history')
        
        if any('ip has' in finding.lower() for finding in analysis['risk_indicators']):
            risk_factors['risk_factors'].append('IP with malicious history')
        
        if any('high frequency' in finding.lower() for finding in analysis['pattern_findings']):
            risk_factors['risk_factors'].append('High frequency attack pattern')
        
        return risk_factors
    
    def generate_history_report(self, alert_context: Dict, analysis: Dict, recurrence_risk: Dict, history_intent: Dict) -> str:
        """Generate comprehensive historical analysis report"""
        current_alert = alert_context['current_alert']
        
        report_sections = [
            f"ALERT HISTORY ANALYSIS REPORT",
            f"Current Alert: {current_alert['alert_name']}",
            f"Alert Type: {current_alert.get('alert_type', 'Unknown')}",
            f"Threat Assessment: {analysis['threat_assessment']}",
            f"Historical Confidence: {analysis['confidence_score']}/100",
            f"Recurrence Risk: {recurrence_risk['recurrence_probability']}",
            f"Mitigation Urgency: {recurrence_risk['mitigation_urgency']}",
            "",
            "HISTORICAL PATTERNS:" if analysis['pattern_findings'] else "",
            *[f"• {finding}" for finding in analysis['pattern_findings']],
            "",
            "TREND ANALYSIS:" if analysis['trend_analysis'] else "",
            *[f"• {trend}" for trend in analysis['trend_analysis']],
            "",
            "RISK INDICATORS:" if analysis['risk_indicators'] else "",
            *[f"• {risk}" for risk in analysis['risk_indicators']],
            "",
            "CORRELATIONS:" if analysis['correlations'] else "",
            *[f"• {correlation}" for correlation in analysis['correlations']],
            "",
            "HISTORICAL CONTEXT:" if analysis['historical_context'] else "",
            *[f"• {context}" for context in analysis['historical_context']],
            "",
            "RECURRENCE RISK FACTORS:" if recurrence_risk['risk_factors'] else "",
            *[f"• {factor}" for factor in recurrence_risk['risk_factors']]
        ]
        
        return '\n'.join([section for section in report_sections if section])
    
    def analyze_alert_history(self, alert: Dict, data_entry: Dict, analysis_prompt: str) -> Dict[str, Any]:
        """Main function: Complete alert history analysis, robust to errors and always returns clean output."""
        try:
            alert_context = self.extract_alert_context(alert, data_entry)
            history_intent = self.analyze_history_intent(analysis_prompt)
            queries = self.generate_history_queries(alert_context, history_intent, analysis_prompt)
            query_results = {}
            for query in queries:
                es_response = self.request_es_query(query)
                results = self.execute_es_query(es_response.get('elasticsearch_query', {}))
                query_results[query['name']] = results
            historical_analysis = self.analyze_historical_patterns(query_results, alert_context, history_intent)
            recurrence_risk = self.calculate_recurrence_risk(historical_analysis, alert_context)
            report = self.generate_history_report(alert_context, historical_analysis, recurrence_risk, history_intent)
            # Map threat_assessment to risk_level
            risk_level = historical_analysis.get('threat_assessment', 'NORMAL')
            confidence_score = historical_analysis.get('confidence_score', 0)
            result = {
                'agent_type': 'alert_history',
                'report': report,
                'risk_level': risk_level,
                'confidence_score': confidence_score,
                'queries_executed': len(queries)
            }
            if 'es_context' in historical_analysis:
                result['es_context'] = historical_analysis['es_context']
            return result
        except Exception as e:
            self.logger.error(f"analyze_alert_history error: {e}", exc_info=True)
            return {
                'agent_type': 'alert_history',
                'report': "Alert history analysis could not be completed.",
                'risk_level': 'NORMAL',
                'confidence_score': 0,
                'queries_executed': 0
            }

# Usage
if __name__ == "__main__":
    print("=== Alert History Agent (Dynamic) ===")
    
    sample_alert = {
        'alertId': 'test-alert',
        'alertName': 'Suspicious Login Activity',
        'severity': 'high',
        'alertType': 'authentication',
        'threatCategory': 'credential_access',
        'techniqueId': 'T1078',
        'createdAt': datetime.now().timestamp()
    }
    
    sample_data = {
        '_source': {
            'eventID': 4625,
            'targetUsername': 'testuser',
            'srcIP': '192.168.1.100',
            'hostName': 'test-host'
        }
    }
    
    agent = AlertHistoryAgent()
    result = agent.analyze_alert_history(sample_alert, sample_data, "find historical patterns and similar incidents")
    
    print(f"Threat Assessment: {result['threat_assessment']}")
    print(f"Confidence Score: {result['confidence_score']}")
    print(f"Recurrence Risk: {result['recurrence_risk']['recurrence_probability']}")
    print("\nReport:")
    print(result['report'])