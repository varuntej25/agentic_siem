import json
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from gpu_utils import to_cuda, move_to_device, amp_autocast

class IPAgent:

    def __init__(self, es_query_agent_url: str = "http://10.9.56.10:30200",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        import logging
        self.es_query_agent_url = es_query_agent_url
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Force GPU placement
        self.model, _ = to_cuda(self.model)
        # Setup logger
        self.logger = logging.getLogger("IPAgent")
        if not self.logger.hasHandlers():
            logging.basicConfig(level=logging.INFO)
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for semantic analysis, with error suppression and device consistency."""
        embeddings = []
        for text in texts:
            try:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                # Move all tensors to the correct device (GPU/CPU)
                inputs = {k: v.to(self.model.device) if hasattr(self.model, 'device') else v for k, v in inputs.items()}
                with amp_autocast():
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                        embeddings.append(embedding)
            except Exception as e:
                if hasattr(self, 'suppress_internal_errors') and self.suppress_internal_errors:
                    self.logger.error(f"Embedding error: {e}")
                    return np.zeros((len(texts), 384))  # Safe fallback: all-zeros (384-dim for MiniLM)
                else:
                    raise
        return np.array(embeddings)
    
    def extract_ip_context(self, alert: Dict, data_entry: Dict) -> Dict[str, Any]:
        """Extract all available IP and network-related context dynamically"""
        source = data_entry.get('_source', {})
        context = {}
        
        # Extract IP-related fields dynamically
        ip_fields = ['srcIP', 'sourceIP', 'clientIP', 'remoteIP', 'destinationIP', 'destIP']
        for field in ip_fields:
            if source.get(field):
                context['primary_ip'] = source[field]
                break
        
        # Extract all network-related context
        network_mapping = {
            'host': ['hostName', 'computerName', 'host', 'machine', 'hostname'],
            'port': ['srcPort', 'sourcePort', 'destPort', 'destinationPort', 'port'],
            'protocol': ['protocol', 'networkProtocol', 'ipProtocol'],
            'connection_type': ['logonType', 'connectionType', 'accessType'],
            'process': ['processName', 'process', 'executable', 'serviceName'],
            'user': ['targetUsername', 'userName', 'user', 'accountName'],
            'domain': ['targetDomain', 'domain', 'workgroup'],
            'location': ['location', 'geoLocation', 'country', 'city']
        }
        
        for context_key, possible_fields in network_mapping.items():
            for field in possible_fields:
                if source.get(field):
                    context[context_key] = source[field]
                    break
        
        # Add alert context
        context['alert_info'] = {
            'alert_id': alert.get('alertId'),
            'alert_name': alert.get('alertName'),
            'severity': alert.get('severity'),
            'timestamp': alert.get('createdAt')
        }
        
        return context
    
    def analyze_ip_intent(self, analysis_prompt: str) -> List[str]:
        """Analyze prompt to determine what type of IP analysis is needed"""
        prompt_embedding = self._get_embeddings([analysis_prompt])
        
        # Define IP analysis categories
        ip_categories = {
            'traffic_analysis': 'network traffic connections bandwidth volume patterns flows',
            'connection_analysis': 'connections sessions endpoints remote access established',
            'geographic_analysis': 'location geographic geolocation country region origin',
            'reputation_analysis': 'reputation threat intelligence malicious blacklist suspicious',
            'behavior_analysis': 'behavior patterns anomalies unusual activities reconnaissance',
            'communication_analysis': 'communication protocols ports services network communication',
            'temporal_analysis': 'time patterns timing frequency chronological temporal',
            'security_analysis': 'security threats attacks intrusion malware indicators'
        }
        
        # Get similarities
        category_descriptions = list(ip_categories.values())
        category_embeddings = self._get_embeddings(category_descriptions)
        similarities = cosine_similarity(prompt_embedding, category_embeddings)[0]
        
        # Return categories above threshold
        threshold = 0.3
        relevant_categories = []
        category_keys = list(ip_categories.keys())
        
        for i, similarity in enumerate(similarities):
            if similarity > threshold:
                relevant_categories.append(category_keys[i])
        
        # Default to basic analysis if no specific intent
        if not relevant_categories:
            relevant_categories = ['connection_analysis', 'behavior_analysis']
            
        return relevant_categories
    
    def generate_ip_queries(self, ip_context: Dict, analysis_categories: List[str], analysis_prompt: str) -> List[Dict]:
        """Generate IP-focused queries based on context and analysis intent"""
        queries = []
        primary_ip = ip_context.get('primary_ip')
        
        if not primary_ip:
            return queries
        
        # Generate queries based on analysis categories
        for category in analysis_categories:
            hint = None
            if category == 'traffic_analysis':
                hint = "Use events related to connections or bandwidth anomalies if applicable."
                prompt = f"Analyze network traffic patterns and volume for IP {primary_ip} to identify anomalies."
            elif category == 'connection_analysis':
                hint = "Focus on events describing new or unusual network connections."
                prompt = f"Find all network connections and sessions originating from IP {primary_ip}."
            elif category == 'geographic_analysis':
                hint = "Consider events that log geolocation or country of origin."
                prompt = f"Investigate geographic location and origin details for IP {primary_ip}."
            elif category == 'reputation_analysis':
                hint = "Use threat intelligence or reputation-related events if available."
                prompt = f"Check threat intelligence and reputation data for IP {primary_ip}."
            elif category == 'behavior_analysis':
                hint = "Look for events indicating behavioral anomalies or suspicious activity."
                prompt = f"Analyze behavioral patterns and anomalies for IP {primary_ip}."
            elif category == 'communication_analysis':
                hint = "Focus on protocol, port, or service usage events."
                prompt = f"Investigate communication protocols and services used by IP {primary_ip}."
            elif category == 'temporal_analysis':
                hint = "Consider events with time-based or frequency information."
                prompt = f"Analyze time-based patterns and frequency of activities from IP {primary_ip}."
            elif category == 'security_analysis':
                hint = "Use events related to security threats, attacks, or indicators of compromise."
                prompt = f"Search for security threats and attack indicators from IP {primary_ip}."
            else:
                prompt = f"Analyze IP {primary_ip}."
            if hint:
                prompt = prompt.strip() + "\n" + hint
            queries.append({
                'name': f'{category}_{primary_ip}',
                'prompt': prompt.strip(),
                'focus': category.replace('_', ' ')
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
    
    def analyze_ip_results(self, query_results: Dict, analysis_prompt: str) -> Dict[str, Any]:
        """Analyze IP results using semantic understanding"""
        analysis = {
            'findings': [],
            'risk_score': 0,
            'network_patterns': [],
            'security_indicators': [],
            'anomalies': []
        }
        
        # Analyze each query result semantically
        for query_name, results in query_results.items():
            total_hits = results.get('hits', {}).get('total', {}).get('value', 0)
            
            # Semantic analysis based on query type - Updated realistic thresholds
            if 'traffic' in query_name:
                if total_hits > 10:
                    analysis['anomalies'].append(f'High traffic volume detected ({total_hits} connections)')
                    analysis['risk_score'] += 30
                elif total_hits > 3:
                    analysis['network_patterns'].append(f'Moderate traffic activity ({total_hits} connections)')
                    analysis['risk_score'] += 15
                elif total_hits > 0:
                    analysis['findings'].append(f'Traffic activity detected: {total_hits} connections')
                    analysis['network_patterns'].append(f'Limited traffic volume ({total_hits} connections)')
                    analysis['risk_score'] += 5
                else:
                    analysis['findings'].append('No direct traffic found for this IP')
            
            elif 'connection' in query_name:
                if total_hits > 20:
                    analysis['security_indicators'].append(f'Multiple connections detected ({total_hits} sessions)')
                    analysis['risk_score'] += 25
                elif total_hits > 5:
                    analysis['anomalies'].append(f'Elevated connection activity ({total_hits} sessions)')
                    analysis['risk_score'] += 12
                elif total_hits > 0:
                    analysis['findings'].append(f'Connection activity detected: {total_hits} sessions')
                    analysis['risk_score'] += 3
                    analysis['risk_score'] += 15
            
            elif 'behavior' in query_name:
                if total_hits > 200:
                    analysis['security_indicators'].append(f'Unusual behavioral patterns detected ({total_hits} events)')
                    analysis['risk_score'] += 35
                elif total_hits > 50:
                    analysis['anomalies'].append(f'Some behavioral anomalies detected ({total_hits} events)')
                    analysis['risk_score'] += 20
            
            elif 'security' in query_name:
                if total_hits > 0:
                    analysis['security_indicators'].append(f'Security threats identified ({total_hits} indicators)')
                    analysis['risk_score'] += 40
            
            elif 'reputation' in query_name:
                if total_hits > 0:
                    analysis['security_indicators'].append(f'Reputation issues found ({total_hits} reports)')
                    analysis['risk_score'] += 45
        
        return analysis
    
    def determine_ip_risk(self, risk_score: int) -> str:
        """Determine IP risk level based on score"""
        if risk_score >= 80:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 40:
            return "MEDIUM"
        elif risk_score >= 20:
            return "LOW"
        else:
            return "NORMAL"
    
    def generate_ip_report(self, ip_context: Dict, analysis: Dict, risk_level: str, analysis_prompt: str) -> str:
        """Generate dynamic IP analysis report"""
        primary_ip = ip_context.get('primary_ip', 'Unknown')
        host = ip_context.get('host', 'Unknown')
        
        report_sections = [
            f"IP CONTEXT ANALYSIS",
            f"IP Address: {primary_ip}",
            f"Host: {host}",
            f"Risk Level: {risk_level}",
            f"Risk Score: {analysis['risk_score']}/100",
            f"Analysis Focus: {analysis_prompt}",
            "",
            "NETWORK FINDINGS:" if analysis['findings'] else "",
            *[f"• {finding}" for finding in analysis['findings']],
            "",
            "SECURITY INDICATORS:" if analysis['security_indicators'] else "",
            *[f"• {indicator}" for indicator in analysis['security_indicators']],
            "",
            "ANOMALIES:" if analysis['anomalies'] else "",
            *[f"• {anomaly}" for anomaly in analysis['anomalies']],
            "",
            "NETWORK PATTERNS:" if analysis['network_patterns'] else "",
            *[f"• {pattern}" for pattern in analysis['network_patterns']]
        ]
        
        return '\n'.join([section for section in report_sections if section])
    
    def analyze_ip(self, alert: Dict, data_entry: Dict, analysis_prompt: str) -> Dict[str, Any]:
        """Main function: Complete IP context analysis. Standardized return format."""
        try:
            ip_context = self.extract_ip_context(alert, data_entry)
            analysis_categories = self.analyze_ip_intent(analysis_prompt)
            queries = self.generate_ip_queries(ip_context, analysis_categories, analysis_prompt)
            query_results = {}
            for query in queries:
                es_response = self.request_es_query(query)
                results = self.execute_es_query(es_response.get('elasticsearch_query', {}))
                query_results[query['name']] = results
            analysis = self.analyze_ip_results(query_results, analysis_prompt)
            risk_level = self.determine_ip_risk(analysis.get('risk_score', 0))
            confidence_score = analysis.get('confidence_score', 0)
            report = self.generate_ip_report(ip_context, analysis, risk_level, analysis_prompt)
            result = {
                'agent_type': 'ip_agent',
                'report': report,
                'risk_level': risk_level,
                'confidence_score': confidence_score,
                'queries_executed': len(queries)
            }
            if 'es_context' in analysis:
                result['es_context'] = analysis['es_context']
            return result
        except Exception as e:
            if hasattr(self, 'suppress_internal_errors') and self.suppress_internal_errors:
                self.logger.error(f"analyze_ip error: {e}")
                return {
                    'agent_type': 'ip_agent',
                    'report': '',
                    'risk_level': 'NORMAL',
                    'confidence_score': 0,
                    'queries_executed': 0
                }
            else:
                raise

# Usage
if __name__ == "__main__":
    print("=== IP Agent (Dynamic) ===")
    
    sample_alert = {
        'alertId': 'test-alert',
        'alertName': 'Network Anomaly',
        'severity': 'high'
    }
    
    sample_data = {
        '_source': {
            'srcIP': '192.168.1.100',
            'hostName': 'test-host',
            'targetUsername': 'testuser'
        }
    }
    
    agent = IPAgent()
    result = agent.analyze_ip(sample_alert, sample_data, "investigate IP network behavior")
    # Print only if keys exist
    if 'analysis_categories' in result:
        print(f"Analysis Categories: {result['analysis_categories']}")
    if 'risk_level' in result:
        print(f"Risk Level: {result['risk_level']}")
    print("\nReport:")
    print(result.get('report', ''))