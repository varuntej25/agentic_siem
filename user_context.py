import json
import logging
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from gpu_utils import to_cuda, move_to_device, amp_autocast

class UserContextAgent:
    def __init__(self, es_query_agent_url: str = "http://10.9.56.10:30200", 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 suppress_internal_errors: bool = False):
        self.es_query_agent_url = es_query_agent_url
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        # Setup logger
        self.logger = logging.getLogger("UserContextAgent")
        if not self.logger.hasHandlers():
            logging.basicConfig(level=logging.INFO)
        self.logger.info(f"Using device for embedding: {self.device}")
        self.suppress_internal_errors = suppress_internal_errors
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for semantic analysis, with error suppression."""
        embeddings = []
        for text in texts:
            try:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                # Move all tensors to the correct device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with amp_autocast():
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                        embeddings.append(embedding)
            except Exception as e:
                if self.suppress_internal_errors:
                    self.logger.error(f"Embedding error: {e}")
                    return np.zeros((len(texts), 384))  # Safe fallback: all-zeros (384-dim for MiniLM)
                else:
                    raise
        return np.array(embeddings)
    
    def extract_user_context(self, alert: Dict, data_entry: Dict) -> Dict[str, Any]:
        """Extract all available user-related context dynamically"""
        source = data_entry.get('_source', {})
        context = {}
        
        # Extract all user-related fields dynamically
        user_fields = ['targetUsername', 'accountName', 'subjectUserName', 'userName', 'user']
        for field in user_fields:
            if source.get(field):
                context['username'] = source[field]
                break
        
        # Extract all available context fields
        context_mapping = {
            'domain': ['targetDomain', 'domain', 'subjectDomainName'],
            'host': ['hostName', 'computerName', 'host', 'machine'],
            'logon_type': ['logonType', 'loginType'],
            'process': ['processName', 'process', 'executable'],
            'privileges': ['privilegeList', 'privileges'],
            'ip': ['srcIP', 'sourceIP', 'clientIP', 'remoteIP'],
            'event_id': ['eventID', 'event_id', 'eventCode']
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
            'severity': alert.get('severity'),
            'timestamp': alert.get('createdAt')
        }
        
        return context
    
    def analyze_query_intent(self, analysis_prompt: str) -> List[str]:
        """Analyze the analysis prompt to determine what type of user queries are needed (intent categories)"""
        try:
            prompt_embedding = self._get_embeddings([analysis_prompt])
            # Define query categories
            query_categories = {
                'authentication_analysis': 'authentication login logon attempts patterns failures',
                'activity_analysis': 'user activities behavior patterns recent actions',
                'privilege_analysis': 'privileges permissions access rights escalation changes',
                'temporal_analysis': 'time patterns schedule timing anomalies chronological',
                'account_analysis': 'account creation modification changes password reset',
                'network_analysis': 'network connections remote access sessions'
            }
            # Get similarities
            category_descriptions = list(query_categories.values())
            category_embeddings = self._get_embeddings(category_descriptions)
            similarities = cosine_similarity(prompt_embedding, category_embeddings)[0]
            # Return categories above threshold
            threshold = 0.3
            relevant_categories = []
            category_keys = list(query_categories.keys())
            for i, similarity in enumerate(similarities):
                if similarity > threshold:
                    relevant_categories.append(category_keys[i])
            # Default to basic analysis if no specific intent
            if not relevant_categories:
                relevant_categories = ['authentication_analysis', 'activity_analysis']
            return relevant_categories
        except Exception as e:
            if self.suppress_internal_errors:
                self.logger.error(f"analyze_query_intent error: {e}")
                return ['authentication_analysis', 'activity_analysis']
            else:
                raise
    
    def generate_dynamic_queries(self, user_context: Dict, intent_categories: List[str], analysis_prompt: str) -> List[Dict]:
        """Generate queries based on available context and analysis intent, and append relevant event IDs to each query prompt."""
        try:
            queries = []
            username = user_context.get('username')
            host = user_context.get('host') or user_context.get('host_name')
            domain = user_context.get('domain') or user_context.get('domain_name')
            ip = user_context.get('ip') or user_context.get('src_ip')
            if not username:
                return queries
            def context_str():
                ctx = []
                if host:
                    ctx.append(f"on host {host}")
                if domain:
                    ctx.append(f"in domain {domain}")
                if ip:
                    ctx.append(f"from IP {ip}")
                return " ".join(ctx)
            # --- Event Catalog Loading and Embedding Similarity ---
            if not hasattr(self, '_event_catalog'):
                with open('Spharaka-Windows_Events_Filtered.json', 'r', encoding='utf-8') as f:
                    self._event_catalog = json.load(f)
            event_catalog = self._event_catalog
            event_texts = [f"{e.get('event_name','')} {e.get('description','')}" for e in event_catalog]
            try:
                event_embeddings = self._get_embeddings(event_texts)
                prompt_embedding = self._get_embeddings([analysis_prompt])[0].reshape(1, -1)
                similarities = cosine_similarity(prompt_embedding, event_embeddings)[0]
                top_indices = similarities.argsort()[-5:][::-1]
                relevant_events = []
                for idx in top_indices:
                    event = event_catalog[idx]
                    eid = event.get('event_id')
                    ename = event.get('event_name')
                    if eid and ename:
                        relevant_events.append(f"{eid} ({ename})")
                if relevant_events:
                    event_line = "Relevant Event IDs: " + ", ".join(relevant_events)
                else:
                    event_line = None
            except Exception as e:
                if self.suppress_internal_errors:
                    self.logger.error(f"Event embedding error: {e}")
                    event_line = None
                else:
                    raise
            exclusion_phrases = ["other users", "excluding", "besides user", "except user", "apart from user", "not user"]
            exclusion_detected = any(phrase in analysis_prompt.lower() for phrase in exclusion_phrases)
            for category in intent_categories:
                ctx = context_str()
                if category == 'authentication_analysis':
                    if exclusion_detected:
                        prompt = f"Find users who logged in {ctx} excluding {username}."
                    else:
                        prompt = f"Analyze authentication patterns and login behavior for user {username} {ctx} including successes and failures."
                elif category == 'activity_analysis':
                    if exclusion_detected:
                        prompt = f"Find user activities {ctx} excluding {username}."
                    else:
                        prompt = f"Find all user activities and behavioral patterns for {username} {ctx} to identify anomalies."
                elif category == 'privilege_analysis':
                    prompt = f"Investigate privilege changes and access modifications for user {username} {ctx}."
                elif category == 'temporal_analysis':
                    prompt = f"Analyze time-based patterns and schedule anomalies for user {username} {ctx}."
                elif category == 'account_analysis':
                    prompt = f"Find account modifications and changes for user {username} {ctx}."
                elif category == 'network_analysis':
                    prompt = f"Investigate network behavior and remote access patterns for user {username} {ctx}."
                else:
                    prompt = f"Analyze user {username} {ctx}."
                if event_line:
                    prompt = prompt.strip() + "\n" + event_line
                queries.append({
                    'name': f'{category}_{username}',
                    'prompt': prompt.strip(),
                    'focus': category.replace('_', ' ')
                })
            return queries
        except Exception as e:
            if self.suppress_internal_errors:
                self.logger.error(f"generate_dynamic_queries error: {e}")
                return []
            else:
                raise
    
    def request_es_query(self, query_request: Dict, analysis_prompt: str = None) -> Dict:
        """Request ES query from Query Agent, always include analysis_prompt, and log intent/dispatch."""
        # Always include analysis_prompt in the request
        if analysis_prompt is not None:
            query_request = dict(query_request)  # shallow copy
            query_request['analysis_prompt'] = analysis_prompt
        self.logger.info(f"Dispatching ES Query: intent={query_request.get('focus','')} | prompt={query_request.get('prompt','')} | analysis_prompt={analysis_prompt}")
        try:
            response = requests.post(
                f"{self.es_query_agent_url}/generate_query",
                json=query_request,
                timeout=30
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"ES Query dispatch failed: {e}")
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
    
    def analyze_results_semantically(self, query_results: Dict, analysis_prompt: str) -> Dict[str, Any]:
        """Analyze results using semantic understanding instead of hardcoded rules"""
        analysis = {
            'findings': [],
            'risk_score': 0,
            'patterns': [],
            'anomalies': []
        }
        
        # Analyze each query result
        for query_name, results in query_results.items():
            total_hits = results.get('hits', {}).get('total', {}).get('value', 0)
            
            # Semantic analysis of result volume
            if 'authentication' in query_name:
                if total_hits > 50:
                    analysis['findings'].append(f'High authentication activity detected ({total_hits} events)')
                    analysis['risk_score'] += 25
                elif total_hits == 0:
                    analysis['findings'].append('No authentication activity found')
                    analysis['risk_score'] += 15
                else:
                    analysis['patterns'].append(f'Normal authentication activity ({total_hits} events)')
            
            elif 'privilege' in query_name:
                if total_hits > 0:
                    analysis['anomalies'].append(f'Privilege changes detected ({total_hits} events)')
                    analysis['risk_score'] += 30
            
            elif 'activity' in query_name:
                if total_hits > 100:
                    analysis['anomalies'].append(f'Unusually high user activity ({total_hits} events)')
                    analysis['risk_score'] += 20
                elif total_hits < 5:
                    analysis['anomalies'].append(f'Unusually low user activity ({total_hits} events)')
                    analysis['risk_score'] += 10
        
        return analysis
    
    def determine_risk_level(self, risk_score: int) -> str:
        """Determine risk level based on score"""
        if risk_score >= 70:
            return "CRITICAL"
        elif risk_score >= 50:
            return "HIGH" 
        elif risk_score >= 30:
            return "MEDIUM"
        elif risk_score >= 15:
            return "LOW"
        else:
            return "NORMAL"
    
    def generate_report(self, user_context: Dict, analysis: Dict, risk_level: str, analysis_prompt: str) -> str:
        """Generate dynamic report based on analysis"""
        username = user_context.get('username', 'Unknown')
        
        report_sections = [
            f"USER CONTEXT ANALYSIS",
            f"User: {username}",
            f"Risk Level: {risk_level}",
            f"Risk Score: {analysis['risk_score']}/100",
            f"Analysis Focus: {analysis_prompt}",
            "",
            "FINDINGS:" if analysis['findings'] else "",
            *[f"• {finding}" for finding in analysis['findings']],
            "",
            "ANOMALIES:" if analysis['anomalies'] else "",
            *[f"• {anomaly}" for anomaly in analysis['anomalies']],
            "",
            "PATTERNS:" if analysis['patterns'] else "", 
            *[f"• {pattern}" for pattern in analysis['patterns']]
        ]
        
        return '\n'.join([section for section in report_sections if section])
    
    def analyze_user(self, alert: Dict, data_entry: Dict, analysis_prompt: str) -> Dict[str, Any]:
        """Main function: Complete user context analysis. Only return user-facing fields, never internal errors."""
        try:
            user_context = self.extract_user_context(alert, data_entry)
            intent_categories = self.analyze_query_intent(analysis_prompt)
            queries = self.generate_dynamic_queries(user_context, intent_categories, analysis_prompt)
            query_results = {}
            for query in queries:
                es_response = self.request_es_query(query, analysis_prompt=analysis_prompt)
                results = self.execute_es_query(es_response.get('elasticsearch_query', {}))
                query_results[query['name']] = results
            analysis = self.analyze_results_semantically(query_results, analysis_prompt)
            risk_level = self.determine_risk_level(analysis.get('risk_score', 0))
            confidence_score = analysis.get('confidence_score', 0)
            report = self.generate_report(user_context, analysis, risk_level, analysis_prompt)
            result = {
                'agent_type': 'user_context',
                'report': report,
                'risk_level': risk_level,
                'confidence_score': confidence_score,
                'queries_executed': len(queries)
            }
            if 'es_context' in analysis:
                result['es_context'] = analysis['es_context']
            return result
        except Exception as e:
            if self.suppress_internal_errors:
                self.logger.error(f"analyze_user error: {e}")
                return {
                    'agent_type': 'user_context',
                    'report': '',
                    'risk_level': 'NORMAL',
                    'confidence_score': 0,
                    'queries_executed': 0
                }
            else:
                raise

# Usage
if __name__ == "__main__":
    print("=== User Context Agent (Dynamic) ===")
    
    sample_alert = {
        'alertId': 'test-alert',
        'alertName': 'Suspicious Activity',
        'severity': 'high'
    }
    
    sample_data = {
        '_source': {
            'targetUsername': 'testuser',
            'targetDomain': 'TESTDOMAIN',
            'hostName': 'test-host'
        }
    }
    
    agent = UserContextAgent()
    result = agent.analyze_user(sample_alert, sample_data, "investigate user authentication patterns")
    # Print only if keys exist
    if 'intent_categories' in result:
        print(f"Intent Categories: {result['intent_categories']}")
    if 'risk_level' in result:
        print(f"Risk Level: {result['risk_level']}")
    print("\nReport:")
    print(result.get('report', ''))