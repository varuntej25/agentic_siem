# enhanced_agent_base.py - Base class for agents with ES integration
import json
import requests
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from gpu_utils import get_device, move_to_device, to_cuda, amp_autocast, get_optimal_batch_size, clear_cache

class BaseAgentWithES:
    """Base class for all agents with Elasticsearch integration capabilities"""
    
    def __init__(self, es_query_agent_url: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.es_query_agent_url = es_query_agent_url
        self.model_name = model_name
        self.device = get_device()
        self.optimal_batch_size = get_optimal_batch_size(base_batch_size=16)
        self.setup_logging()
        self.init_embeddings()
        
    def setup_logging(self):
        """Setup agent-specific logging"""
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def init_embeddings(self):
        """Initialize embedding model for semantic analysis with GPU optimization"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Force GPU placement
            self.model, _ = to_cuda(self.model)
            
        except Exception as e:
            self.logger.warning(f"Could not initialize embeddings: {e}")
            self.tokenizer = None
            self.model = None
    
    def execute_es_query(self, query: Dict, timeout: int = 30) -> Dict[str, Any]:
        """Execute Elasticsearch query against loaded data entries"""
        try:
            # Use the data entries that are already loaded in the orchestrator
            # These should be passed via the es_context
            if hasattr(self, 'current_es_context') and 'all_data_entries' in self.current_es_context:
                data_entries = self.current_es_context['all_data_entries']
                self.logger.info(f"Executing ES query against {len(data_entries)} loaded data entries")
            else:
                self.logger.warning("No data entries available in ES context - query cannot be executed")
                return {"hits": {"hits": [], "total": {"value": 0}}}
            
            # Execute the query by filtering the loaded data
            results = self._execute_query_on_data(query, data_entries)
            
            self.logger.info(f"ES query found {results['hits']['total']['value']} matching results")
            return results
                
        except Exception as e:
            self.logger.error(f"ES query execution error: {e}")
            return {"error": str(e), "hits": {"hits": [], "total": {"value": 0}}}

    def _execute_query_on_data(self, query: Dict, data_entries: List[Dict]) -> Dict[str, Any]:
        """Execute ES query logic on loaded data entries"""
        try:
            matching_results = []
            query_obj = query.get('query', {})
            size_limit = query.get('size', 100)
            
            # Process the ES query structure to find matches
            for entry in data_entries:
                if self._entry_matches_query(entry, query_obj):
                    matching_results.append({
                        '_source': entry.get('_source', {}),
                        '_score': 1.0,
                        '_id': entry.get('_id', 'unknown'),
                        '_index': entry.get('_index', 'unknown'),
                        '_type': entry.get('_type', '_doc')
                    })
                    
                    if len(matching_results) >= size_limit:
                        break
            
            return {
                "hits": {
                    "hits": matching_results,
                    "total": {"value": len(matching_results), "relation": "eq"}
                },
                "took": 5,
                "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
                "timed_out": False
            }
            
        except Exception as e:
            self.logger.error(f"Error executing query on data: {e}")
            return {"hits": {"hits": [], "total": {"value": 0}}}
    
    def _entry_matches_query(self, entry: Dict, query_obj: Dict) -> bool:
        """Check if a data entry matches the ES query structure"""
        try:
            source = entry.get('_source', {})
            
            # Handle bool queries (most common structure)
            if 'bool' in query_obj:
                return self._matches_bool_query(source, query_obj['bool'])
            
            # Handle term queries
            elif 'term' in query_obj:
                return self._matches_term_query(source, query_obj['term'])
            
            # Handle terms queries (multiple values)
            elif 'terms' in query_obj:
                return self._matches_terms_query(source, query_obj['terms'])
            
            # Handle match queries
            elif 'match' in query_obj:
                return self._matches_match_query(source, query_obj['match'])
            
            # Handle range queries
            elif 'range' in query_obj:
                return self._matches_range_query(source, query_obj['range'])
            
            # Default: match all if no specific query
            else:
                return True
                
        except Exception as e:
            self.logger.error(f"Error matching entry against query: {e}")
            return False
    
    def _matches_bool_query(self, source: Dict, bool_query: Dict) -> bool:
        """Check if source matches bool query structure"""
        # Handle 'should' clauses (OR logic)
        if 'should' in bool_query:
            should_clauses = bool_query['should']
            for clause in should_clauses:
                if self._entry_matches_query({'_source': source}, clause):
                    return True
            # If no should clauses match, check if minimum_should_match is satisfied
            return False
        
        # Handle 'must' clauses (AND logic)
        if 'must' in bool_query:
            must_clauses = bool_query['must']
            for clause in must_clauses:
                if not self._entry_matches_query({'_source': source}, clause):
                    return False
            return True
        
        # Handle 'filter' clauses
        if 'filter' in bool_query:
            filter_clauses = bool_query['filter']
            if isinstance(filter_clauses, list):
                for clause in filter_clauses:
                    if not self._entry_matches_query({'_source': source}, clause):
                        return False
            else:
                return self._entry_matches_query({'_source': source}, filter_clauses)
            return True
        
        # Default to True if no specific bool clauses
        return True
    
    def _matches_terms_query(self, source: Dict, terms_query: Dict) -> bool:
        """Check if source matches terms query (field contains any of the specified values)"""
        for field, values in terms_query.items():
            source_value = source.get(field)
            if source_value and source_value in values:
                return True
        return False
    
    def _matches_term_query(self, source: Dict, term_query: Dict) -> bool:
        """Check if source matches term query (exact match)"""
        for field, value in term_query.items():
            source_value = source.get(field)
            if source_value == value:
                return True
        return False
    
    def _matches_match_query(self, source: Dict, match_query: Dict) -> bool:
        """Check if source matches match query (text matching)"""
        for field, value in match_query.items():
            source_value = str(source.get(field, '')).lower()
            search_value = str(value).lower()
            if search_value in source_value:
                return True
        return False
    
    def _matches_range_query(self, source: Dict, range_query: Dict) -> bool:
        """Check if source matches range query (numeric/date ranges)"""
        for field, range_conditions in range_query.items():
            source_value = source.get(field)
            if source_value is None:
                continue
                
            # Convert to comparable format
            try:
                if isinstance(source_value, (int, float)):
                    numeric_value = source_value
                else:
                    numeric_value = float(str(source_value))
                
                # Check range conditions
                if 'gte' in range_conditions and numeric_value < range_conditions['gte']:
                    continue
                if 'lte' in range_conditions and numeric_value > range_conditions['lte']:
                    continue
                if 'gt' in range_conditions and numeric_value <= range_conditions['gt']:
                    continue
                if 'lt' in range_conditions and numeric_value >= range_conditions['lt']:
                    continue
                
                return True
                
            except (ValueError, TypeError):
                # Handle date strings or other formats
                continue
                
        return False
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for semantic analysis with GPU acceleration"""
        if not self.model or not self.tokenizer:
            return np.array([])
        
        embeddings = []
        
        # Process in optimal batches
        batch_size = min(self.optimal_batch_size, len(texts))
        
        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(batch_texts, return_tensors='pt', truncation=True, 
                                      padding=True, max_length=512)
                
                # Move to device
                inputs = move_to_device(inputs)
                
                with amp_autocast():
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        embeddings.extend(batch_embeddings)
            
            # Clear cache periodically
            if len(embeddings) > 100:
                clear_cache()
                
        except Exception as e:
            self.logger.warning(f"Error generating embeddings: {e}")
            # Fallback to single text processing
            for text in texts:
                try:
                    inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                    inputs = move_to_device(inputs)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                        embeddings.append(embedding)
                except Exception as inner_e:
                    self.logger.warning(f"Error generating embedding for text: {inner_e}")
                    embeddings.append(np.zeros(384))  # Default embedding size
        
        return np.array(embeddings)
    
    def analyze_semantic_similarity(self, text: str, reference_texts: List[str], threshold: float = 0.3) -> List[Dict]:
        """Analyze semantic similarity between text and reference texts"""
        if not reference_texts:
            return []
        
        try:
            text_embedding = self.get_embeddings([text])
            ref_embeddings = self.get_embeddings(reference_texts)
            
            if text_embedding.size == 0 or ref_embeddings.size == 0:
                return []
            
            similarities = cosine_similarity(text_embedding, ref_embeddings)[0]
            
            results = []
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    results.append({
                        'text': reference_texts[i],
                        'similarity': float(similarity),
                        'index': i
                    })
            
            return sorted(results, key=lambda x: x['similarity'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error in semantic analysis: {e}")
            return []
    
    def extract_context_from_es_results(self, es_results: Dict) -> Dict[str, Any]:
        """Extract useful context from ES query results"""
        context = {
            'total_hits': 0,
            'events_found': [],
            'unique_ips': set(),
            'unique_users': set(),
            'unique_hosts': set(),
            'time_range': {'earliest': None, 'latest': None},
            'event_patterns': {},
            'severity_distribution': {}
        }
        
        try:
            hits = es_results.get('hits', {}).get('hits', [])
            context['total_hits'] = len(hits)
            
            for hit in hits:
                source = hit.get('_source', {})
                
                # Extract event information
                event_id = source.get('eventID') or source.get('eventId')
                if event_id:
                    context['events_found'].append(event_id)
                    context['event_patterns'][event_id] = context['event_patterns'].get(event_id, 0) + 1
                
                # Extract network information
                if source.get('srcIP'):
                    context['unique_ips'].add(source['srcIP'])
                if source.get('destIP'):
                    context['unique_ips'].add(source['destIP'])
                
                # Extract user information
                if source.get('targetUsername'):
                    context['unique_users'].add(source['targetUsername'])
                if source.get('accountName'):
                    context['unique_users'].add(source['accountName'])
                
                # Extract host information
                if source.get('hostName'):
                    context['unique_hosts'].add(source['hostName'])
                
                # Extract time information
                timestamp = source.get('timestamp') or source.get('@timestamp')
                if timestamp:
                    if context['time_range']['earliest'] is None or timestamp < context['time_range']['earliest']:
                        context['time_range']['earliest'] = timestamp
                    if context['time_range']['latest'] is None or timestamp > context['time_range']['latest']:
                        context['time_range']['latest'] = timestamp
                
                # Extract severity information
                severity = source.get('severity')
                if severity:
                    context['severity_distribution'][severity] = context['severity_distribution'].get(severity, 0) + 1
            
            # Convert sets to lists for JSON serialization
            context['unique_ips'] = list(context['unique_ips'])
            context['unique_users'] = list(context['unique_users'])
            context['unique_hosts'] = list(context['unique_hosts'])
            
        except Exception as e:
            self.logger.error(f"Error extracting context from ES results: {e}")
        
        return context
    
    def build_enhanced_query(self, base_query: Dict, additional_filters: Dict = None, 
                           time_range: str = None, limit: int = 100) -> Dict:
        """Build enhanced ES query with additional filters"""
        enhanced_query = json.deepcopy(base_query)
        
        # Ensure query structure exists
        if 'query' not in enhanced_query:
            enhanced_query['query'] = {'bool': {'filter': []}}
        elif 'bool' not in enhanced_query['query']:
            enhanced_query['query'] = {'bool': {'filter': [enhanced_query['query']]}}
        elif 'filter' not in enhanced_query['query']['bool']:
            enhanced_query['query']['bool']['filter'] = []
        
        # Add additional filters
        if additional_filters:
            for field, value in additional_filters.items():
                if isinstance(value, list):
                    enhanced_query['query']['bool']['filter'].append({
                        'terms': {field: value}
                    })
                else:
                    enhanced_query['query']['bool']['filter'].append({
                        'term': {field: value}
                    })
        
        # Add time range filter
        if time_range:
            time_filter = self._build_time_filter(time_range)
            if time_filter:
                enhanced_query['query']['bool']['filter'].append(time_filter)
        
        # Set limit
        enhanced_query['size'] = limit
        
        # Add sorting by timestamp
        if 'sort' not in enhanced_query:
            enhanced_query['sort'] = [{'timestamp': {'order': 'desc'}}]
        
        return enhanced_query
    
    def _build_time_filter(self, time_range: str) -> Dict:
        """Build time filter for ES query"""
        try:
            if time_range.startswith('now-'):
                return {
                    'range': {
                        'timestamp': {
                            'gte': time_range
                        }
                    }
                }
            elif time_range == 'today':
                return {
                    'range': {
                        'timestamp': {
                            'gte': 'now/d',
                            'lt': 'now+1d/d'
                        }
                    }
                }
            elif time_range == 'yesterday':
                return {
                    'range': {
                        'timestamp': {
                            'gte': 'now-1d/d',
                            'lt': 'now/d'
                        }
                    }
                }
        except Exception as e:
            self.logger.error(f"Error building time filter: {e}")
        
        return {}
    
    def analyze_with_es_context(self, alert: Dict, data_entry: Dict, prompt: str, 
                              es_context: Dict) -> Dict[str, Any]:
        """Base method for analysis with ES context - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement analyze_with_es_context")
    
    def extract_risk_indicators(self, es_context: Dict, context_data: Dict) -> List[Dict]:
        """Extract risk indicators from ES context and other data"""
        risk_indicators = []
        
        try:
            # Check for high-frequency events
            event_patterns = context_data.get('event_patterns', {})
            for event_id, count in event_patterns.items():
                if count > 10:  # Threshold for suspicious frequency
                    risk_indicators.append({
                        'type': 'high_frequency_event',
                        'event_id': event_id,
                        'count': count,
                        'risk_level': 'medium' if count < 50 else 'high'
                    })
            
            # Check for multiple unique IPs
            unique_ips = context_data.get('unique_ips', [])
            if len(unique_ips) > 5:
                risk_indicators.append({
                    'type': 'multiple_source_ips',
                    'count': len(unique_ips),
                    'ips': unique_ips[:10],  # Limit for display
                    'risk_level': 'medium' if len(unique_ips) < 20 else 'high'
                })
            
            # Check for failed authentication patterns
            relevant_events = es_context.get('relevant_events', [])
            failed_auth_events = [e for e in relevant_events if '4625' in str(e.get('event_id', ''))]
            if len(failed_auth_events) > 0:
                risk_indicators.append({
                    'type': 'failed_authentication_pattern',
                    'count': len(failed_auth_events),
                    'events': failed_auth_events,
                    'risk_level': 'medium' if len(failed_auth_events) < 5 else 'high'
                })
            
            # Check severity distribution
            severity_dist = context_data.get('severity_distribution', {})
            high_severity_count = severity_dist.get('high', 0) + severity_dist.get('critical', 0)
            if high_severity_count > 0:
                risk_indicators.append({
                    'type': 'high_severity_events',
                    'count': high_severity_count,
                    'distribution': severity_dist,
                    'risk_level': 'high' if high_severity_count > 10 else 'medium'
                })
            
        except Exception as e:
            self.logger.error(f"Error extracting risk indicators: {e}")
        
        return risk_indicators
    
    def calculate_risk_score(self, risk_indicators: List[Dict], base_score: int = 0) -> int:
        """Calculate overall risk score based on indicators"""
        risk_score = base_score
        
        for indicator in risk_indicators:
            risk_level = indicator.get('risk_level', 'low')
            if risk_level == 'critical':
                risk_score += 30
            elif risk_level == 'high':
                risk_score += 20
            elif risk_level == 'medium':
                risk_score += 10
            elif risk_level == 'low':
                risk_score += 5
        
        return min(risk_score, 100)  # Cap at 100
    
    def format_analysis_report(self, agent_name: str, analysis_data: Dict, 
                             context_data: Dict, risk_indicators: List[Dict]) -> str:
        """Format a standardized analysis report"""
        risk_score = self.calculate_risk_score(risk_indicators)
        risk_level = self._get_risk_level(risk_score)
        
        report = f"""
{agent_name.upper()} ANALYSIS REPORT
{'=' * (len(agent_name) + 17)}
Risk Level: {risk_level}
Risk Score: {risk_score}/100
Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ES CONTEXT SUMMARY:
• Total Events Analyzed: {context_data.get('total_hits', 0)}
• Unique IPs Found: {len(context_data.get('unique_ips', []))}
• Unique Users Found: {len(context_data.get('unique_users', []))}
• Unique Hosts Found: {len(context_data.get('unique_hosts', []))}
• Event Types: {len(context_data.get('event_patterns', {}))}

RISK INDICATORS:
"""
        
        if risk_indicators:
            for indicator in risk_indicators:
                report += f"• [{indicator['risk_level'].upper()}] {indicator['type']}: "
                if 'count' in indicator:
                    report += f"Count={indicator['count']}"
                report += "\n"
        else:
            report += "• No significant risk indicators detected\n"
        
        report += f"\nKEY FINDINGS:\n"
        findings = analysis_data.get('findings', [])
        if findings:
            for finding in findings[:5]:  # Limit to top 5
                report += f"• {finding}\n"
        else:
            report += "• No significant findings\n"
        
        return report
    
    def _get_risk_level(self, risk_score: int) -> str:
        """Convert risk score to risk level"""
        if risk_score >= 80:
            return 'CRITICAL'
        elif risk_score >= 60:
            return 'HIGH'
        elif risk_score >= 40:
            return 'MEDIUM'
        elif risk_score >= 20:
            return 'LOW'
        else:
            return 'NORMAL'


# Example implementation for IP Agent with ES integration
class EnhancedIPAgent(BaseAgentWithES):
    """Enhanced IP Agent with full ES integration"""
    
    def __init__(self, es_query_agent_url: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__(es_query_agent_url, model_name)
        self.agent_name = "IP Intelligence Agent"
    
    def analyze_ip(self, alert: Dict, data_entry: Dict, prompt: str) -> Dict[str, Any]:
        """Standard IP analysis method"""
        # Basic analysis without ES context
        ip_context = self._extract_ip_context(alert, data_entry)
        
        analysis = {
            'findings': ['Basic IP analysis completed'],
            'risk_score': 20,
            'network_patterns': [],
            'security_indicators': [],
            'anomalies': []
        }
        
        return {
            'agent_type': 'ip_agent',
            'ip_context': ip_context,
            'analysis': analysis,
            'risk_level': 'LOW',
            'report': f"Basic IP analysis for {ip_context.get('primary_ip', 'unknown')}",
            'queries_executed': 1
        }
    
    def analyze_ip_with_context(self, alert: Dict, data_entry: Dict, prompt: str, 
                               es_context: Dict) -> Dict[str, Any]:
        """Enhanced IP analysis with ES context"""
        try:
            self.logger.info("Starting enhanced IP analysis with ES context")
            
            # Store ES context for query execution
            self.current_es_context = es_context
            
            # Extract IP context
            ip_context = self._extract_ip_context(alert, data_entry)
            primary_ip = ip_context.get('primary_ip')
            
            if not primary_ip:
                return self.analyze_ip(alert, data_entry, prompt)  # Fallback
            
            # Execute IP-specific ES queries
            ip_queries = self._build_ip_queries(primary_ip, es_context)
            es_results = {}
            queries_executed = 0
            
            for query_name, query in ip_queries.items():
                result = self.execute_es_query(query)
                es_results[query_name] = result
                queries_executed += 1
            
            # Extract context from ES results
            context_data = {}
            for query_name, result in es_results.items():
                context_data[query_name] = self.extract_context_from_es_results(result)
            
            # Perform enhanced analysis
            analysis = self._analyze_ip_behavior(ip_context, context_data, es_context)
            
            # Extract risk indicators
            risk_indicators = self.extract_risk_indicators(es_context, context_data.get('ip_traffic', {}))
            
            # Add IP-specific risk indicators
            ip_risk_indicators = self._extract_ip_risk_indicators(context_data, ip_context)
            risk_indicators.extend(ip_risk_indicators)
            
            # Calculate risk score
            risk_score = self.calculate_risk_score(risk_indicators, base_score=10)
            risk_level = self._get_risk_level(risk_score)
            
            # Generate report
            report = self.format_analysis_report(
                self.agent_name, 
                analysis, 
                context_data.get('ip_traffic', {}), 
                risk_indicators
            )
            
            return {
                'agent_type': 'ip_agent',
                'ip_context': ip_context,
                'es_results': es_results,
                'analysis': analysis,
                'risk_indicators': risk_indicators,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'report': report,
                'queries_executed': queries_executed,
                'es_context': {
                    'queries_available': len(es_context.get('es_queries', {})),
                    'relevant_events_count': len(es_context.get('relevant_events', [])),
                    'data_entries_available': len(es_context.get('all_data_entries', []))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in enhanced IP analysis: {e}")
            return self.analyze_ip(alert, data_entry, prompt)  # Fallback to basic analysis
    
    def _extract_ip_context(self, alert: Dict, data_entry: Dict) -> Dict[str, Any]:
        """Extract IP-related context from alert and data"""
        source = data_entry.get('_source', {})
        
        return {
            'primary_ip': source.get('srcIP') or source.get('sourceIP'),
            'destination_ip': source.get('destIP') or source.get('destinationIP'),
            'host': source.get('hostName'),
            'connection_type': source.get('logonType'),
            'user': source.get('targetUsername'),
            'domain': source.get('targetDomain'),
            'alert_info': {
                'alert_id': alert.get('alertId'),
                'alert_name': alert.get('alertName'),
                'severity': alert.get('severity'),
                'timestamp': alert.get('createdAt')
            }
        }
    
    def _build_ip_queries(self, primary_ip: str, es_context: Dict) -> Dict[str, Dict]:
        """Build IP-specific ES queries"""
        base_queries = es_context.get('es_queries', {})
        
        queries = {}
        
        # Query 1: All traffic from this IP
        if 'contextual_query' in base_queries:
            ip_traffic_query = json.deepcopy(base_queries['contextual_query'])
            ip_traffic_query = self.build_enhanced_query(
                ip_traffic_query,
                additional_filters={'srcIP': primary_ip},
                time_range='now-24h',
                limit=200
            )
            queries['ip_traffic'] = ip_traffic_query
        
        # Query 2: Failed authentication attempts from this IP
        auth_failure_query = {
            'query': {
                'bool': {
                    'filter': [
                        {'term': {'srcIP': primary_ip}},
                        {'terms': {'eventID': ['4625', '4771', '4776']}}  # Common auth failure events
                    ]
                }
            },
            'size': 100,
            'sort': [{'timestamp': {'order': 'desc'}}]
        }
        queries['auth_failures'] = auth_failure_query
        
        # Query 3: Network connections patterns
        network_query = {
            'query': {
                'bool': {
                    'should': [
                        {'term': {'srcIP': primary_ip}},
                        {'term': {'destIP': primary_ip}}
                    ]
                }
            },
            'size': 150,
            'sort': [{'timestamp': {'order': 'desc'}}]
        }
        queries['network_connections'] = network_query
        
        return queries
    
    def _analyze_ip_behavior(self, ip_context: Dict, context_data: Dict, es_context: Dict) -> Dict[str, Any]:
        """Analyze IP behavior based on ES results"""
        analysis = {
            'findings': [],
            'risk_score': 0,
            'network_patterns': [],
            'security_indicators': [],
            'anomalies': []
        }
        
        primary_ip = ip_context.get('primary_ip')
        
        # Analyze traffic patterns - Updated thresholds for realistic data volumes
        ip_traffic_data = context_data.get('ip_traffic', {})
        total_events = ip_traffic_data.get('total_hits', 0)
        
        if total_events > 10:
            analysis['findings'].append(f"High activity volume: {total_events} events from {primary_ip}")
            analysis['anomalies'].append('high_traffic_volume')
            analysis['risk_score'] += 15
        elif total_events > 3:
            analysis['findings'].append(f"Moderate activity: {total_events} events from {primary_ip}")
            analysis['risk_score'] += 8
        elif total_events > 0:
            analysis['findings'].append(f"Activity detected: {total_events} events from {primary_ip}")
            analysis['risk_score'] += 3
        else:
            analysis['findings'].append(f"No direct traffic found for {primary_ip}")
        
        # Analyze authentication failures - Lowered thresholds for real-world detection
        auth_data = context_data.get('auth_failures', {})
        auth_failures = auth_data.get('total_hits', 0)
        
        if auth_failures > 3:
            analysis['findings'].append(f"Multiple authentication failures: {auth_failures} attempts")
            analysis['security_indicators'].append('brute_force_pattern')
            analysis['risk_score'] += 25
        elif auth_failures > 0:
            analysis['findings'].append(f"Authentication failures detected: {auth_failures} attempts")
            analysis['security_indicators'].append('auth_anomaly')
            analysis['risk_score'] += 12
        
        # Analyze network patterns - More realistic thresholds for lateral movement detection
        network_data = context_data.get('network_connections', {})
        unique_hosts = len(network_data.get('unique_hosts', []))
        
        if unique_hosts > 3:
            analysis['findings'].append(f"Connections to multiple hosts: {unique_hosts}")
            analysis['network_patterns'].append('lateral_movement_potential')
            analysis['risk_score'] += 20
        elif unique_hosts > 1:
            analysis['findings'].append(f"Multi-host connections detected: {unique_hosts}")
            analysis['network_patterns'].append('network_activity')
            analysis['risk_score'] += 8
        
        # Enhanced analysis of IP-specific patterns
        unique_ips = len(ip_traffic_data.get('unique_ips', []))
        if unique_ips > 2:
            analysis['findings'].append(f"IP communications with {unique_ips} different addresses")
            analysis['network_patterns'].append('multi_ip_communication')
            analysis['risk_score'] += 5
        
        # Check for event type patterns
        event_patterns = ip_traffic_data.get('event_patterns', {})
        if event_patterns:
            analysis['findings'].append(f"Event types detected: {list(event_patterns.keys())}")
            if '4625' in event_patterns:  # Failed logon
                analysis['security_indicators'].append('failed_logon_events')
                analysis['risk_score'] += 10
            if '4624' in event_patterns:  # Successful logon
                analysis['findings'].append(f"Successful logon events: {event_patterns['4624']}")
        
        # Check against relevant events from semantic analysis
        relevant_events = es_context.get('relevant_events', [])
        if len(relevant_events) > 3:
            analysis['findings'].append(f"Multiple relevant security events detected: {len(relevant_events)}")
            analysis['risk_score'] += 10
        elif len(relevant_events) > 0:
            analysis['findings'].append(f"Relevant security context found: {len(relevant_events)} events")
            analysis['risk_score'] += 5
        
        return analysis
    
    def _extract_ip_risk_indicators(self, context_data: Dict, ip_context: Dict) -> List[Dict]:
        """Extract IP-specific risk indicators"""
        indicators = []
        
        # Check for authentication brute force patterns - Realistic thresholds
        auth_data = context_data.get('auth_failures', {})
        auth_failures = auth_data.get('total_hits', 0)
        
        if auth_failures > 5:
            indicators.append({
                'type': 'authentication_brute_force',
                'count': auth_failures,
                'ip': ip_context.get('primary_ip'),
                'risk_level': 'high'
            })
        elif auth_failures > 1:
            indicators.append({
                'type': 'multiple_auth_failures',
                'count': auth_failures,
                'ip': ip_context.get('primary_ip'),
                'risk_level': 'medium'
            })
        elif auth_failures > 0:
            indicators.append({
                'type': 'auth_failure_detected',
                'count': auth_failures,
                'ip': ip_context.get('primary_ip'),
                'risk_level': 'low'
            })
        
        # Check for lateral movement indicators
        network_data = context_data.get('network_connections', {})
        unique_hosts = len(network_data.get('unique_hosts', []))
        
        if unique_hosts > 15:
            indicators.append({
                'type': 'potential_lateral_movement',
                'host_count': unique_hosts,
                'ip': ip_context.get('primary_ip'),
                'risk_level': 'high'
            })
        elif unique_hosts > 5:
            indicators.append({
                'type': 'multiple_host_connections',
                'host_count': unique_hosts,
                'ip': ip_context.get('primary_ip'),
                'risk_level': 'medium'
            })
        
        return indicators