import pandas as pd
import json
import os
import yaml
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from datetime import datetime
from gpu_utils import to_cuda

class ElasticsearchQueryBuilder:
    def __init__(self, csv_file_path: str = None, event_df: pd.DataFrame = None,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2", config_path: str = None):

        if event_df is not None:
            self.events_df = event_df
        elif csv_file_path:
            self.events_df = self._load_events_file(csv_file_path)
        else:
            raise ValueError("Provide either csv_file_path or event_df")
            
        # Load configuration
        self.config = self._load_config(config_path)
        
        self._detect_columns()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Force GPU placement
        self.model, _ = to_cuda(self.model)
        self._precompute_embeddings()
        
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not config_path:
            config_path = "config/paths.yml"
        
        default_config = {
            "es": {"host": "http://localhost:9200", "index": "spharaka-windows-000001", "timeout": 30},
            "search": {"default_time_range": "now-30d", "size": 500, "sort_field": "@timestamp", "sort_order": "desc", "ip_behavior_size": 1000},
            "paths": {"data": "data.json"}
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f) or {}
                # Merge with defaults
                for key in default_config:
                    if key in loaded_config:
                        default_config[key].update(loaded_config[key])
                    elif key in loaded_config:
                        default_config[key] = loaded_config[key]
                return default_config
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}. Using defaults.")
                return default_config
        else:
            return default_config
    
    def _is_ip_behavior_intent(self, user_prompt: str) -> bool:
        """Detect if the user prompt is asking for PURE IP behavior analysis (not mixed with authentication)"""
        prompt_lower = user_prompt.lower().strip()
        
        # First check if prompt contains authentication/user keywords
        # If it does, prefer contextual analysis over IP-only behavior
        auth_keywords = ['authentication', 'login', 'logon', 'auth', 'user', 'username', 'account', 'credential']
        has_auth_keywords = any(keyword in prompt_lower for keyword in auth_keywords)
        
        if has_auth_keywords:
            # If authentication keywords are present, don't route to IP-only behavior
            # Let the contextual query handle both IP and authentication aspects
            return False
        
        # IP behavior keywords (only trigger if no auth keywords)
        ip_behavior_patterns = [
            r'\bip\s+behavio[ur]',
            r'\bnetwork\s+behavio[ur]', 
            r'investigate\s+ip\b(?!\s+auth)',  # "investigate ip" but not "investigate ip auth"
            r'analyze\s+ip\b(?!\s+auth)',     # "analyze ip" but not "analyze ip auth"
            r'ip\s+activity\b(?!\s+pattern)',  # "ip activity" but be more specific
            r'network\s+activity\b(?!\s+pattern)',
            r'\bip\s+patterns\b(?!\s+auth)',
            r'\bnetwork\s+patterns\b(?!\s+auth)',
            r'connection\s+patterns\b',
            r'traffic\s+analysis\b'
        ]
        
        return any(re.search(pattern, prompt_lower) for pattern in ip_behavior_patterns)
        
    def _detect_columns(self):
        cols = [col.lower().strip() for col in self.events_df.columns]
        
        self.event_id_col = None
        for i, col in enumerate(cols):
            if any(pattern in col for pattern in ['eventid', 'event_id', 'id', 'event']):
                self.event_id_col = self.events_df.columns[i]
                break
        if not self.event_id_col:
            self.event_id_col = self.events_df.columns[0]
            
        self.desc_col = None
        for i, col in enumerate(cols):
            if any(pattern in col for pattern in ['description', 'desc', 'message', 'summary']):
                self.desc_col = self.events_df.columns[i]
                break
        if not self.desc_col:
            self.desc_col = self.events_df.columns[1] if len(self.events_df.columns) > 1 else self.events_df.columns[0]

    def _load_events_file(self, file_path: str) -> pd.DataFrame:
        """Load events file supporting CSV, JSON, JSONL, or dict mapping formats.

        Accepted JSON layouts:
          1. Array of objects: [ {"eventId": 4625, "description": "..."}, ... ]
          2. Newline-delimited JSON (JSONL): one object per line
          3. Object mapping: {"4625": "Failed logon", "4634": "Logoff"}
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Events file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext in ('.csv', '.tsv'):
                sep = '\t' if ext == '.tsv' else ','
                return pd.read_csv(file_path, sep=sep)
            elif ext in ('.json', '.jsonl', '.ndjson'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # JSONL / NDJSON
                    if ext in ('.jsonl', '.ndjson') or '\n' in content and not content.startswith('[') and not content.startswith('{"'):
                        records: List[Dict[str, Any]] = []
                        for line_num, line in enumerate(content.splitlines(), 1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                records.append(obj)
                            except json.JSONDecodeError as e:
                                print(f"Warning: Skipping invalid JSON line {line_num}: {e}")
                        if not records:
                            raise ValueError("No valid JSON objects found in JSONL file")
                        return pd.DataFrame(records)
                    # Standard JSON (array or mapping)
                    data = json.loads(content)
                    if isinstance(data, list):
                        return pd.DataFrame(data)
                    if isinstance(data, dict):
                        # Assume mapping eventId -> description
                        rows = []
                        for k, v in data.items():
                            rows.append({"eventId": k, "description": v})
                        return pd.DataFrame(rows)
                    raise ValueError("Unsupported JSON structure for events file")
            else:
                # Fallback: try CSV then JSON
                try:
                    return pd.read_csv(file_path)
                except Exception:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            return pd.DataFrame(data)
                        if isinstance(data, dict):
                            rows = [{"eventId": k, "description": v} for k, v in data.items()]
                            return pd.DataFrame(rows)
                        raise
        except Exception as e:
            raise Exception(
                f"Failed to load events file '{file_path}': {e}. "
                "If it's JSON ensure it is either an array of objects or a mapping of eventId to description."
            )
            
    def _precompute_embeddings(self):
        descriptions = self.events_df[self.desc_col].fillna('').astype(str).tolist()
        self.event_embeddings = self._get_embeddings(descriptions)
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        device = next(self.model.parameters()).device  # Always get the model's device
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            # Explicitly move all input tensors to the model's device
            for k in inputs:
                if isinstance(inputs[k], torch.Tensor):
                    inputs[k] = inputs[k].to(device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                embeddings.append(embedding)
        return np.array(embeddings)
    
    def load_alert(self, alert_data: Dict) -> Dict[str, Any]:
        """Extract event IDs and context from alert"""
        alert_info = {
            'alert_id': alert_data.get('alertId'),
            'event_ids': [],
            'rule_event_ids': [],
            'context': {}
        }
        
        # Parse events field
        if 'events' in alert_data:
            events_str = alert_data['events']
            if isinstance(events_str, str):
                try:
                    events_dict = eval(events_str) if events_str.startswith('{') else {}
                    alert_info['event_ids'] = list(events_dict.keys())
                except:
                    alert_info['event_ids'] = []
        
        # Parse rule conditions for eventID
        if 'ruleJson' in alert_data:
            try:
                rule_json = json.loads(alert_data['ruleJson'])
                conditions = rule_json.get('conditions', {}).get('ruleset', [])
                for condition in conditions:
                    if condition.get('fieldname') == 'eventID':
                        alert_info['rule_event_ids'].append(int(condition.get('value')))
            except:
                pass
        
        # Extract context
        alert_info['context'] = {
            'severity': alert_data.get('severity'),
            'alert_type': alert_data.get('alertType'),
            'threat_category': alert_data.get('threatCategory'),
            'technique_id': alert_data.get('techniqueId')
        }
        
        return alert_info
    
    def load_data_context(self, data_entry: Dict) -> Dict[str, Any]:
        """Extract context from data entry for enhanced search"""
        source = data_entry.get('_source', {})
        return {
            'host_name': source.get('hostName'),
            'src_ip': source.get('srcIP'),
            'target_username': source.get('targetUsername'),
            'account_name': source.get('accountName'),
            'target_domain': source.get('targetDomain'),
            'process_name': source.get('processName'),
            'event_id': source.get('eventID'),
            'logon_type': source.get('logonType'),
            'event_type': source.get('eventType')
        }
    
    def find_relevant_events(self, user_prompt: str, top_k: int = None, threshold: float = None) -> List[Dict]:
        """Find contextually relevant event IDs based on investigation intent"""
        # Define event ID mappings for different investigation types
        event_mappings = {
            'authentication': [4624, 4625, 4648, 4634, 4647],  # Logon success/failure/special/logoff
            'account_management': [4720, 4722, 4725, 4740, 4767, 4781],  # Account created/enabled/disabled/locked
            'privilege_escalation': [4672, 4673, 4674, 4648, 4656],  # Special privileges, sensitive privileges
            'process_activity': [4688, 4689, 4696, 4697, 1, 5],  # Process creation/termination, services
            'network_activity': [5156, 5158, 5159, 3, 22],  # Network connections, network shares
            'file_access': [4656, 4658, 4663, 4670, 11],  # Object access, file creation/deletion
            'system_events': [6005, 6006, 6008, 6009, 1074, 1076],  # System startup/shutdown/restart
            'security_events': [4625, 4648, 4672, 4673, 4674, 4697],  # Security-related events
            'audit_events': [4719, 4738, 4739, 4904, 1101, 1104, 1105, 1108],  # Audit policy changes
            'ip_behavior': [5156, 5158, 5159, 3, 22, 4624, 4625],  # Network + authentication
            'user_behavior': [4624, 4625, 4648, 4634, 4647, 4720, 4722, 4725, 4740]  # User authentication & account
        }
        
        prompt_lower = user_prompt.lower()
        selected_events = []
        
        # Determine investigation type based on prompt keywords - use additive logic
        # Check for user/authentication keywords
        if any(keyword in prompt_lower for keyword in ['user', 'username', 'account', 'login', 'logon', 'authentication', 'behavior', 'behaviour']):
            selected_events.extend(event_mappings['user_behavior'])
        
        # Check for IP/network keywords
        if any(keyword in prompt_lower for keyword in ['ip', 'address', 'network', 'connection', 'traffic']):
            selected_events.extend(event_mappings['ip_behavior'])
        
        # Check for process keywords
        if any(keyword in prompt_lower for keyword in ['process', 'execution', 'command', 'program']):
            selected_events.extend(event_mappings['process_activity'])
        
        # Check for file access keywords
        if any(keyword in prompt_lower for keyword in ['file', 'access', 'create', 'delete', 'modify']):
            selected_events.extend(event_mappings['file_access'])
        
        # Check for privilege keywords
        if any(keyword in prompt_lower for keyword in ['privilege', 'admin', 'elevation', 'escalation']):
            selected_events.extend(event_mappings['privilege_escalation'])
        
        # Check for system keywords
        if any(keyword in prompt_lower for keyword in ['system', 'startup', 'shutdown', 'reboot']):
            selected_events.extend(event_mappings['system_events'])
        
        # Check for audit keywords
        if any(keyword in prompt_lower for keyword in ['audit', 'policy', 'configuration']):
            selected_events.extend(event_mappings['audit_events'])
        
        # If no specific keywords found, default to authentication and security events
        if not selected_events:
            selected_events.extend(event_mappings['authentication'])
            selected_events.extend(event_mappings['security_events'])
        
        # Remove duplicates and convert to event info format
        unique_event_ids = list(set(selected_events))
        relevant_events = []
        
        for event_id in unique_event_ids[:top_k if top_k else 10]:
            # Try to get description from events_df if available
            description = f"Windows Event ID {event_id}"
            if hasattr(self, 'events_df') and not self.events_df.empty:
                matching_rows = self.events_df[self.events_df[self.event_id_col] == event_id]
                if not matching_rows.empty:
                    description = str(matching_rows.iloc[0][self.desc_col])
            
            event_info = {
                'event_id': int(event_id),
                'description': description,
                'similarity_score': 0.8  # High confidence for contextual matching
            }
            relevant_events.append(event_info)
                
        return relevant_events
    
    def extract_intent(self, user_prompt: str) -> Dict[str, Any]:
        intent = {
            'time_range': None,
            'filters': {},
            'sort_by': None,
            'limit': None,
            'aggregations': []
        }
        
        prompt_lower = user_prompt.lower()
        
        # Time range patterns
        time_patterns = {
            r'(?:last|past)\s+(\d+)\s+hours?': lambda m: f"now-{m.group(1)}h",
            r'(?:last|past)\s+(\d+)\s+days?': lambda m: f"now-{m.group(1)}d",
            r'(?:last|past)\s+(\d+)\s+weeks?': lambda m: f"now-{int(m.group(1))*7}d",
            r'(?:last|past)\s+(\d+)\s+months?': lambda m: f"now-{int(m.group(1))*30}d",
            r'(?:last|past)\s+hour': lambda m: "now-1h",
            r'(?:last|past)\s+day': lambda m: "now-1d",
            r'(?:last|past)\s+week': lambda m: "now-7d",
            r'(?:last|past)\s+month': lambda m: "now-30d",
            r'today': lambda m: "now/d",
            r'yesterday': lambda m: "now-1d/d"
        }
        
        for pattern, time_func in time_patterns.items():
            match = re.search(pattern, prompt_lower)
            if match:
                intent['time_range'] = time_func(match)
                break
        
        # Extract filters
        field_patterns = {
            r'severity\s*[:=]\s*(\w+)': 'severity',
            r'host\s*[:=]\s*(\w+)': 'hostName',
            r'user\s*[:=]\s*(\w+)': 'targetUsername',
            r'ip\s*[:=]\s*([\d\.]+)': 'srcIP'
        }
        
        for pattern, field in field_patterns.items():
            match = re.search(pattern, prompt_lower)
            if match:
                intent['filters'][field] = match.group(1)
        
        # Extract limits
        limit_patterns = [
            r'(?:top|first|limit|show)\s+(\d+)',
            r'(\d+)\s+(?:results|records|events|items)'
        ]
        
        for pattern in limit_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                intent['limit'] = int(match.group(1))
                break
        
        # Extract aggregations
        if any(word in prompt_lower for word in ['count', 'total', 'number of']):
            intent['aggregations'].append('count')
        if any(word in prompt_lower for word in ['group by', 'breakdown']):
            intent['aggregations'].append('terms')
            
        return intent
    
    def extract_ips_from_alert_events(self, alert: Dict, data_file_path: str = None) -> Dict[str, Any]:
        """Extract IP addresses from the alert's mapped events via data.json"""
        result = {
            "ips": [],
            "event_ids": [],
            "no_ip_extracted": False,
            "reason": None
        }
        
        try:
            # Get event IDs from alert
            event_ids = []
            if 'events' in alert:
                events_str = alert['events']
                if isinstance(events_str, str):
                    try:
                        events_dict = eval(events_str) if events_str.startswith('{') else {}
                        event_ids = list(events_dict.keys())
                    except:
                        pass
            
            if not event_ids:
                result["no_ip_extracted"] = True
                result["reason"] = "No event IDs found in alert mapping"
                return result
            
            result["event_ids"] = event_ids
            
            # Load data.json if path provided, otherwise try default
            if not data_file_path:
                data_file_path = "data.json"  # Default fallback
                
            if not os.path.exists(data_file_path):
                result["no_ip_extracted"] = True
                result["reason"] = f"Data file not found: {data_file_path}"
                return result
                
            # Read data.json line by line (it's JSONL format)
            found_ips = set()
            with open(data_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data_entry = json.loads(line)
                        if "_source" in data_entry:
                            source = data_entry["_source"]
                            # Check if this entry matches any of our event IDs
                            if "eventId" in source and source["eventId"] in event_ids:
                                # Extract IP addresses
                                ip_fields = ["srcIP", "src_ip", "sourceIP", "source_ip", "clientIP", "client_ip"]
                                for ip_field in ip_fields:
                                    if ip_field in source and source[ip_field]:
                                        ip_value = source[ip_field].strip()
                                        if ip_value and ip_value != "0.0.0.0" and ip_value != "127.0.0.1":
                                            found_ips.add(ip_value)
                    except json.JSONDecodeError:
                        continue
            
            result["ips"] = list(found_ips)
            
            if not found_ips:
                result["no_ip_extracted"] = True
                result["reason"] = f"No IP addresses found for event IDs: {event_ids}"
            
            return result
            
        except Exception as e:
            result["no_ip_extracted"] = True
            result["reason"] = f"Error extracting IPs: {str(e)}"
            return result
    
    def build_ip_focus_query(self, ips: List[str], cfg: Dict = None) -> Dict[str, Any]:
        """Build IP-focused ES query that searches the full index for IP behavior patterns"""
        if not ips:
            return {"error": "No IPs provided for query"}
            
        # Set defaults from config or use fallbacks
        default_config = {
            "es": {"host": "http://10.9.56.10:30200", "index": "spharaka-windows", "timeout": 30},
            "search": {"default_time_range": "now-30d", "size": 1000, "sort_field": "@timestamp", "sort_order": "desc"}
        }
        
        if cfg:
            config = cfg
        else:
            config = default_config
            
        # Build the IP-focused query
        query = {
            "size": config.get("search", {}).get("ip_behavior_size", config.get("search", {}).get("size", 1000)),
            "query": {
                "bool": {
                    "should": [
                        # Match IPs in various IP fields
                        {"terms": {"srcIP": ips}},
                        {"terms": {"src_ip": ips}},
                        {"terms": {"sourceIP": ips}},
                        {"terms": {"source_ip": ips}},
                        {"terms": {"destIP": ips}},
                        {"terms": {"dest_ip": ips}},
                        {"terms": {"destinationIP": ips}},
                        {"terms": {"destination_ip": ips}},
                        {"terms": {"clientIP": ips}},
                        {"terms": {"client_ip": ips}},
                        {"terms": {"serverIP": ips}},
                        {"terms": {"server_ip": ips}}
                    ],
                    "minimum_should_match": 1,
                    "filter": [
                        # Add time range filter
                        {
                            "range": {
                                config.get("search", {}).get("sort_field", "@timestamp"): {
                                    "gte": config.get("search", {}).get("default_time_range", "now-30d")
                                }
                            }
                        }
                    ]
                }
            },
            "sort": [
                {
                    config.get("search", {}).get("sort_field", "@timestamp"): {
                        "order": config.get("search", {}).get("sort_order", "desc")
                    }
                }
            ],
            "aggregations": {
                "ip_activity_summary": {
                    "multi_terms": {
                        "terms": [
                            {"field": "srcIP.keyword"},
                            {"field": "eventType.keyword"},
                            {"field": "severity.keyword"}
                        ],
                        "size": 100
                    }
                },
                "timeline_activity": {
                    "date_histogram": {
                        "field": config.get("search", {}).get("sort_field", "@timestamp"),
                        "interval": "1h"
                    }
                }
            }
        }
        
        # Add metadata for this query type
        query_metadata = {
            "query_type": "ip_behavior",
            "target_ips": ips,
            "es_host": config.get("es", {}).get("host", "http://10.9.56.10:30200"),
            "es_index": config.get("es", {}).get("index", "spharaka-windows"),
            "description": f"IP behavior analysis query for: {', '.join(ips)}"
        }
        
        return {
            "elasticsearch_query": query,
            "metadata": query_metadata
        }

    def build_exact_match_query(self, alert_info: Dict) -> Dict[str, Any]:
        """Build query for exact alert event match"""
        query = {
            "query": {
                "bool": {
                    "filter": []
                }
            }
        }
        
        # Add exact event ID filters
        if alert_info['event_ids']:
            query["query"]["bool"]["filter"].append({
                "terms": {"eventId": alert_info['event_ids']}
            })
        
        return query
    
    def analyze_investigation_intent(self, user_prompt: str, data_context: Dict) -> Dict[str, Any]:
        """Use semantic understanding to determine what context fields to include"""
        investigation_context = {}
        
        # Get available context fields and their values
        available_context = {}
        if data_context.get('src_ip'):
            available_context['srcIP'] = data_context['src_ip']
        if data_context.get('target_username'):
            available_context['targetUsername'] = data_context['target_username']
        if data_context.get('host_name'):
            available_context['hostName'] = data_context['host_name']
        if data_context.get('target_domain'):
            available_context['targetDomain'] = data_context['target_domain']
        if data_context.get('process_name'):
            available_context['processName'] = data_context['process_name']
        
        if not available_context:
            return investigation_context
        
        prompt_lower = user_prompt.lower()
        
        # For authentication investigations, be more selective to avoid over-filtering
        if any(keyword in prompt_lower for keyword in ['authentication', 'login', 'logon', 'user', 'account']):
            # For authentication investigations, prioritize user and IP, but don't require all context
            if data_context.get('target_username'):
                investigation_context['targetUsername'] = data_context['target_username']
            # Only add IP if explicitly mentioned or if it's an IP behavior investigation
            if any(keyword in prompt_lower for keyword in ['ip', 'address', 'network']) and data_context.get('src_ip'):
                investigation_context['srcIP'] = data_context['src_ip']
        else:
            # For other investigations, use semantic similarity as before
            field_descriptions = {}
            for field, value in available_context.items():
                if field == 'srcIP':
                    field_descriptions[field] = f"network traffic connections from IP address {value}"
                elif field == 'targetUsername':
                    field_descriptions[field] = f"user account activities for username {value}"
                elif field == 'hostName':
                    field_descriptions[field] = f"system activities on host computer {value}"
                elif field == 'targetDomain':
                    field_descriptions[field] = f"domain activities in {value}"
                elif field == 'processName':
                    field_descriptions[field] = f"process activities for {value}"
            
            # Get embeddings
            prompt_embedding = self._get_embeddings([user_prompt])
            descriptions = list(field_descriptions.values())
            field_embeddings = self._get_embeddings(descriptions)
            
            # Calculate similarities
            similarities = cosine_similarity(prompt_embedding, field_embeddings)[0]
            
            # Include fields above similarity threshold
            threshold = 0.3
            field_names = list(field_descriptions.keys())
            
            for i, similarity in enumerate(similarities):
                if similarity > threshold:
                    field_name = field_names[i]
                    investigation_context[field_name] = available_context[field_name]
        
        return investigation_context
    
    def build_contextual_query(self, alert_info: Dict, data_context: Dict, user_prompt: str) -> Dict[str, Any]:
        """Build enhanced query based on alert context and user prompt - optimized for broader analysis"""
        relevant_events = self.find_relevant_events(user_prompt)
        intent = self.extract_intent(user_prompt)
        
        prompt_lower = user_prompt.lower()
        
        # Default size - larger for authentication investigations
        query_size = intent.get('limit') or 1000  # Use 1000 if limit is None
        if any(keyword in prompt_lower for keyword in ['user', 'authentication', 'login', 'logon', 'behavior', 'behaviour']):
            query_size = max(query_size, 1000)  # Ensure we get enough data for pattern analysis
        
        query = {
            "size": query_size,
            "query": {
                "bool": {
                    "filter": []
                }
            }
        }
        
        # Add sorting - default to timestamp desc for pattern analysis
        if intent['sort_by']:
            if intent['sort_by'] == 'timestamp_asc':
                query["sort"] = [{"@timestamp": {"order": "asc"}}]
            elif intent['sort_by'] == 'timestamp':
                query["sort"] = [{"@timestamp": {"order": "desc"}}]
            else:
                query["sort"] = [{intent['sort_by']: {"order": "desc"}}]
        else:
            query["sort"] = [{"@timestamp": {"order": "desc"}}]
        
        # Add relevant event IDs from semantic analysis - ALWAYS required
        if relevant_events:
            event_ids = []
            for event in relevant_events:
                event_id = event['event_id']
                # Convert numpy types to Python types
                if hasattr(event_id, 'item'):
                    event_id = event_id.item()
                elif isinstance(event_id, (np.int64, np.int32)):
                    event_id = int(event_id)
                event_ids.append(event_id)
                
            if alert_info['rule_event_ids']:
                event_ids.extend(alert_info['rule_event_ids'])
            
            # Use both eventID and eventId field names to handle variations
            query["query"]["bool"]["filter"].append({
                "bool": {
                    "should": [
                        {"terms": {"eventID": list(set(event_ids))}},
                        {"terms": {"eventId": list(set(event_ids))}}
                    ],
                    "minimum_should_match": 1
                }
            })
        
        # Add time range filter
        if intent['time_range']:
            time_filter = self._build_dynamic_time_filter(intent['time_range'])
            if time_filter:
                query["query"]["bool"]["filter"].append(time_filter)
        
        # Smart context filtering - focus on primary investigation target
        primary_filter = None
        
        # For authentication/user behavior investigations - focus on username
        if any(keyword in prompt_lower for keyword in ['user', 'username', 'authentication', 'login', 'logon', 'behavior', 'behaviour']):
            if data_context.get('target_username'):
                # Use .keyword field for exact match
                primary_filter = {
                    "bool": {
                        "should": [
                            {"term": {"targetUsername.keyword": data_context['target_username']}},
                            {"term": {"targetUsername": data_context['target_username']}},
                            {"term": {"username.keyword": data_context['target_username']}},
                            {"term": {"username": data_context['target_username']}}
                        ],
                        "minimum_should_match": 1
                    }
                }
        
        # For IP behavior investigations - focus on IP
        elif any(keyword in prompt_lower for keyword in ['ip', 'address', 'network', 'connection', 'traffic']):
            if data_context.get('src_ip'):
                primary_filter = {
                    "bool": {
                        "should": [
                            {"term": {"srcIP": data_context['src_ip']}},
                            {"term": {"src_ip": data_context['src_ip']}},
                            {"term": {"sourceIP": data_context['src_ip']}},
                            {"term": {"clientIP": data_context['src_ip']}}
                        ],
                        "minimum_should_match": 1
                    }
                }
        
        # For host/system investigations - focus on host
        elif any(keyword in prompt_lower for keyword in ['host', 'system', 'computer', 'machine']):
            if data_context.get('host_name'):
                primary_filter = {
                    "bool": {
                        "should": [
                            {"term": {"hostName.keyword": data_context['host_name']}},
                            {"term": {"hostName": data_context['host_name']}},
                            {"term": {"host.keyword": data_context['host_name']}},
                            {"term": {"computerName": data_context['host_name']}}
                        ],
                        "minimum_should_match": 1
                    }
                }
        
        # Add the primary filter if we found one
        if primary_filter:
            query["query"]["bool"]["filter"].append(primary_filter)
        
        # Add any explicit filters from the user prompt
        for field, value in intent.get('filters', {}).items():
            query["query"]["bool"]["filter"].append({
                "term": {field: value}
            })
        
        # Add useful aggregations for pattern analysis
        query["aggs"] = {
            "events_over_time": {
                "date_histogram": {
                    "field": "@timestamp",
                    "interval": "1h"
                }
            },
            "by_event_type": {
                "terms": {
                    "field": "eventID",
                    "size": 20
                }
            }
        }
        
        # Add investigation-specific aggregations
        if any(keyword in prompt_lower for keyword in ['user', 'username', 'authentication', 'login']):
            query["aggs"]["by_host"] = {
                "terms": {
                    "field": "hostName.keyword",
                    "size": 50
                }
            }
            query["aggs"]["by_source_ip"] = {
                "terms": {
                    "field": "srcIP.keyword", 
                    "size": 50
                }
            }
            query["aggs"]["by_logon_type"] = {
                "terms": {
                    "field": "logonType",
                    "size": 10
                }
            }
        
        # Add custom aggregations from intent
        if intent.get('aggregations'):
            custom_aggs = self._build_dynamic_aggregations(intent['aggregations'])
            query["aggs"].update(custom_aggs)
            
        return query
    
    def build_combined_query(self, exact_query: Dict, contextual_query: Dict) -> Dict[str, Any]:
        """Combine exact and contextual queries using should clause"""
        return {
            "query": {
                "bool": {
                    "should": [
                        exact_query["query"],
                        contextual_query["query"]
                    ],
                    "minimum_should_match": 1
                }
            },
            "sort": contextual_query.get("sort", [{"timestamp": {"order": "desc"}}]),
            "size": contextual_query.get("size", 100)
        }
    
    def _build_dynamic_time_filter(self, time_range: str) -> Optional[Dict]:
        if time_range.startswith('now/d'):
            return {"range": {"timestamp": {"gte": "now/d", "lt": "now+1d/d"}}}
        elif time_range.startswith('now-1d/d'):
            return {"range": {"timestamp": {"gte": "now-1d/d", "lt": "now/d"}}}
        else:
            return {"range": {"timestamp": {"gte": time_range}}}
    
    def _build_dynamic_aggregations(self, agg_types: List[str]) -> Dict:
        aggs = {}
        
        if 'count' in agg_types:
            aggs['total_count'] = {"value_count": {"field": "eventID"}}
            
        if 'terms' in agg_types:
            aggs['by_event_type'] = {"terms": {"field": "eventID", "size": 50}}
            aggs['by_severity'] = {"terms": {"field": "severity", "size": 10}}
            
        return aggs
    
    def load_alerts_file(self, alerts_file_path: str) -> List[Dict]:
        """Load alerts from JSON or JSONL file"""
        try:
            alerts = []
            with open(alerts_file_path, 'r', encoding='utf-8') as f:
                if alerts_file_path.endswith('.jsonl'):
                    # Handle JSONL format (one JSON object per line)
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                alert = json.loads(line)
                                alerts.append(alert)
                            except json.JSONDecodeError as e:
                                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                                continue
                    return alerts
                else:
                    # Handle regular JSON format
                    content = f.read().strip()
                    if content.startswith('['):
                        # JSON array
                        alerts = json.loads(content)
                    else:
                        # Single JSON object or multiple objects separated by newlines
                        for line in content.split('\n'):
                            line = line.strip()
                            if line:
                                try:
                                    alert = json.loads(line)
                                    alerts.append(alert)
                                except json.JSONDecodeError:
                                    # Try parsing as single object
                                    alerts = json.loads(content)
                                    break
                    return alerts if isinstance(alerts, list) else [alerts]
        except Exception as e:
            raise Exception(f"Error loading alerts file: {str(e)}")
    
    def load_data_file(self, data_file_path: str) -> List[Dict]:
        """Load data from JSON or JSONL file with robust error handling"""
        try:
            data_entries = []
            with open(data_file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                print(f"Data file size: {len(content)} characters")
                
                # Split into lines and process each
                lines = content.split('\n')
                print(f"Found {len(lines)} lines in data file")
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        data_entry = json.loads(line)
                        data_entries.append(data_entry)
                        if line_num <= 2:  # Show first 2 entries for debugging
                            print(f"Loaded data entry {line_num}: {list(data_entry.keys()) if isinstance(data_entry, dict) else type(data_entry)}")
                    except json.JSONDecodeError as e:
                        print(f"Skipping line {line_num} (JSON error): {str(e)[:100]}...")
                        # Show problematic line (first 100 chars)
                        print(f"Line content: {line[:100]}...")
                        continue
                
                print(f"Successfully loaded {len(data_entries)} data entries")
                return data_entries
                
        except Exception as e:
            print(f"File reading error: {str(e)}")
            # Try alternative approach - maybe it's a single JSON object
            try:
                with open(data_file_path, 'r', encoding='utf-8') as f:
                    single_object = json.load(f)
                    print("Loaded as single JSON object")
                    return [single_object] if not isinstance(single_object, list) else single_object
            except:
                raise Exception(f"Could not parse data file in any format: {str(e)}")
    
    def get_alert_by_number(self, alerts: List[Dict], alert_number: int) -> Dict:
        """Get alert by number selection"""
        try:
            alert_num = int(alert_number)
            if 1 <= alert_num <= len(alerts):
                selected_alert = alerts[alert_num - 1]
                # Add id if not present
                if 'id' not in selected_alert:
                    selected_alert['id'] = alert_num
                return selected_alert
            else:
                raise ValueError(f"Alert number must be between 1 and {len(alerts)}")
        except ValueError as e:
            raise Exception(f"Invalid alert number: {str(e)}")
    
    def find_matching_data(self, data_list: List[Dict], alert_info: Dict) -> Dict:
        """Find data entry that matches alert event IDs"""
        for data_entry in data_list:
            source = data_entry.get('_source', {})
            event_id = source.get('eventId')
            
            # Check if this data entry matches any alert event ID
            if event_id in alert_info['event_ids']:
                return data_entry
                
        # If no exact match, return first entry as fallback
        return data_list[0] if data_list else {}
    
    def display_alerts(self, alerts: List[Dict]) -> None:
        """Display available alerts for user selection"""
        print("\nAvailable Alerts:")
        print("-" * 50)
        for i, alert in enumerate(alerts, 1):
            alert_id = alert.get('alertId', 'Unknown')[:20] + "..."
            severity = alert.get('severity', 'Unknown')
            alert_name = alert.get('alertName', 'No name')
            print(f"{i}. {alert_name} | Severity: {severity} | ID: {alert_id}")
        print("-" * 50)
    
    def process_file_based_query(self, alerts_file: str, data_file: str, 
                                alert_number: str, user_prompt: str) -> Dict[str, Any]:
        """Main function: Process query using file-based alert and data"""
        
        # Load alerts and data files
        alerts = self.load_alerts_file(alerts_file)
        data_list = self.load_data_file(data_file)
        
        # Get selected alert
        selected_alert = self.get_alert_by_number(alerts, int(alert_number))
        
        # Load alert information
        alert_info = self.load_alert(selected_alert)
        
        # Find matching data entry
        matching_data = self.find_matching_data(data_list, alert_info)
        
        # Load data context
        data_context = self.load_data_context(matching_data)
        
        # Build queries
        exact_query = self.build_exact_match_query(alert_info)
        contextual_query = self.build_contextual_query(alert_info, data_context, user_prompt)
        combined_query = self.build_combined_query(exact_query, contextual_query)
        
        return {
            'selected_alert': selected_alert,
            'matching_data': matching_data,
            'alert_info': alert_info,
            'data_context': data_context,
            'user_prompt': user_prompt,
            'exact_match_query': exact_query,
            'contextual_query': contextual_query,
            'combined_query': combined_query
        }
        """Main function: Process alert + data + prompt to generate ES queries"""
        
        # Load alert information
        alert_info = self.load_alert(alert_data)
        
        # Load data context
        data_context = self.load_data_context(data_entry)
        
        # Build exact match query for alert event
        exact_query = self.build_exact_match_query(alert_info)
        
        # Build contextual query based on prompt and context
        contextual_query = self.build_contextual_query(alert_info, data_context, user_prompt)
        
        # Build combined query
        combined_query = self.build_combined_query(exact_query, contextual_query)
        
        return {
            'alert_id': alert_info['alert_id'],
            'user_prompt': user_prompt,
            'exact_match_query': exact_query,
            'contextual_query': contextual_query,
            'combined_query': combined_query,
            'alert_context': alert_info,
            'data_context': data_context
        }

    def process_alert_query(self, alert_data: Dict, data_entry: Dict, user_prompt: str) -> Dict[str, Any]:
        """Process alert + data + prompt to generate ES queries with IP behavior routing"""
        
        # Check if this is an IP behavior intent
        if self._is_ip_behavior_intent(user_prompt):
            return self._process_ip_behavior_query(alert_data, user_prompt)
        
        # Otherwise, use existing contextual query processing
        return self._process_contextual_query(alert_data, data_entry, user_prompt)
    
    def _process_ip_behavior_query(self, alert_data: Dict, user_prompt: str) -> Dict[str, Any]:
        """Process IP behavior specific queries"""
        
        # Extract IPs from the alert's mapped events
        data_file_path = self.config.get("paths", {}).get("data", "data.json")
        ip_extraction = self.extract_ips_from_alert_events(alert_data, data_file_path)
        
        if ip_extraction["no_ip_extracted"]:
            return {
                'alert_id': alert_data.get('alertId'),
                'user_prompt': user_prompt,
                'query_type': 'ip_behavior',
                'error': 'No IPs could be extracted for IP behavior analysis',
                'reason': ip_extraction["reason"],
                'extracted_event_ids': ip_extraction.get("event_ids", [])
            }
        
        # Build IP-focused query
        ip_query_result = self.build_ip_focus_query(ip_extraction["ips"], self.config)
        
        # Prepare response
        result = {
            'alert_id': alert_data.get('alertId'),
            'user_prompt': user_prompt,
            'query_type': 'ip_behavior',
            'extracted_ips': ip_extraction["ips"],
            'extracted_event_ids': ip_extraction["event_ids"],
            'ip_focused_query': ip_query_result.get("elasticsearch_query", {}),
            'query_metadata': ip_query_result.get("metadata", {}),
            # For backward compatibility, also provide as main query
            'elasticsearch_query': ip_query_result.get("elasticsearch_query", {}),
            'subject': ', '.join(ip_extraction["ips"]),  # Subject for agent routing
            'description': f"IP behavior analysis for: {', '.join(ip_extraction['ips'])}"
        }
        
        return result
    
    def _process_contextual_query(self, alert_data: Dict, data_entry: Dict, user_prompt: str) -> Dict[str, Any]:
        """Process non-IP behavior queries using existing logic"""
        
        # Load alert information
        alert_info = self.load_alert(alert_data)
        
        # Load data context
        data_context = self.load_data_context(data_entry)
        
        # Build exact match query for alert event
        exact_query = self.build_exact_match_query(alert_info)
        
        # Build contextual query based on prompt and context
        contextual_query = self.build_contextual_query(alert_info, data_context, user_prompt)
        
        # Build combined query
        combined_query = self.build_combined_query(exact_query, contextual_query)
        
        return {
            'alert_id': alert_info['alert_id'],
            'user_prompt': user_prompt,
            'query_type': 'contextual',
            'exact_match_query': exact_query,
            'contextual_query': contextual_query,
            'combined_query': combined_query,
            'elasticsearch_query': combined_query,  # Main query for execution
            'alert_context': alert_info,
            'data_context': data_context
        }

# Usage
if __name__ == "__main__":
    print("=== Alert-Based Elasticsearch Query Builder ===")
    
    try:
        # Get file paths from user
        print("\nEnter Event file path (CSV or JSON with EventIDs and descriptions):")
        event_file = input().strip().strip('"').strip("'")
        
        print("\nEnter alerts JSON file path:")
        alerts_file = input().strip().strip('"').strip("'")
        
        print("\nEnter data JSON file path:")
        data_file = input().strip().strip('"').strip("'")
        
        if not all([event_file, alerts_file, data_file]):
            print("All file paths are required.")
            exit(1)
        
        # Initialize builder with uploaded event file (CSV or JSON)
        print(f"\nLoading Event file: {event_file}")
        builder = ElasticsearchQueryBuilder(event_file)
        
        # Load and display alerts
        print(f"Loading alerts: {alerts_file}")
        alerts = builder.load_alerts_file(alerts_file)
        builder.display_alerts(alerts)
        
        # Get user selections
        print("\nEnter alert number:")
        alert_number = input().strip()
        
        print("\nEnter your analysis query:")
        user_prompt = input().strip()
        
        if not alert_number or not user_prompt:
            print("Both alert number and query are required.")
            exit(1)
            
        # Process the query
        result = builder.process_file_based_query(alerts_file, data_file, alert_number, user_prompt)
        
        print(f"\nSelected Alert: {result['selected_alert'].get('alertName', 'Unknown')}")
        print(f"User Query: {user_prompt}")
        print(f"Matched Event ID: {result['alert_info']['event_ids']}")
        print(f"Data Context: Host={result['data_context']['host_name']}, IP={result['data_context']['src_ip']}")
        
        print("\nGenerated Elasticsearch Queries:")
        print("\n1. Exact Match Query:")
        print(json.dumps(result['exact_match_query'], indent=2))
        
        print("\n2. Contextual Query:")
        print(json.dumps(result['contextual_query'], indent=2))
        
        print("\n3. Combined Query:")
        print(json.dumps(result['combined_query'], indent=2))
        
    except FileNotFoundError as e:
        print(f"File not found: {str(e)}")
        print("Please check file paths and ensure files exist")
    except Exception as e:
        print(f"Error: {str(e)}")