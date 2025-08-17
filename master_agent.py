import json
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gpu_utils import get_device, move_to_device, to_cuda, amp_autocast, get_optimal_batch_size

class MasterAgent:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Force GPU placement
        self.model, _ = to_cuda(self.model)
        self.optimal_batch_size = get_optimal_batch_size(base_batch_size=16)
        
    def load_alerts(self, alerts_file_path: str) -> List[Dict]:
        """Load alerts from file"""
        with open(alerts_file_path, 'r', encoding='utf-8') as f:
            if alerts_file_path.endswith('.jsonl'):
                alerts = []
                for line in f:
                    line = line.strip()
                    if line:
                        alerts.append(json.loads(line))
                return alerts
            else:
                alerts = json.load(f)
                return alerts if isinstance(alerts, list) else [alerts]
    
    def get_alert_by_number(self, alerts: List[Dict], alert_number: int) -> Dict:
        """Get specific alert by number"""
        return alerts[alert_number - 1]
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for text analysis with GPU acceleration"""
        embeddings = []
        
        # Process in optimal batches
        batch_size = min(self.optimal_batch_size, len(texts))
        
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
        
        return np.array(embeddings)
    
    def analyze_intent(self, user_prompt: str, current_alert: Dict = None) -> Dict[str, Any]:
        """Prompt-anchored subject selection with global defaults policy, with fallback for non-matching prompts."""
        import re
        import json

        # Step 1: Start with always-included agents (policy)
        included_agents = ["summary", "alert_history", "timeline"]
        excluded_agents = []
        reasons = {
            "summary": "included: policy default",
            "alert_history": "included: policy default",
            "timeline": "included: policy default"
        }

        # Normalize prompt for analysis
        prompt_lower = user_prompt.lower().strip()

        # Handle typos and variations
        prompt_lower = re.sub(r'\bib\s+behavio[ur]', 'ip behavior', prompt_lower)
        prompt_lower = re.sub(r'\bip\s+behavio[ur]', 'ip behavior', prompt_lower)

        # Step 2: Handle opt-out directives first
        if 'summary only' in prompt_lower:
            included_agents = ["summary"]
            excluded_agents = ["alert_history", "timeline", "user_context", "ip_agent"]
            reasons = {
                "summary": "included: explicit 'summary only' directive",
                "alert_history": "excluded: 'summary only' directive",
                "timeline": "excluded: 'summary only' directive",
                "user_context": "excluded: 'summary only' directive",
                "ip_agent": "excluded: 'summary only' directive"
            }
            primary_subjects = set()
        elif 'timeline only' in prompt_lower:
            included_agents = ["timeline", "summary"]
            excluded_agents = ["alert_history", "user_context", "ip_agent"]
            reasons = {
                "timeline": "included: explicit 'timeline only' directive",
                "summary": "included: policy default",
                "alert_history": "excluded: 'timeline only' directive",
                "user_context": "excluded: 'timeline only' directive",
                "ip_agent": "excluded: 'timeline only' directive"
            }
            primary_subjects = set()
        else:
            # Apply other opt-out directives
            if 'exclude history' in prompt_lower:
                included_agents.remove('alert_history')
                excluded_agents.append('alert_history')
                reasons["alert_history"] = "excluded: explicit 'exclude history' directive"

            if 'exclude timeline' in prompt_lower:
                included_agents.remove('timeline')
                excluded_agents.append('timeline')
                reasons["timeline"] = "excluded: explicit 'exclude timeline' directive"

            # Step 3: Primary subject inference from prompt
            primary_subjects = self._infer_primary_subjects(prompt_lower)

            # Step 4: Dynamic agent inclusion based on prompt subjects
            # Check for explicit exclusions
            exclude_ip = 'exclude ip' in prompt_lower or 'exclude network' in prompt_lower
            exclude_user = 'exclude user' in prompt_lower or 'exclude account' in prompt_lower

            # Include ip_agent if IP is a primary subject and not excluded
            if 'ip' in primary_subjects and not exclude_ip:
                included_agents.append('ip_agent')
                reasons['ip_agent'] = "included: IP subject detected in prompt"
            else:
                excluded_agents.append('ip_agent')
                if exclude_ip:
                    reasons['ip_agent'] = "excluded: explicit IP exclusion directive"
                else:
                    reasons['ip_agent'] = "excluded: not relevant to prompt"

            # Include user_context if user is a primary subject and not excluded
            if 'user' in primary_subjects and not exclude_user:
                included_agents.append('user_context')
                reasons['user_context'] = "included: user subject detected in prompt"
            else:
                excluded_agents.append('user_context')
                if exclude_user:
                    reasons['user_context'] = "excluded: explicit user exclusion directive"
                else:
                    reasons['user_context'] = "excluded: not relevant to prompt"

            # If neither ip nor user is included, ensure fallback reasons for both
            if 'ip_agent' not in included_agents and 'ip_agent' not in reasons:
                excluded_agents.append('ip_agent')
                reasons['ip_agent'] = "excluded: not relevant to prompt"
            if 'user_context' not in included_agents and 'user_context' not in reasons:
                excluded_agents.append('user_context')
                reasons['user_context'] = "excluded: not relevant to prompt"

        # Build routing decision
        agent_prompts = self._generate_subject_prompts(included_agents, current_alert or {}, user_prompt, primary_subjects if 'primary_subjects' in locals() else set())

        # Inject fallback summary prompt if only default agents are included and no specific agent matched
        default_set = set(["summary", "alert_history", "timeline"])
        if set(included_agents).issubset(default_set) and len(included_agents) == len(default_set):
            agent_prompts["summary"] = "This prompt doesn't match specific agent types (like user or IP). But here’s a general summary of the alert and context."

        routing_decision = {
            "included_agents": included_agents,
            "excluded_agents": excluded_agents,
            "reasons": reasons,
            "agent_prompts": agent_prompts
        }

        # Legacy compatibility
        return {
            'required_agents': routing_decision["included_agents"],
            'analysis_type': self._determine_analysis_type(routing_decision["included_agents"]),
            'routing_decision': routing_decision,
            'primary_subjects': primary_subjects if 'primary_subjects' in locals() else set()
        }
    
    def _infer_primary_subjects(self, prompt_lower: str) -> set:
        """Infer primary subjects from user prompt only (prompt > alert)"""
        subjects = set()
        
        # IP/Network signals
        ip_keywords = ['ip', 'network', 'connection', 'traffic', 'firewall', 'source', 'destination', 'behavior', 'behaviour']
        has_ip_signals = any(keyword in prompt_lower for keyword in ip_keywords)
        
        # User/Account signals  
        user_keywords = ['user', 'account', 'login', 'credential', 'privilege', 'authentication', 'identity']
        has_user_signals = any(keyword in prompt_lower for keyword in user_keywords)
        
        # Correlation signals (indicates both subjects)
        correlation_keywords = ['correlate', 'lateral movement', 'between', 'from', 'to', 'and']
        has_correlation = any(keyword in prompt_lower for keyword in correlation_keywords)
        
        # Subject determination logic
        if has_correlation and has_ip_signals and has_user_signals:
            # Explicit correlation between both
            subjects.update(['ip', 'user'])
        elif has_ip_signals and not has_user_signals:
            # IP-focused, no user mentions
            subjects.add('ip')
        elif has_user_signals and not has_ip_signals:
            # User-focused, no IP mentions  
            subjects.add('user')
        elif has_ip_signals and has_user_signals:
            # Both mentioned, assume correlation
            subjects.update(['ip', 'user'])
        # If neither, subjects remains empty (no dynamic agents added)
        
        return subjects
    
    def _generate_subject_prompts(self, included_agents: List[str], current_alert: Dict, 
                                 user_prompt: str, primary_subjects: set) -> Dict[str, str]:
        """Generate targeted prompts based on primary subjects"""
        prompts = {}
        alert_context = f"Alert: {current_alert.get('alertName', 'Unknown')} (Severity: {current_alert.get('severity', 'Unknown')})"
        
        for agent in included_agents:
            if agent == 'user_context':
                # Focus on user-specific analysis
                prompts[agent] = f"{alert_context}. Analyze user behavior, authentication patterns, and account activities. Task: {user_prompt}"
                
            elif agent == 'ip_agent':
                # Focus on IP/network-specific analysis
                prompts[agent] = f"{alert_context}. Investigate network behavior, IP connections, and traffic patterns. Task: {user_prompt}"
                
            elif agent == 'timeline':
                prompts[agent] = f"{alert_context}. Sort and analyze events chronologically from calling agents. Task: {user_prompt}"
                
            elif agent == 'alert_history':
                prompts[agent] = f"{alert_context}. Find historical patterns and similar incidents. Task: {user_prompt}"
                
            elif agent == 'summary':
                prompts[agent] = f"{alert_context}. Compile findings from all agents and provide threat assessment. Task: {user_prompt}"
        
        return prompts
    
    def _extract_evidence_signals(self, prompt_lower: str, current_alert: Dict) -> Dict[str, Any]:
        """Extract concrete evidence signals from prompt and alert"""
        import re
        
        signals = {
            'user_focused': False,
            'ip_focused': False, 
            'timeline_requested': False,
            'history_requested': False,
            'user_entities': [],
            'ip_entities': [],
            'explicit_keywords': []
        }
        
        # User signals
        user_keywords = ['user', 'account', 'login', 'credential', 'privilege', 'authentication', 'identity']
        signals['user_focused'] = any(keyword in prompt_lower for keyword in user_keywords)
        
        # Extract user entities from prompt and alert
        username_patterns = [r'\b[a-zA-Z][a-zA-Z0-9._-]*@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',  # emails
                           r'\b[a-zA-Z][a-zA-Z0-9._-]{2,}\b']  # usernames
        for pattern in username_patterns:
            signals['user_entities'].extend(re.findall(pattern, prompt_lower))
        
        # Check alert for user fields
        user_fields = ['targetUsername', 'subjectUserName', 'userName', 'user', 'accountName']
        for field in user_fields:
            if current_alert.get(field) and current_alert[field] != 'unknown_user':
                signals['user_entities'].append(current_alert[field])
                signals['user_focused'] = True
        
        # IP/Network signals
        ip_keywords = ['ip', 'network', 'connection', 'traffic', 'firewall', 'source', 'destination', 'behavior', 'behaviour']
        signals['ip_focused'] = any(keyword in prompt_lower for keyword in ip_keywords)
        
        # Extract IP literals
        ip_patterns = [r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',  # IPv4
                      r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b']  # IPv6 (simplified)
        for pattern in ip_patterns:
            signals['ip_entities'].extend(re.findall(pattern, prompt_lower))
        
        # Check alert for IP fields  
        ip_fields = ['srcIP', 'sourceIP', 'clientIP', 'remoteIP', 'destinationIP', 'destIP']
        for field in ip_fields:
            if current_alert.get(field):
                signals['ip_entities'].append(current_alert[field])
                signals['ip_focused'] = True
        
        # Timeline signals
        timeline_keywords = ['timeline', 'chronological', 'sequence', 'ordered', 'order', 'before', 'after']
        signals['timeline_requested'] = any(keyword in prompt_lower for keyword in timeline_keywords)
        
        # History signals
        history_keywords = ['history', 'historical', 'trend', 'recurrence', 'pattern', 'past', 'previous']
        signals['history_requested'] = any(keyword in prompt_lower for keyword in history_keywords)
        
        return signals
    
    def _generate_evidence_prompts(self, included_agents: List[str], current_alert: Dict, 
                                 user_prompt: str, signals: Dict) -> Dict[str, str]:
        """Generate targeted prompts based on evidence"""
        prompts = {}
        alert_context = f"Alert: {current_alert.get('alertName', 'Unknown')} (Severity: {current_alert.get('severity', 'Unknown')})"
        
        for agent in included_agents:
            if agent == 'user_context':
                entities = f" Focus on entities: {', '.join(signals['user_entities'])}" if signals['user_entities'] else ""
                prompts[agent] = f"{alert_context}. Analyze user behavior, authentication patterns, and account activities.{entities} Task: {user_prompt}"
                
            elif agent == 'ip_agent': 
                entities = f" Focus on IPs: {', '.join(signals['ip_entities'])}" if signals['ip_entities'] else ""
                prompts[agent] = f"{alert_context}. Investigate network behavior, IP connections, and traffic patterns.{entities} Task: {user_prompt}"
                
            elif agent == 'timeline':
                prompts[agent] = f"{alert_context}. Sort and analyze events chronologically from calling agents. Task: {user_prompt}"
                
            elif agent == 'alert_history':
                prompts[agent] = f"{alert_context}. Find historical patterns and similar incidents. Task: {user_prompt}"
                
            elif agent == 'summary':
                prompts[agent] = f"{alert_context}. Compile findings from all agents and provide threat assessment. Task: {user_prompt}"
        
        return prompts
    
    def _determine_analysis_type(self, included_agents: List[str]) -> str:
        """Determine analysis type based on included agents"""
        agent_count = len([a for a in included_agents if a != 'summary'])
        
        if agent_count >= 4:
            return "comprehensive"
        elif agent_count >= 2:
            return "focused"
        else:
            return "minimal"
    
    def generate_agent_prompts(self, alert: Dict, user_prompt: str, required_agents: List[str]) -> Dict[str, str]:
        """Generate specific prompts for each required agent"""
        prompts = {}
        
        # Extract alert context
        alert_context = {
            'alert_id': alert.get('alertId'),
            'alert_name': alert.get('alertName'),
            'severity': alert.get('severity'),
            'description': alert.get('description')
        }
        
        # Parse events and rules for context
        events_str = alert.get('events', '{}')
        rule_json = alert.get('ruleJson', '{}')
        
        base_context = f"Alert: {alert_context['alert_name']} (Severity: {alert_context['severity']})"
        
        if 'user_context' in required_agents:
            prompts['user_context'] = f"{base_context}. Analyze user behavior patterns, authentication activities, account changes, and identify any anomalies or suspicious user activities. Focus on: {user_prompt}"
        
        if 'ip_agent' in required_agents:
            prompts['ip_agent'] = f"{base_context}. Investigate network behavior, IP connections, traffic patterns, geographic anomalies, and network-based threats. Focus on: {user_prompt}"
        
        if 'timeline' in required_agents:
            prompts['timeline'] = f"{base_context}. Create chronological timeline of events, sequence analysis, and temporal correlations. Organize events in time order for: {user_prompt}"
        
        if 'alert_history' in required_agents:
            prompts['alert_history'] = f"{base_context}. Find historical patterns, similar previous alerts, recurring incidents, and historical threat correlations. Search history for: {user_prompt}"
        
        if 'summary' in required_agents:
            prompts['summary'] = f"{base_context}. Compile all agent findings, provide comprehensive threat assessment, determine if this is a real threat, and generate actionable recommendations. Overall analysis for: {user_prompt}"
        
        return prompts
    
    def orchestrate_investigation(self, alerts_file: str, alert_number: int, user_prompt: str) -> Dict[str, Any]:
        """Main orchestration function with evidence-first routing"""
        
        # Load alerts
        alerts = self.load_alerts(alerts_file)
        selected_alert = self.get_alert_by_number(alerts, alert_number)
        
        # Evidence-first intent analysis
        intent_analysis = self.analyze_intent(user_prompt, selected_alert)
        
        # Generate agent prompts using the routing decision
        agent_prompts = intent_analysis['routing_decision']['agent_prompts']
        
        return {
            'selected_alert': selected_alert,
            'user_prompt': user_prompt,
            'intent_analysis': intent_analysis,
            'agent_prompts': agent_prompts,
            'execution_plan': {
                'step_1': 'Deploy selected agents based on evidence: ' + ', '.join([agent for agent in intent_analysis['required_agents'] if agent != 'summary']),
                'step_2': 'Collect agent results',
                'step_3': 'Execute timeline and alert history analysis if selected',
                'step_4': 'Generate summary report'
            }
        }

# Usage
if __name__ == "__main__":
    print("=== Master Agent ===")
    
    # Get inputs
    alerts_file = input("Enter alerts file path: ").strip().strip('"').strip("'")
    alert_number = int(input("Enter alert number: ").strip())
    user_prompt = input("Enter investigation prompt: ").strip()
    
    # Initialize and run
    master = MasterAgent()
    result = master.orchestrate_investigation(alerts_file, alert_number, user_prompt)
    
    # Display orchestration plan
    print(f"\nAlert: {result['selected_alert']['alertName']}")
    print(f"User Query: {result['user_prompt']}")
    print(f"Analysis Type: {result['intent_analysis']['analysis_type']}")
    print(f"Required Agents: {result['intent_analysis']['required_agents']}")
    
    print("\nExecution Plan:")
    for step, action in result['execution_plan'].items():
        print(f"{step}: {action}")
    
    print("\nAgent Prompts:")
    for agent, prompt in result['agent_prompts'].items():
        print(f"\n{agent.upper()}:")
        print(f"'{prompt}'")