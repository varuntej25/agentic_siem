# integrated_main_orchestrator.py - Properly Integrated Security Analysis System

# Suppress TensorFlow and Protobuf warnings
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')

# --- UTF-8 safe logging helpers ---
import sys
import io
import json
import logging

def _supports_utf8_stream(stream) -> bool:
    try:
        enc = getattr(stream, "encoding", None)
        return bool(enc) and enc.lower().replace("-", "") == "utf8"
    except Exception:
        return False

def configure_logging_utf8_safe():
    root = logging.getLogger()
    root.handlers.clear()

    # Stream handler to console with UTF-8 if possible; else replace
    stream = sys.stdout
    if not _supports_utf8_stream(stream):
        # Wrap stdout as UTF-8 with replacement so emojis won't crash
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sh = logging.StreamHandler(stream)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    # File handler always UTF-8
    log_file = os.getenv("APP_LOG_FILE", "security_analysis.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    root.addHandler(sh)
    root.addHandler(fh)
    root.setLevel(logging.INFO)

def emoji(s: str) -> str:
    """Emit emoji only if stdout truly supports UTF-8; else empty string."""
    return s if _supports_utf8_stream(sys.stdout) else ""

# Call this once at startup (before any logging)
configure_logging_utf8_safe()

from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime

def _supports_utf8_stream(stream) -> bool:
    """Check if a stream supports UTF-8 encoding"""
    try:
        enc = getattr(stream, "encoding", None)
        return bool(enc) and enc.lower().replace("-", "") == "utf8"
    except Exception:
        return False

def configure_logging_utf8_safe():
    """Configure UTF-8 safe logging for Windows console compatibility"""
    root = logging.getLogger()
    root.handlers.clear()

    # Stream handler to console with UTF-8 if possible; else replace
    stream = sys.stdout
    if not _supports_utf8_stream(stream):
        # Wrap stdout as UTF-8 with replacement so emojis won't crash
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sh = logging.StreamHandler(stream)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    # File handler always UTF-8
    log_file = os.getenv("APP_LOG_FILE", "security_analysis.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    root.addHandler(sh)
    root.addHandler(fh)
    root.setLevel(logging.INFO)

def emoji(s: str) -> str:
    """Emit emoji only if stdout truly supports UTF-8; else empty string."""
    return s if _supports_utf8_stream(sys.stdout) else ""

# Call this once at startup (before any logging)
configure_logging_utf8_safe()

# Import the ES Query Builder
from elastic_search import ElasticsearchQueryBuilder

# Import all agents (assuming these exist)
from master_agent import MasterAgent
from user_context import UserContextAgent
from ip_intelligence import IPAgent
from timeline_agent import TimelineAgent
from alert_history import AlertHistoryAgent
from summary_agent import SummaryAgent

@dataclass
class Config:
    """Centralized configuration management with GPU acceleration settings"""
    ES_URL: str = field(default_factory=lambda: os.getenv('ES_URL', 'http://localhost:9200'))
    ES_QUERY_AGENT_URL: str = field(default_factory=lambda: os.getenv('ES_QUERY_AGENT_URL', 'http://localhost:5000'))
    MODEL_NAME: str = field(default_factory=lambda: os.getenv('MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2'))
    
    # Risk thresholds
    RISK_CRITICAL: int = field(default_factory=lambda: int(os.getenv('RISK_CRITICAL', '80')))
    RISK_HIGH: int = field(default_factory=lambda: int(os.getenv('RISK_HIGH', '60')))
    RISK_MEDIUM: int = field(default_factory=lambda: int(os.getenv('RISK_MEDIUM', '40')))
    RISK_LOW: int = field(default_factory=lambda: int(os.getenv('RISK_LOW', '20')))
    
    # Similarity thresholds
    SIMILARITY_THRESHOLD: float = field(default_factory=lambda: float(os.getenv('SIMILARITY_THRESHOLD', '0.3')))
    
    # Timeouts
    REQUEST_TIMEOUT: int = field(default_factory=lambda: int(os.getenv('REQUEST_TIMEOUT', '30')))
    QUERY_TIMEOUT: int = field(default_factory=lambda: int(os.getenv('QUERY_TIMEOUT', '60')))
    
    # GPU Acceleration Settings
    FORCE_CPU: bool = field(default_factory=lambda: os.getenv('FORCE_CPU', '0') == '1')
    CUDA_DEVICE: int = field(default_factory=lambda: int(os.getenv('CUDA_DEVICE', '0')))
    ENABLE_AMP: bool = field(default_factory=lambda: os.getenv('ENABLE_AMP', 'True').lower() in ['true', '1', 'yes'])
    GPU_MEMORY_FRACTION: Optional[float] = field(default_factory=lambda: float(os.getenv('GPU_MEMORY_FRACTION')) if os.getenv('GPU_MEMORY_FRACTION') else None)

class IntegratedSecurityAnalysisOrchestrator:
    """Integrated orchestrator with proper ES Query Builder integration and GPU acceleration"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        # Logging already configured globally, just get logger
        self.logger = logging.getLogger(__name__)
        self._initialize_gpu()
        self.master_agent = MasterAgent(model_name=self.config.MODEL_NAME)
        self.agents = self._initialize_agents()
        self.shared_context = {}
        self.es_query_builder = None  # Will be initialized with event file
        
    def _initialize_gpu(self):
        """Initialize GPU acceleration with detailed logging"""
        self.logger = logging.getLogger(__name__)
        try:
            from gpu_utils import get_accelerator, get_device_info
            self.accelerator = get_accelerator()
            device_info = get_device_info()
            
            # Use UTF-8 safe emoji function
            gpu_emoji = emoji("🚀")
            self.logger.info(f"{gpu_emoji} GPU Acceleration initialized: {device_info}")
        except Exception as e:
            self.logger.warning(f"GPU initialization failed, using CPU: {e}")
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents with config"""
        return {
            'user_context': UserContextAgent(
                es_query_agent_url=self.config.ES_QUERY_AGENT_URL,
                model_name=self.config.MODEL_NAME
            ),
            'ip_agent': IPAgent(
                es_query_agent_url=self.config.ES_QUERY_AGENT_URL,
                model_name=self.config.MODEL_NAME
            ),
            'timeline': TimelineAgent(
                es_query_agent_url=self.config.ES_QUERY_AGENT_URL,
                model_name=self.config.MODEL_NAME
            ),
            'alert_history': AlertHistoryAgent(
                es_query_agent_url=self.config.ES_QUERY_AGENT_URL,
                model_name=self.config.MODEL_NAME
            ),
            'summary': SummaryAgent(
                model_name=self.config.MODEL_NAME
            )
        }
    
    def initialize_es_query_builder(self, events_file: str) -> None:
        """Initialize the Elasticsearch Query Builder with events file"""
        try:
            self.logger.info(f"Initializing ES Query Builder with events file: {events_file}")
            self.es_query_builder = ElasticsearchQueryBuilder(csv_file_path=events_file)
            self.logger.info("ES Query Builder initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ES Query Builder: {e}")
            raise
    
    def load_alert_and_data(self, alerts_file: str, data_file: str, alert_number: int) -> tuple:
        """Load alert and find matching data entry using ES Query Builder approach"""
        try:
            if not self.es_query_builder:
                raise ValueError("ES Query Builder not initialized. Call initialize_es_query_builder first.")
            
            # Use the ES Query Builder to process files and find matching data
            self.logger.info("Processing files using ES Query Builder")
            
            # Load alerts using ES Query Builder method
            alerts = self.es_query_builder.load_alerts_file(alerts_file)
            self.logger.info(f"Loaded {len(alerts)} alerts from {alerts_file}")
            
            # Get selected alert
            selected_alert = self.es_query_builder.get_alert_by_number(alerts, alert_number)
            self.logger.info(f"Selected alert: {selected_alert.get('alertName', 'Unknown')}")
            
            # Load data file using ES Query Builder method
            data_list = self.es_query_builder.load_data_file(data_file)
            self.logger.info(f"Loaded {len(data_list)} data entries from {data_file}")
            
            # Load alert information to get event IDs
            alert_info = self.es_query_builder.load_alert(selected_alert)
            self.logger.info(f"Alert info extracted: event_ids={alert_info['event_ids']}")
            
            # Find matching data entry
            matching_data = self.es_query_builder.find_matching_data(data_list, alert_info)
            self.logger.info("Found matching data entry")
            
            # Store additional context for agents
            self.shared_context.update({
                'all_alerts': alerts,
                'all_data_entries': data_list,
                'alert_info': alert_info,
                'es_queries': {}  # Will store generated queries
            })
            
            return selected_alert, matching_data
            
        except Exception as e:
            self.logger.error(f"Error loading alert and data: {e}")
            raise
    
    def generate_elasticsearch_queries(self, alert: Dict, data_entry: Dict, user_prompt: str) -> Dict[str, Any]:
        """Generate Elasticsearch queries using the ES Query Builder"""
        try:
            if not self.es_query_builder:
                raise ValueError("ES Query Builder not initialized")
            
            self.logger.info("Generating Elasticsearch queries")
            
            # Use the ES Query Builder to generate all types of queries
            query_result = self.es_query_builder.process_alert_query(alert, data_entry, user_prompt)
            
            # Handle IP behavior queries differently
            if query_result.get('query_type') == 'ip_behavior':
                self.logger.info(f"🔍 IP Behavior query detected for IPs: {query_result.get('extracted_ips', [])}")
                
                # Store IP-focused queries in shared context
                self.shared_context['es_queries'] = {
                    'ip_focused_query': query_result.get('ip_focused_query', {}),
                    'query_metadata': query_result.get('query_metadata', {}),
                    'query_type': 'ip_behavior',
                    'subject_ips': query_result.get('extracted_ips', [])
                }
                
                # Check configuration for semantic similarity on IP behavior
                import yaml
                import os
                
                def load_config():
                    """Load configuration from paths.yml"""
                    try:
                        config_path = os.path.join(os.path.dirname(__file__), 'config', 'paths.yml')
                        with open(config_path, 'r') as f:
                            return yaml.safe_load(f)
                    except Exception as e:
                        self.logger.warning(f"Could not load config: {e}")
                        return {}
                
                config_data = load_config()
                enable_semantic = config_data.get('semantic', {}).get('enable_for_ip_behavior', True)
                
                if enable_semantic:
                    # Use IP behavior specific threshold and settings
                    ip_threshold = config_data.get('semantic', {}).get('ip_behavior_threshold', 0.2)
                    max_events = config_data.get('semantic', {}).get('max_relevant_events', 10)
                    
                    try:
                        relevant_events = self.es_query_builder.find_relevant_events(
                            user_prompt, top_k=max_events, threshold=ip_threshold
                        )
                        self.shared_context['relevant_events'] = relevant_events
                        self.logger.info(f"Found {len(relevant_events)} relevant events for IP behavior (threshold: {ip_threshold})")
                    except Exception as e:
                        self.logger.warning(f"Could not find relevant events for IP behavior: {e}")
                        self.shared_context['relevant_events'] = []
                else:
                    # Semantic similarity disabled for IP behavior
                    self.shared_context['relevant_events'] = []
                    self.logger.info("Semantic similarity disabled for IP behavior queries")
                
            else:
                # Store queries in shared context for agents to use (existing logic)
                self.shared_context['es_queries'] = {
                    'exact_match_query': query_result.get('exact_match_query', {}),
                    'contextual_query': query_result.get('contextual_query', {}),
                    'combined_query': query_result.get('combined_query', {}),
                    'alert_context': query_result.get('alert_context', {}),
                    'data_context': query_result.get('data_context', {}),
                    'query_type': 'contextual'
                }
                
                # Also find relevant events for context
                try:
                    # Use configurable threshold and max events
                    import yaml
                    import os
                    
                    def load_config():
                        """Load configuration from paths.yml"""
                        try:
                            config_path = os.path.join(os.path.dirname(__file__), 'config', 'paths.yml')
                            with open(config_path, 'r') as f:
                                return yaml.safe_load(f)
                        except Exception as e:
                            self.logger.warning(f"Could not load config: {e}")
                            return {}
                    
                    config_data = load_config()
                    default_threshold = config_data.get('semantic', {}).get('default_threshold', 0.3)
                    max_events = config_data.get('semantic', {}).get('max_relevant_events', 10)
                    
                    relevant_events = self.es_query_builder.find_relevant_events(
                        user_prompt, top_k=max_events, threshold=default_threshold
                    )
                    self.shared_context['relevant_events'] = relevant_events
                    self.logger.info(f"Found {len(relevant_events)} relevant events (threshold: {default_threshold})")
                except Exception as e:
                    self.logger.warning(f"Could not find relevant events: {e}")
                    self.shared_context['relevant_events'] = []
            
            self.logger.info("Generated queries successfully")
            return query_result
            
        except Exception as e:
            self.logger.error(f"Error generating Elasticsearch queries: {e}")
            raise
    
    def execute_agent_with_es_context(self, agent_name: str, prompt: str, 
                                    alert: Dict, data_entry: Dict) -> Dict[str, Any]:
        """Execute a single agent with ES query context"""
        try:
            self.logger.info(f"Executing agent: {agent_name} with ES context")
            
            agent = self.agents[agent_name]
            
            # Prepare enhanced context with ES queries
            enhanced_context = {
                'es_queries': self.shared_context.get('es_queries', {}),
                'relevant_events': self.shared_context.get('relevant_events', []),
                'alert_info': self.shared_context.get('alert_info', {}),
                'all_data_entries': self.shared_context.get('all_data_entries', [])
            }
            
            # Call the correct method for each agent with enhanced context
            if agent_name == 'user_context':
                result = agent.analyze_user(alert, data_entry, prompt)
                # Add ES query context to result
                if hasattr(agent, 'analyze_user_with_context'):
                    result = agent.analyze_user_with_context(alert, data_entry, prompt, enhanced_context)
                
            elif agent_name == 'ip_agent':
                result = agent.analyze_ip(alert, data_entry, prompt)
                # Add ES query context if method exists
                if hasattr(agent, 'analyze_ip_with_context'):
                    result = agent.analyze_ip_with_context(alert, data_entry, prompt, enhanced_context)
                
            elif agent_name == 'timeline':
                other_agent_data = self.shared_context.get('agent_results', [])
                result = agent.analyze_timeline(alert, data_entry, prompt, other_agent_data)
                # Add ES query context if method exists
                if hasattr(agent, 'analyze_timeline_with_context'):
                    result = agent.analyze_timeline_with_context(alert, data_entry, prompt, other_agent_data, enhanced_context)
                
            elif agent_name == 'alert_history':
                result = agent.analyze_alert_history(alert, data_entry, prompt)
                # Add ES query context if method exists
                if hasattr(agent, 'analyze_alert_history_with_context'):
                    result = agent.analyze_alert_history_with_context(alert, data_entry, prompt, enhanced_context)
                
            elif agent_name == 'summary':
                other_results = self.shared_context.get('agent_results', [])
                if hasattr(agent, 'compile_analysis'):
                    result = agent.compile_analysis(alert, other_results, prompt)
                else:
                    result = {
                        'agent_type': 'summary',
                        'status': 'completed',
                        'summary': f'Analysis completed for {alert.get("alertName", "Unknown alert")}'
                    }
                # Add ES query context if method exists
                if hasattr(agent, 'compile_analysis_with_context'):
                    result = agent.compile_analysis_with_context(alert, other_results, prompt, enhanced_context)
            else:
                raise ValueError(f"Unknown agent: {agent_name}")
            
            # Enhance result with ES query information
            result['es_context'] = {
                'queries_available': len(self.shared_context.get('es_queries', {})),
                'relevant_events_count': len(self.shared_context.get('relevant_events', [])),
                'data_entries_available': len(self.shared_context.get('all_data_entries', []))
            }
            
            self.logger.info(f"Agent {agent_name} completed successfully with ES context")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing agent {agent_name}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'agent_type': agent_name,
                'error': str(e),
                'status': 'failed'
            }
    
    def execute_agents_parallel_with_es(self, agent_prompts: Dict, alert: Dict, 
                                      data_entry: Dict) -> List[Dict[str, Any]]:
        """Execute multiple agents in parallel with ES context"""
        results = []
        
        # Separate agents that can run in parallel vs those that need sequential execution
        parallel_agents = {k: v for k, v in agent_prompts.items() 
                         if k not in ['timeline', 'summary']}
        sequential_agents = {k: v for k, v in agent_prompts.items() 
                           if k in ['timeline', 'summary']}
        
        # Execute parallel agents with ES context
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_agent = {
                executor.submit(self.execute_agent_with_es_context, agent_name, prompt, alert, data_entry): agent_name
                for agent_name, prompt in parallel_agents.items()
            }
            
            parallel_results = []
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result()
                    parallel_results.append(result)
                    self.logger.info(f"Parallel agent {agent_name} completed")
                except Exception as e:
                    self.logger.error(f"Parallel agent {agent_name} failed: {e}")
                    parallel_results.append({
                        'agent_type': agent_name,
                        'error': str(e),
                        'status': 'failed'
                    })
        
        results.extend(parallel_results)
        
        # Store results for sequential agents
        self.shared_context['agent_results'] = parallel_results
        
        # Execute sequential agents with ES context
        for agent_name, prompt in sequential_agents.items():
            result = self.execute_agent_with_es_context(agent_name, prompt, alert, data_entry)
            results.append(result)
        
        return results
    
    def compile_final_report_with_es(self, orchestration_result: Dict, 
                                   agent_results: List[Dict], 
                                   es_query_result: Dict) -> Dict[str, Any]:
        """Compile final comprehensive report with ES query information"""
        successful_results = [r for r in agent_results if r.get('status') != 'failed']
        failed_results = [r for r in agent_results if r.get('status') == 'failed']
        
        # Calculate overall risk score
        risk_scores = []
        for result in successful_results:
            if 'risk_level' in result:
                risk_level = result['risk_level']
                if risk_level == 'CRITICAL':
                    risk_scores.append(90)
                elif risk_level == 'HIGH':
                    risk_scores.append(70)
                elif risk_level == 'MEDIUM':
                    risk_scores.append(50)
                elif risk_level == 'LOW':
                    risk_scores.append(30)
                else:
                    risk_scores.append(10)
            elif 'risk_score' in result:
                risk_scores.append(result['risk_score'])
        
        overall_risk_score = max(risk_scores) if risk_scores else 0
        
        # Determine overall risk level
        if overall_risk_score >= self.config.RISK_CRITICAL:
            overall_risk = 'CRITICAL'
        elif overall_risk_score >= self.config.RISK_HIGH:
            overall_risk = 'HIGH'
        elif overall_risk_score >= self.config.RISK_MEDIUM:
            overall_risk = 'MEDIUM'
        elif overall_risk_score >= self.config.RISK_LOW:
            overall_risk = 'LOW'
        else:
            overall_risk = 'NORMAL'
        
        # Extract ES query information
        es_info = self._extract_es_query_info(es_query_result)
        
        # Create executive summary with ES context
        alert = orchestration_result['selected_alert']
        
        # Add IP behavior info if detected
        ip_behavior_info = ""
        if es_query_result.get('query_type') == 'ip_behavior':
            target_ips = es_query_result.get('extracted_ips', [])
            ip_behavior_info = f"""
🎯 IP BEHAVIOR ANALYSIS ACTIVE:
• Target IP(s): {', '.join(target_ips)}
• Analysis Focus: Network patterns and IP-centric behaviors
• Query Type: Full-index IP behavior search
"""
        
        executive_summary = f"""
INTEGRATED SECURITY ANALYSIS REPORT
===================================
Alert: {alert.get('alertName', 'Unknown')}
Severity: {alert.get('severity', 'Unknown')}
Overall Risk Assessment: {overall_risk} ({overall_risk_score}/100)
Analysis Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{ip_behavior_info}

ELASTICSEARCH INTEGRATION:
• Analysis Type: {es_query_result.get('query_type', 'contextual')}
• Data Entries Analyzed: {len(self.shared_context.get('all_data_entries', []))}
• Relevant Events Found: {len(self.shared_context.get('relevant_events', []))}
• Query Success: {'✓' if es_query_result else '✗'}

EXECUTION SUMMARY:
• Agents Executed: {len(successful_results)} successful, {len(failed_results)} failed
• Total Queries: {sum(r.get('queries_executed', 0) for r in successful_results)}
• ES Context Utilized: {sum(1 for r in successful_results if r.get('es_context'))}

DETAILED ELASTICSEARCH QUERIES:
{es_info}

AGENT RESULTS:
{self._extract_agent_summaries(successful_results)}

KEY FINDINGS:
{self._extract_key_findings(successful_results)}

RECOMMENDATIONS:
{self._generate_recommendations(overall_risk, successful_results)}

ERROR SUMMARY:
{self._extract_error_summary(failed_results)}
"""
        
        return {
            'orchestration': orchestration_result,
            'elasticsearch_context': es_query_result,
            'agent_results': agent_results,
            'overall_risk_score': overall_risk_score,
            'overall_risk_level': overall_risk,
            'executive_summary': executive_summary,
            'successful_agents': len(successful_results),
            'failed_agents': len(failed_results),
            'es_integration_stats': {
                'relevant_events_found': len(self.shared_context.get('relevant_events', [])),
                'queries_generated': len(es_query_result),
                'data_entries_processed': len(self.shared_context.get('all_data_entries', [])),
                'agents_with_es_context': sum(1 for r in successful_results if r.get('es_context'))
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _extract_es_query_info(self, es_query_result: Dict) -> str:
        """Extract ES query information for the report"""
        if not es_query_result:
            return "• No Elasticsearch queries generated"
        
        info_lines = []
        
        # Check if this is an IP behavior query
        if es_query_result.get('query_type') == 'ip_behavior':
            info_lines.append("🎯 IP BEHAVIOR ANALYSIS QUERY:")
            
            # IP extraction info
            extracted_ips = es_query_result.get('extracted_ips', [])
            if extracted_ips:
                info_lines.append(f"• Target IPs: {', '.join(extracted_ips)}")
            
            extracted_events = es_query_result.get('extracted_event_ids', [])
            if extracted_events:
                info_lines.append(f"• Source Event IDs: {', '.join(extracted_events)}")
            
            # Query metadata
            query_meta = es_query_result.get('query_metadata', {})
            if query_meta:
                info_lines.append(f"• ES Host: {query_meta.get('es_host', 'Unknown')}")
                info_lines.append(f"• ES Index: {query_meta.get('es_index', 'Unknown')}")
                info_lines.append(f"• Description: {query_meta.get('description', 'N/A')}")
            
            # Show actual query structure
            es_query = es_query_result.get('ip_focused_query', {})
            if es_query:
                query_size = es_query.get('size', 'N/A')
                info_lines.append(f"• Query Size Limit: {query_size}")
                
                # Show IP fields being searched
                bool_query = es_query.get('query', {}).get('bool', {})
                should_clauses = bool_query.get('should', [])
                ip_fields = []
                for clause in should_clauses:
                    if 'terms' in clause:
                        ip_fields.extend(clause['terms'].keys())
                if ip_fields:
                    info_lines.append(f"• IP Fields Searched: {', '.join(set(ip_fields))}")
                
                # Show aggregations
                aggs = es_query.get('aggregations', {})
                if aggs:
                    info_lines.append(f"• Aggregations: {', '.join(aggs.keys())}")
            
            # Show the formatted query (first 500 chars)
            if 'elasticsearch_query' in es_query_result:
                import json
                query_json = json.dumps(es_query_result['elasticsearch_query'], indent=2)
                if len(query_json) > 500:
                    query_preview = query_json[:500] + "...\n  [Query truncated for display]"
                else:
                    query_preview = query_json
                info_lines.append(f"• Generated Query Preview:\n{query_preview}")
        
        else:
            # Handle traditional contextual queries
            if 'exact_match_query' in es_query_result:
                exact_query = es_query_result['exact_match_query']
                event_ids = exact_query.get('query', {}).get('bool', {}).get('filter', [])
                if event_ids:
                    info_lines.append(f"• Exact Match Query: Targeting specific event IDs")
            
            if 'contextual_query' in es_query_result:
                contextual_query = es_query_result['contextual_query']
                filters = contextual_query.get('query', {}).get('bool', {}).get('filter', [])
                info_lines.append(f"• Contextual Query: {len(filters)} filters applied")
            
            if 'combined_query' in es_query_result:
                info_lines.append("• Combined Query: Merging exact and contextual searches")
        
        # Show relevant events info
        relevant_events = self.shared_context.get('relevant_events', [])
        if relevant_events:
            top_event = relevant_events[0]
            info_lines.append(f"• Top Relevant Event: {top_event.get('event_id')} (similarity: {top_event.get('similarity_score', 0):.2f})")
        
        return '\n'.join(info_lines) if info_lines else "• ES queries generated but details unavailable"
    
    def _extract_agent_summaries(self, results: List[Dict]) -> str:
        """Extract summaries from each agent"""
        summaries = []
        for result in results:
            agent_type = result.get('agent_type', 'unknown')
            risk_level = result.get('risk_level', 'UNKNOWN')
            es_context = "✓" if result.get('es_context') else "✗"
            
            # Add execution time if available
            exec_time = result.get('execution_time', 0)
            time_str = f" ({exec_time:.1f}s)" if exec_time > 0 else ""
            
            # Add queries count if available
            queries = result.get('queries_executed', 0)
            query_str = f" | {queries} queries" if queries > 0 else ""
            
            summaries.append(f"• {agent_type.upper()}: {risk_level} [ES: {es_context}]{time_str}{query_str}")
            
            # Add brief description if analysis summary available
            if 'analysis' in result and isinstance(result['analysis'], dict):
                analysis = result['analysis']
                if 'brief_summary' in analysis:
                    summaries.append(f"  └─ {analysis['brief_summary']}")
                elif 'status' in analysis:
                    summaries.append(f"  └─ Status: {analysis['status']}")
        
        return '\n'.join(summaries) if summaries else "• No agent summaries available"
    
    def _extract_key_findings(self, results: List[Dict]) -> str:
        """Extract key findings from agent results"""
        findings = []
        for result in results:
            agent_type = result.get('agent_type', 'unknown')
            
            # Extract from analysis section
            if 'analysis' in result and isinstance(result['analysis'], dict):
                analysis = result['analysis']
                
                # Get findings list
                agent_findings = analysis.get('findings', [])
                if agent_findings:
                    for finding in agent_findings[:2]:  # Limit to 2 per agent
                        findings.append(f"• [{agent_type.upper()}] {finding}")
                
                # Get summary or key points
                if 'summary' in analysis:
                    findings.append(f"• [{agent_type.upper()}] {analysis['summary']}")
                
                # Get risk assessment
                if 'risk_assessment' in analysis:
                    findings.append(f"• [{agent_type.upper()}] Risk: {analysis['risk_assessment']}")
            
            # Extract from report section - look for structured content
            if 'report' in result and isinstance(result['report'], str):
                report_lines = result['report'].split('\n')
                in_findings = False
                
                for line in report_lines:
                    line = line.strip()
                    if not line or line.startswith('='):
                        continue
                    
                    # Look for key information patterns
                    if any(keyword in line.lower() for keyword in ['ip address:', 'detected:', 'found:', 'analysis:', 'activity:', 'pattern:']):
                        findings.append(f"• [{agent_type.upper()}] {line}")
                    
                    # Stop after getting reasonable amount
                    if len([f for f in findings if agent_type.upper() in f]) >= 3:
                        break
            
            # Extract direct string results
            if isinstance(result.get('result'), str):
                result_text = result['result']
                # Look for meaningful lines
                for line in result_text.split('\n')[:3]:
                    if line.strip() and len(line.strip()) > 10:
                        findings.append(f"• [{agent_type.upper()}] {line.strip()}")
                        break
        
        return '\n'.join(findings[:8]) if findings else "• No significant findings detected"
    
    def _extract_error_summary(self, failed_results: List[Dict]) -> str:
        """Extract error summary"""
        if not failed_results:
            return "• No errors encountered"
        
        errors = []
        for result in failed_results:
            agent_type = result.get('agent_type', 'unknown')
            error = result.get('error', 'Unknown error')
            errors.append(f"• {agent_type.upper()}: {error}")
        
        return '\n'.join(errors)
    
    def _generate_recommendations(self, risk_level: str, results: List[Dict]) -> str:
        """Generate recommendations based on risk level and findings"""
        if risk_level == 'CRITICAL':
            return """• IMMEDIATE ACTION REQUIRED
• Isolate affected systems and accounts
• Begin incident response procedures
• Notify security team immediately
• Preserve forensic evidence
• Execute generated Elasticsearch queries for detailed investigation"""
        elif risk_level == 'HIGH':
            return """• Investigate within 2 hours
• Monitor affected accounts/systems closely
• Review and strengthen access controls
• Consider temporary access restrictions
• Escalate to security team
• Use Elasticsearch queries to gather additional context"""
        elif risk_level == 'MEDIUM':
            return """• Investigate within 24 hours
• Review user/system activity patterns
• Update monitoring rules if needed
• Document findings for future reference
• Monitor for escalation
• Analyze Elasticsearch query results for patterns"""
        else:
            return """• Continue monitoring situation
• Review alert tuning if false positive
• Document analysis for trend analysis
• Update detection rules as needed
• Store Elasticsearch queries for future reference"""
    
    def analyze_security_alert_integrated(self, events_file: str, alerts_file: str, 
                                        data_file: str, alert_number: int, 
                                        user_prompt: str) -> Dict[str, Any]:
        """Main function: Complete integrated security alert analysis"""
        try:
            self.logger.info(f"Starting integrated security analysis for alert {alert_number}")
            
            # Step 1: Initialize ES Query Builder with events file
            self.initialize_es_query_builder(events_file)
            
            # Step 2: Load alert and matching data using ES Query Builder
            alert, data_entry = self.load_alert_and_data(alerts_file, data_file, alert_number)
            self.logger.info(f"Loaded alert: {alert.get('alertName')}")
            
            # Step 3: Generate Elasticsearch queries
            es_query_result = self.generate_elasticsearch_queries(alert, data_entry, user_prompt)
            self.logger.info("Generated Elasticsearch queries")
            
            # Step 4: Orchestration planning with ES context
            orchestration_result = self.master_agent.orchestrate_investigation(
                alerts_file, alert_number, user_prompt
            )
            
            # Step 4.5: Override agent prompts for IP behavior queries
            if es_query_result.get('query_type') == 'ip_behavior':
                ips = es_query_result.get('extracted_ips', [])
                ip_subject = ', '.join(ips)
                
                self.logger.info(f"🎯 IP Behavior routing active: focusing on IPs {ip_subject}")
                
                # Update agent prompts to use IP-focused subject
                updated_prompts = {}
                for agent_name, original_prompt in orchestration_result['agent_prompts'].items():
                    if agent_name == 'ip_agent':
                        updated_prompts[agent_name] = f"Alert: {alert.get('alertName', 'Unknown')} (Severity: {alert.get('severity', 'Unknown')}). Focus on IP behavior analysis for: {ip_subject}. Task: {user_prompt}"
                    else:
                        # Keep other agents but mention IP focus
                        updated_prompts[agent_name] = original_prompt
                
                orchestration_result['agent_prompts'] = updated_prompts
                orchestration_result['ip_behavior_active'] = True
                orchestration_result['subject_ips'] = ips
            
            self.logger.info(f"Orchestration planned: {len(orchestration_result['agent_prompts'])} agents")
            
            # Step 5: Execute agents in parallel with ES context
            agent_results = self.execute_agents_parallel_with_es(
                orchestration_result['agent_prompts'],
                alert,
                data_entry
            )
            
            # Step 6: Compile final report with ES integration
            final_result = self.compile_final_report_with_es(
                orchestration_result, 
                agent_results, 
                es_query_result
            )
            
            self.logger.info("Integrated security analysis completed successfully")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error in integrated security analysis: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

# Enhanced CLI Interface
def main():
    """Enhanced command line interface with proper file integration"""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Integrated Security Alert Analysis System')
    parser.add_argument('events_file', nargs='?', help='Path to events file (CSV/JSON with event IDs and descriptions)')
    parser.add_argument('alerts_file', nargs='?', help='Path to alerts file (JSON/JSONL)')
    parser.add_argument('data_file', nargs='?', help='Path to data file (JSON/JSONL)')
    parser.add_argument('alert_number', nargs='?', type=int, help='Alert number to analyze')
    parser.add_argument('prompt', nargs='?', help='Analysis prompt/question')
    parser.add_argument('--config', help='Config file path (optional)', default='config/paths.yml')
    parser.add_argument('--output', help='Output file path (optional)')
    parser.add_argument('--save-queries', help='Save generated ES queries to file (optional)')
    
    args = parser.parse_args()
    
    # Load config file if arguments are not provided
    if not all([args.events_file, args.alerts_file, args.data_file, args.alert_number, args.prompt]):
        try:
            with open(args.config, 'r') as f:
                config_data = yaml.safe_load(f)
            paths = config_data['paths']
            
            # Use config values for missing arguments
            args.events_file = args.events_file or paths['events']
            args.alerts_file = args.alerts_file or paths['alerts'] 
            args.data_file = args.data_file or paths['data']
            
            # Interactive prompts for alert_number and prompt if not provided
            if args.alert_number is None:
                while True:
                    try:
                        args.alert_number = int(input("🔢 Enter Alert ID: "))
                        break
                    except ValueError:
                        print("❌ Please enter a valid integer")
            
            if not args.prompt:
                print("📝 Enter analysis command/question:")
                print("   Examples: investigate ip behaviour, analyze failed authentication attempts")
                args.prompt = input("🎯 Analysis command: ").strip()
                if not args.prompt:
                    print("❌ Analysis command cannot be empty")
                    return 1
            
            # Set output directory from config if not specified
            if not args.output:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_command = "".join(c for c in args.prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_command = safe_command.replace(' ', '_')[:50]
                args.output = os.path.join(paths['output_dir'], f"analysis_alert{args.alert_number}_{safe_command}_{timestamp}.json")
                
        except Exception as e:
            print(f"❌ Error loading config file '{args.config}': {e}")
            print("💡 Either provide all arguments or ensure config/paths.yml exists")
            return 1
    
    try:
        # Validate file existence
        for file_path, file_type in [
            (args.events_file, 'events'),
            (args.alerts_file, 'alerts'), 
            (args.data_file, 'data')
        ]:
            if not os.path.exists(file_path):
                print(f"Error: {file_type} file not found: {file_path}")
                return 1
        
        # Initialize orchestrator
        config = Config()
        orchestrator = IntegratedSecurityAnalysisOrchestrator(config)
        
        # Run integrated analysis
        result = orchestrator.analyze_security_alert_integrated(
            args.events_file,
            args.alerts_file,
            args.data_file,
            args.alert_number,
            args.prompt
        )
        
        # Save ES queries if requested
        if args.save_queries:
            # Always ensure queries go to the reports directory
            config_data = yaml.safe_load(open(args.config, 'r'))
            output_dir = config_data['paths']['output_dir']
            
            # Extract just the filename from args.save_queries if it contains a path
            queries_filename = os.path.basename(args.save_queries)
            queries_path = os.path.join(output_dir, queries_filename)
            
            es_queries = result.get('elasticsearch_context', {})
            with open(queries_path, 'w') as f:
                json.dump(es_queries, f, indent=2, default=str)
            print(f"Elasticsearch queries saved to {queries_path}")
        
        # Output results
        if args.output:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alert_id = result['orchestration']['selected_alert'].get('alertId', 'unknown')[:8]
            
            # Always ensure output goes to the reports directory
            config_data = yaml.safe_load(open(args.config, 'r'))
            output_dir = config_data['paths']['output_dir']
            
            # Extract just the filename from args.output if it contains a path
            output_filename = os.path.basename(args.output)
            base_name, ext = os.path.splitext(output_filename)
            if not ext:
                ext = '.json'
            
            # Create the filename in the reports directory
            unique_filename = os.path.join(output_dir, f"{base_name}_integrated_alert{args.alert_number}_{alert_id}_{timestamp}{ext}")
            
            with open(unique_filename, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Integrated analysis results saved to {unique_filename}")
        else:
            print(result['executive_summary'])
            print(f"\nDetailed analysis completed with {result['successful_agents']} successful agents")
            print(f"ES Integration: {result['es_integration_stats']['relevant_events_found']} relevant events found")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())