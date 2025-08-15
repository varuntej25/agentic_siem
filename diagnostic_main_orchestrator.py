# diagnostic_main_orchestrator.py - Debug version to identify issues
import json
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from master_agent import MasterAgent

@dataclass
class Config:
    """Centralized configuration management"""
    MODEL_NAME: str = field(default_factory=lambda: os.getenv('MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2'))
    RISK_CRITICAL: int = field(default_factory=lambda: int(os.getenv('RISK_CRITICAL', '80')))
    RISK_HIGH: int = field(default_factory=lambda: int(os.getenv('RISK_HIGH', '60')))
    RISK_MEDIUM: int = field(default_factory=lambda: int(os.getenv('RISK_MEDIUM', '40')))
    RISK_LOW: int = field(default_factory=lambda: int(os.getenv('RISK_LOW', '20')))

class SecurityAnalysisOrchestrator:
    """Diagnostic version to identify issues"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.setup_logging()
        self.master_agent = MasterAgent(model_name=self.config.MODEL_NAME)
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
    
    def debug_data_structure(self, alerts_file: str, alert_number: int, user_prompt: str):
        """Debug the data structures to identify issues"""
        try:
            self.logger.info("=== DEBUGGING DATA STRUCTURES ===")
            
            # Step 1: Test alert loading
            self.logger.info("Step 1: Loading alerts...")
            alerts = self.master_agent.load_alerts(alerts_file)
            self.logger.info(f"Loaded {len(alerts)} alerts")
            self.logger.info(f"First alert keys: {list(alerts[0].keys()) if alerts else 'No alerts'}")
            
            # Step 2: Get specific alert
            self.logger.info(f"Step 2: Getting alert #{alert_number}...")
            selected_alert = self.master_agent.get_alert_by_number(alerts, alert_number)
            self.logger.info(f"Selected alert type: {type(selected_alert)}")
            self.logger.info(f"Selected alert keys: {list(selected_alert.keys()) if isinstance(selected_alert, dict) else 'Not a dict'}")
            self.logger.info(f"Alert name: {selected_alert.get('alertName') if isinstance(selected_alert, dict) else 'Cannot access'}")
            
            # Step 3: Test orchestration
            self.logger.info("Step 3: Testing orchestration...")
            orchestration_result = self.master_agent.orchestrate_investigation(alerts_file, alert_number, user_prompt)
            self.logger.info(f"Orchestration result keys: {list(orchestration_result.keys())}")
            self.logger.info(f"Required agents: {orchestration_result.get('intent_analysis', {}).get('required_agents')}")
            self.logger.info(f"Agent prompts keys: {list(orchestration_result.get('agent_prompts', {}).keys())}")
            
            # Step 4: Show agent prompt structure
            agent_prompts = orchestration_result.get('agent_prompts', {})
            for agent_name, prompt in agent_prompts.items():
                self.logger.info(f"Agent {agent_name} prompt type: {type(prompt)}")
                self.logger.info(f"Agent {agent_name} prompt: {prompt[:100]}..." if isinstance(prompt, str) else f"Not a string: {prompt}")
            
            # Step 5: Test mock data entry
            mock_data_entry = {
                '_source': {
                    'eventID': '4625',
                    'targetUsername': 'test_user',
                    'srcIP': '192.168.1.100',
                    'hostName': 'test_host',
                    'timestamp': datetime.now().timestamp()
                }
            }
            self.logger.info(f"Mock data entry: {mock_data_entry}")
            
            return {
                'alerts_loaded': len(alerts),
                'selected_alert': selected_alert,
                'orchestration_result': orchestration_result,
                'mock_data_entry': mock_data_entry
            }
            
        except Exception as e:
            self.logger.error(f"Debug error: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

def main():
    """Diagnostic main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnostic Security Alert Analysis')
    parser.add_argument('alerts_file', help='Path to alerts file (JSON/JSONL)')
    parser.add_argument('alert_number', type=int, help='Alert number to analyze')
    parser.add_argument('prompt', help='Analysis prompt/question')
    
    args = parser.parse_args()
    
    try:
        orchestrator = SecurityAnalysisOrchestrator()
        debug_result = orchestrator.debug_data_structure(args.alerts_file, args.alert_number, args.prompt)
        
        if debug_result:
            print("\n=== DEBUG SUMMARY ===")
            print(f"Alerts loaded: {debug_result['alerts_loaded']}")
            print(f"Selected alert type: {type(debug_result['selected_alert'])}")
            print(f"Orchestration successful: {debug_result['orchestration_result'] is not None}")
            
            # Show the actual selected alert content
            alert = debug_result['selected_alert']
            if isinstance(alert, dict):
                print(f"Alert name: {alert.get('alertName', 'No name')}")
                print(f"Alert severity: {alert.get('severity', 'No severity')}")
                print(f"Alert keys: {list(alert.keys())}")
            else:
                print(f"Alert is not a dict, it's: {type(alert)}")
                print(f"Alert content: {alert}")
        
    except Exception as e:
        print(f"Diagnostic error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()