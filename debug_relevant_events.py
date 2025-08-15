#!/usr/bin/env python3
"""Debug why main orchestrator gets 0 relevant events"""

import sys
sys.path.append('.')
from main_orchestrator import IntegratedSecurityAnalysisOrchestrator

print("=== Debugging Relevant Events Issue ===")

# Create a test class that can access the es_query_builder
class DebuggingOrchestrator(IntegratedSecurityAnalysisOrchestrator):
    def debug_relevant_events(self, user_prompt):
        """Debug the relevant events call"""
        print(f"Checking relevant events for: '{user_prompt}'")
        
        if not hasattr(self, 'es_query_builder') or self.es_query_builder is None:
            print("❌ es_query_builder is None or not initialized")
            return
            
        print("✅ es_query_builder exists")
        
        try:
            print("🔍 Calling find_relevant_events...")
            relevant_events = self.es_query_builder.find_relevant_events(user_prompt)
            print(f"✅ Found {len(relevant_events)} relevant events")
            
            if relevant_events:
                for i, event in enumerate(relevant_events[:3]):
                    event_id = event.get('event_id', 'N/A')  
                    similarity = event.get('similarity_score', 0)
                    print(f"   Event {i+1}: ID={event_id}, similarity={similarity:.3f}")
            else:
                print("   No events returned")
                
        except Exception as e:
            print(f"❌ Exception in find_relevant_events: {e}")
            import traceback
            traceback.print_exc()

# Test
orchestrator = DebuggingOrchestrator()

# First initialize the orchestrator properly
events_file = 'Spharaka-Windows_Events_Filtered.json'
alerts_file = 'alerts.jsonl'
data_file = 'data.json'
alert_number = 15
user_prompt = 'investigate ip behaviour'

print("Initializing orchestrator...")
try:
    # Initialize the ES query builder first
    from elastic_search import ElasticsearchQueryBuilder
    orchestrator.es_query_builder = ElasticsearchQueryBuilder(events_file)
    print("✅ es_query_builder initialized manually")
    
    # Now debug the relevant events call
    orchestrator.debug_relevant_events(user_prompt)
    
except Exception as e:
    print(f"❌ Error during initialization: {e}")
    import traceback
    traceback.print_exc()
