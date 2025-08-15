#!/usr/bin/env python3

import json
import sys
sys.path.append('.')
from enhanced_agent_base import BaseAgentWithES

# Create a test IP agent
class DebugIPAgent(BaseAgentWithES):
    def __init__(self):
        super().__init__("debug-ip", "Debug IP Agent")
    
    def analyze(self, alert, data_entry, es_context):
        return {"findings": ["debug"], "risk_score": 0}

def test_ip_analysis():
    print("=== Debug IP Analysis Process ===")
    
    # Load the alert data
    with open('data.json', 'r') as f:
        data_entries = [json.loads(line) for line in f]
    
    print(f"Loaded {len(data_entries)} data entries")
    
    # Find an entry with our target IP
    target_ip = "192.168.1.104"
    ip_entry = None
    for entry in data_entries:
        source = entry.get('_source', {})
        if source.get('srcIP') == target_ip:
            ip_entry = entry
            break
    
    if not ip_entry:
        print(f"❌ No entry found with IP {target_ip}")
        return
    
    print(f"✅ Found entry with IP {target_ip}")
    
    # Create debug agent
    agent = DebugIPAgent()
    
    # Mock alert
    alert = {
        'alertId': 'debug-123',
        'alertName': 'Debug IP Test',
        'severity': 'Medium',
        'createdAt': '2024-08-14T12:30:00Z'
    }
    
    # Extract IP context
    ip_context = agent._extract_ip_context(alert, ip_entry)
    print(f"📍 IP Context: {json.dumps(ip_context, indent=2)}")
    
    # Mock ES context with all data entries
    es_context = {
        'all_data_entries': data_entries,
        'es_queries': {
            'contextual_query': {
                'query': {'match_all': {}},
                'size': 100
            }
        },
        'relevant_events': []
    }
    
    # Build IP queries
    ip_queries = agent._build_ip_queries(ip_context['primary_ip'], es_context)
    print(f"🔧 Built {len(ip_queries)} IP queries")
    for query_name in ip_queries.keys():
        print(f"  - {query_name}")
    
    # Execute queries and analyze
    context_data = {}
    for query_name, query in ip_queries.items():
        print(f"\n🔍 Executing query: {query_name}")
        result = agent.execute_es_query(query)
        print(f"   Raw result: {result.get('hits', {}).get('total', 0)} hits")
        
        context_data[query_name] = agent.extract_context_from_es_results(result)
        print(f"   Context data: {json.dumps(context_data[query_name], indent=2, default=str)}")
    
    # Run the analysis
    print(f"\n🧠 Running IP behavior analysis...")
    analysis = agent._analyze_ip_behavior(ip_context, context_data, es_context)
    print(f"📊 Analysis results:")
    print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    test_ip_analysis()
