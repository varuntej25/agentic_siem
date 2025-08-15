#!/usr/bin/env python3
"""Debug ES query execution"""

import sys
sys.path.append('.')
import json
from enhanced_agent_base import BaseAgentWithES

print("=== Testing ES Query Execution ===")

# Create a test agent
agent = BaseAgentWithES("http://test")

# Mock data entries (using the data we know contains the IP)
mock_data_entries = []
with open('data.json', 'r') as f:
    lines = f.read().strip().split('\n')
    for line in lines[:10]:  # First 10 entries
        try:
            entry = json.loads(line)
            mock_data_entries.append(entry)
        except json.JSONDecodeError:
            continue

print(f"Loaded {len(mock_data_entries)} mock data entries")

# Set the context
agent.current_es_context = {
    'all_data_entries': mock_data_entries
}

# Create a test IP query
test_query = {
    "size": 100,
    "query": {
        "bool": {
            "should": [
                {
                    "terms": {
                        "srcIP": ["192.168.1.104"]
                    }
                },
                {
                    "terms": {
                        "sourceIP": ["192.168.1.104"]
                    }
                }
            ]
        }
    }
}

print(f"\n🔍 Testing ES query execution:")
print(f"Query: {json.dumps(test_query, indent=2)}")

# Execute the query
result = agent.execute_es_query(test_query)

print(f"\n📊 Query Results:")
print(f"Total hits: {result.get('hits', {}).get('total', {}).get('value', 0)}")

hits = result.get('hits', {}).get('hits', [])
print(f"Actual hits returned: {len(hits)}")

if hits:
    print(f"\n✅ Sample results:")
    for i, hit in enumerate(hits[:3]):
        source = hit.get('_source', {})
        hit_id = hit.get('_id', 'unknown')
        score = hit.get('_score', 0)
        print(f"  Hit {i+1}: ID={hit_id}, score={score}")
        for field, value in source.items():
            if '192.168.1.104' in str(value):
                print(f"    -> {field}: {value}")
else:
    print(f"\n❌ No matching results found")
    
    # Debug: Check if our test data has the IP
    print(f"\n🔍 Debug: Checking if test data contains the IP...")
    for i, entry in enumerate(mock_data_entries):
        source = entry.get('_source', {})
        for field, value in source.items():
            if '192.168.1.104' in str(value):
                print(f"  Entry {i}: Found IP in {field}: {value}")
