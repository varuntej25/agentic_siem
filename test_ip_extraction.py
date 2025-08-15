#!/usr/bin/env python3
"""Test IP extraction functionality"""

import json
from elastic_search import ElasticsearchQueryBuilder

def test_ip_extraction():
    print("=== Testing IP Extraction for IP Behavior Flow ===\n")
    
    try:
        # Load first alert
        with open('alerts.jsonl', 'r') as f:
            first_alert = json.loads(f.readline())
        
        print(f"Alert ID: {first_alert['alertId']}")
        print(f"Alert Name: {first_alert['alertName']}")
        print(f"Alert Events: {first_alert['events']}")
        print()
        
        # Initialize ES Query Builder
        print("Initializing ES Query Builder...")
        builder = ElasticsearchQueryBuilder('Spharaka-Windows_Events_Filtered.json')
        print("✅ ES Query Builder initialized successfully\n")
        
        # Test IP extraction
        print("Testing IP extraction...")
        result = builder.extract_ips_from_alert_events(first_alert, 'data.json')
        
        print(f"IP Extraction Result:")
        print(f"  IPs found: {result['ips']}")
        print(f"  Event IDs: {result['event_ids']}")
        print(f"  No IP extracted: {result['no_ip_extracted']}")
        if result['no_ip_extracted']:
            print(f"  Reason: {result['reason']}")
        print()
        
        # Test IP behavior intent detection
        print("Testing IP behavior intent detection...")
        test_prompts = [
            "investigate ip behaviour",
            "analyze user activity", 
            "investigate network behavior",
            "show me ip patterns",
            "what is the timeline",
            "ip activity analysis"
        ]
        
        for prompt in test_prompts:
            is_ip_intent = builder._is_ip_behavior_intent(prompt)
            print(f"  '{prompt}' -> IP behavior: {is_ip_intent}")
        print()
        
        # Test IP-focused query building if IPs were found
        if result['ips'] and not result['no_ip_extracted']:
            print("Testing IP-focused query building...")
            ip_query = builder.build_ip_focus_query(result['ips'])
            print(f"✅ IP-focused query generated for IPs: {result['ips']}")
            print(f"Query type: {ip_query.get('metadata', {}).get('query_type', 'unknown')}")
            print(f"Target IPs: {ip_query.get('metadata', {}).get('target_ips', [])}")
            print()
        
        # Test full IP behavior query flow
        print("Testing full IP behavior query flow...")
        full_result = builder._process_ip_behavior_query(first_alert, "investigate ip behaviour")
        
        if 'error' in full_result:
            print(f"❌ Error in IP behavior flow: {full_result['error']}")
            print(f"   Reason: {full_result['reason']}")
        else:
            print(f"✅ IP behavior query flow successful!")
            print(f"   Query type: {full_result['query_type']}")
            print(f"   Extracted IPs: {full_result['extracted_ips']}")
            print(f"   Subject: {full_result['subject']}")
        
        print("\n=== Test Complete ===")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ip_extraction()
