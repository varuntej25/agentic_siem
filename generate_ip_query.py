#!/usr/bin/env python3
"""Generate and save the IP behavior Elasticsearch query for external testing"""

import json
import sys
import os
sys.path.append(os.getcwd())

from elastic_search import ElasticsearchQueryBuilder

def generate_ip_query_for_testing():
    print("=== Generating IP Behavior Query for Testing ===\n")
    
    try:
        # Load first alert
        with open('alerts.jsonl', 'r') as f:
            first_alert = json.loads(f.readline())
        
        print(f"Alert: {first_alert['alertName']}")
        print(f"Alert Events: {first_alert['events']}")
        print()
        
        # Initialize ES Query Builder
        builder = ElasticsearchQueryBuilder('Spharaka-Windows_Events_Filtered.json')
        
        # Extract IPs and generate query
        ip_result = builder.extract_ips_from_alert_events(first_alert, 'data.json')
        print(f"Extracted IPs: {ip_result['ips']}")
        
        if ip_result['ips']:
            # Generate the IP-focused query
            query_result = builder.build_ip_focus_query(ip_result['ips'])
            
            # Save the query to a file for manual testing
            query_file = 'ip_behavior_query.json'
            with open(query_file, 'w', encoding='utf-8') as f:
                json.dump(query_result['elasticsearch_query'], f, indent=2)
            
            print(f"✅ IP-focused query saved to: {query_file}")
            print()
            
            # Show what the query would search for
            print("📊 Query Summary:")
            query = query_result['elasticsearch_query']
            print(f"  Size limit: {query.get('size', 'N/A')}")
            print(f"  Target IPs: {ip_result['ips']}")
            
            # Show the query structure
            bool_query = query.get('query', {}).get('bool', {})
            should_clauses = bool_query.get('should', [])
            print(f"  IP field searches: {len(should_clauses)}")
            
            # Show time range
            filters = bool_query.get('filter', [])
            for f in filters:
                if 'range' in f:
                    print(f"  Time range: {f['range']}")
                    break
            
            # Show aggregations
            aggs = query.get('aggregations', {})
            print(f"  Aggregations: {list(aggs.keys())}")
            
            print()
            print("📝 To test this query manually:")
                print(f"   curl -X POST 'http://10.9.56.10:30200/spharaka-windows/_search' \\")
            print(f"        -H 'Content-Type: application/json' \\")
            print(f"        -d @{query_file}")
            
            # Also show what data exists for this IP in our local data
            print()
            print("🔍 Local data analysis for IP 10.0.0.1:")
            
            # Search through data.json for this IP
            ip_count = 0
            sample_events = []
            
            with open('data.json', 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num > 1000:  # Limit search for performance
                        break
                    try:
                        entry = json.loads(line.strip())
                        if "_source" in entry:
                            source = entry["_source"]
                            if source.get("srcIP") == "10.0.0.1":
                                ip_count += 1
                                if len(sample_events) < 3:
                                    sample_events.append({
                                        "eventID": source.get("eventID"),
                                        "eventType": source.get("eventType"),
                                        "timestamp": source.get("timestamp"),
                                        "hostName": source.get("hostName")
                                    })
                    except:
                        continue
            
            print(f"   Found {ip_count} events with IP 10.0.0.1 in first 1000 entries")
            if sample_events:
                print("   Sample events:")
                for event in sample_events:
                    print(f"     - Event {event['eventID']}: {event['eventType']} on {event['hostName']}")
            
            print()
            print("✅ IP behavior query generation complete!")
            
        else:
            print("❌ No IPs found to generate query")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_ip_query_for_testing()
