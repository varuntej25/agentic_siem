#!/usr/bin/env python3
"""Quick test to check event ID selection"""

from elastic_search import ElasticsearchQueryBuilder

def test_event_selection():
    """Test the new event ID selection logic"""
    
    # Initialize the query builder with minimal setup
    try:
        query_builder = ElasticsearchQueryBuilder(csv_file_path="Spharaka-Windows_Events_Filtered.json")
    except:
        # Create a minimal query builder just for testing event selection
        query_builder = ElasticsearchQueryBuilder.__new__(ElasticsearchQueryBuilder)
        # We just need the find_relevant_events method, so minimal init
    
    # Test different prompts
    test_prompts = [
        "investigate user logins",
        "analyze user behavior patterns",
        "check IP address activity",
        "investigate network connections",
        "user authentication analysis",
        "investigate process execution",
        "check file access patterns"
    ]
    
    print("=== Event ID Selection Test ===")
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        relevant_events = query_builder.find_relevant_events(prompt)
        
        event_ids = [event['event_id'] for event in relevant_events]
        print(f"Selected Event IDs: {event_ids}")
        
        # Show first few descriptions
        for event in relevant_events[:3]:
            print(f"  - {event['event_id']}: {event['description'][:60]}...")

if __name__ == '__main__':
    test_event_selection()
