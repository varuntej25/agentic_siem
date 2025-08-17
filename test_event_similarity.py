#!/usr/bin/env python3
"""Test what events are found for different prompts"""

import sys
import os
import pandas as pd
sys.path.append(os.getcwd())

from elastic_search import ElasticsearchQueryBuilder

def test_event_similarity():
    print("=== Testing Event Similarity Matching ===\n")
    
    try:
        builder = ElasticsearchQueryBuilder('Spharaka-Windows_Events_Filtered.json')
        print("✅ ES Query Builder initialized\n")
        
        test_prompts = [
            "investigate ip behaviour",
            "network connection",
            "authentication failed", 
            "logon attempt",
            "security event",
            "firewall",
            "connection established"
        ]
        
        for prompt in test_prompts:
            print(f"🔍 Testing prompt: '{prompt}'")
            relevant_events = builder.find_relevant_events(prompt, top_k=5, threshold=0.2)
            
            if relevant_events:
                print(f"   Found {len(relevant_events)} relevant events:")
                for event in relevant_events:
                    print(f"     - Event {event['event_id']}: {event['description'][:100]}... (similarity: {event['similarity_score']:.3f})")
            else:
                print("   No relevant events found (similarity too low)")
            print()
            
        # Also show some sample events from the file
        print("📋 Sample events in the file:")
        print(f"   Total events: {len(builder.events_df)}")
        for idx in range(min(10, len(builder.events_df))):
            event_id = builder.events_df.iloc[idx]['event_id']
            description = builder.events_df.iloc[idx]['description']
            print(f"     - Event {event_id}: {description}")
            
        # Search for network-related events specifically
        print("\n🌐 Looking for network-related event descriptions...")
        network_keywords = ['network', 'connection', 'firewall', 'share', 'remote', 'ip']
        network_events = []
        
        for idx, row in builder.events_df.iterrows():
            desc = row['description']
            if pd.isna(desc):  # Skip NaN descriptions
                continue
            desc_lower = str(desc).lower()
            if any(keyword in desc_lower for keyword in network_keywords):
                network_events.append((row['event_id'], desc))
                
        print(f"   Found {len(network_events)} network-related events:")
        for event_id, desc in network_events[:10]:  # Show first 10
            print(f"     - Event {event_id}: {desc}")
            
        print("\n🎯 Let's check the exact threshold used in main analysis...")
        # Test with different thresholds
        for threshold in [0.3, 0.5, 0.7, 0.8]:
            relevant = builder.find_relevant_events("investigate ip behaviour", top_k=5, threshold=threshold)
            print(f"   Threshold {threshold}: {len(relevant)} events found")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_event_similarity()
