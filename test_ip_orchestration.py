#!/usr/bin/env python3
"""Test the complete IP behavior flow in main orchestrator"""

import sys
import os
sys.path.append(os.getcwd())

from main_orchestrator import IntegratedSecurityAnalysisOrchestrator

def test_ip_behavior_orchestration():
    print("=== Testing Complete IP Behavior Flow ===\n")
    
    try:
        # Initialize orchestrator
        print("Initializing Security Orchestrator...")
        orchestrator = IntegratedSecurityAnalysisOrchestrator()
        print("✅ Security Orchestrator initialized\n")
        
        # Test with IP behavior prompt
        print("Testing IP behavior orchestration...")
        
        # Use the test files
        events_file = "Spharaka-Windows_Events_Filtered.json"
        alerts_file = "alerts.jsonl"  
        data_file = "data.json"
        alert_number = 1  # First alert
        user_prompt = "investigate ip behaviour"
        
        print(f"Configuration:")
        print(f"  Events file: {events_file}")
        print(f"  Alerts file: {alerts_file}")
        print(f"  Data file: {data_file}")
        print(f"  Alert number: {alert_number}")
        print(f"  User prompt: '{user_prompt}'")
        print()
        
        # Run the analysis
        print("Running integrated security analysis...")
        result = orchestrator.analyze_security_alert_integrated(
            events_file, alerts_file, data_file, alert_number, user_prompt
        )
        
        # Display key results
        print("✅ Analysis completed successfully!\n")
        
        print("Key Results:")
        print(f"  Alert: {result['selected_alert']['alertName']}")
        print(f"  Query Type: {result.get('query_type', 'unknown')}")
        
        if result.get('orchestration_result', {}).get('ip_behavior_active'):
            print(f"  🎯 IP Behavior Active: YES")
            print(f"  Target IPs: {result['orchestration_result']['subject_ips']}")
        else:
            print(f"  🎯 IP Behavior Active: NO")
            
        print(f"  Agents Executed: {list(result.get('agent_results', {}).keys())}")
        print(f"  ES Queries Generated: {result.get('es_query_count', 'unknown')}")
        
        # Show ES query details
        if 'es_query_result' in result:
            es_result = result['es_query_result']
            if es_result.get('query_type') == 'ip_behavior':
                print(f"\n📊 IP Behavior Query Details:")
                print(f"  Extracted IPs: {es_result.get('extracted_ips', [])}")
                print(f"  Subject: {es_result.get('subject', 'N/A')}")
                print(f"  Description: {es_result.get('description', 'N/A')}")
        
        print(f"\n📋 Executive Summary Preview:")
        summary = result.get('executive_summary', '')
        # Show first few lines
        lines = summary.split('\n')[:10]
        for line in lines:
            if line.strip():
                print(f"  {line}")
        
        print("\n=== Test Complete ===")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ip_behavior_orchestration()
    sys.exit(0 if success else 1)
