#!/usr/bin/env python3

import json
from pathlib import Path

def debug_latest_report():
    print("=== Debug Latest IP Behavior Report ===")
    
    # Find the latest report
    report_files = list(Path('reports').glob('ip_behavior_fixed*.json'))
    if not report_files:
        print("❌ No fixed IP behavior reports found")
        return
    
    latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Analyzing: {latest_report}")
    
    with open(latest_report, 'r') as f:
        report = json.load(f)
    
    # Look for IP agent analysis details
    agent_analyses = report.get('agent_analyses', {})
    ip_agent_data = None
    
    for analysis in agent_analyses:
        if analysis.get('agent_type') == 'ip_agent':
            ip_agent_data = analysis
            break
    
    if ip_agent_data:
        print(f"✅ Found IP agent analysis")
        print(f"📈 Queries executed: {ip_agent_data.get('queries_executed', 'Not specified')}")
        print(f"🎯 Risk score: {ip_agent_data.get('risk_score', 'Not specified')}")
        print(f"🔍 Findings count: {len(ip_agent_data.get('findings', []))}")
        
        findings = ip_agent_data.get('findings', [])
        print(f"\n📋 Findings:")
        for i, finding in enumerate(findings):
            print(f"   {i+1}. {finding}")
        
        # Check metadata
        metadata = ip_agent_data.get('metadata', {})
        if metadata:
            print(f"\n🔧 Metadata:")
            for key, value in metadata.items():
                print(f"   {key}: {value}")
        
        # Check ES context
        es_context = ip_agent_data.get('es_context', {})
        print(f"\n🔍 ES Context:")
        print(f"   Queries available: {es_context.get('queries_available', 'N/A')}")
        print(f"   Relevant events: {es_context.get('relevant_events_count', 'N/A')}")
        print(f"   Data entries available: {es_context.get('data_entries_available', 'N/A')}")
    else:
        print("❌ No IP agent analysis found")
    
    # Check overall summary
    summary = report.get('summary', {})
    print(f"\n📄 Overall Summary:")
    print(f"   Risk Level: {summary.get('risk_assessment', {}).get('risk_level', 'N/A')}")
    print(f"   Overall Score: {summary.get('risk_assessment', {}).get('overall_score', 'N/A')}")

if __name__ == "__main__":
    debug_latest_report()
