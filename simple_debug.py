#!/usr/bin/env python3

import json
from pathlib import Path

def simple_debug():
    print("=== Simple IP Analysis Debug ===")
    
    # Load the latest report
    report_files = list(Path('.').glob('*ip_behavior*alert15*.json'))
    if not report_files:
        print("❌ No IP behavior report found")
        return
    
    latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Loading report: {latest_report}")
    
    with open(latest_report, 'r') as f:
        report = json.load(f)
    
    # Check what agents were involved
    agents = report.get('agent_analyses', {})
    print(f"🤖 Agents involved: {list(agents.keys())}")
    
    # Look at IP agent analysis
    for agent_name, analysis in agents.items():
        if 'ip' in agent_name.lower():
            print(f"\n🔍 {agent_name} Analysis:")
            print(f"   Findings: {len(analysis.get('findings', []))}")
            for i, finding in enumerate(analysis.get('findings', [])[:3]):
                print(f"     {i+1}. {finding}")
            
            print(f"   Risk Score: {analysis.get('risk_score', 0)}")
            print(f"   Queries Executed: {analysis.get('queries_executed', 0)}")
            
            # Check context data
            if 'metadata' in analysis:
                print(f"   Metadata keys: {list(analysis['metadata'].keys())}")
    
    # Check overall findings
    overall_findings = report.get('findings', [])
    print(f"\n📋 Overall findings: {len(overall_findings)}")
    
    # Check risk assessment
    risk = report.get('risk_assessment', {})
    print(f"🎯 Overall risk score: {risk.get('overall_score', 0)}")
    print(f"🎯 Risk level: {risk.get('risk_level', 'Unknown')}")

if __name__ == "__main__":
    simple_debug()
