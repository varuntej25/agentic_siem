import json
from typing import Dict, List, Any
from datetime import datetime

class SummaryAgent:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Import dynamic config system
        from dynamic_config_system import get_config
        self.config = get_config()
        
    def compile_analysis(self, alert: Dict, agent_results: List[Dict], user_prompt: str) -> Dict[str, Any]:
        """Main function: Label Normalization + Story-Style Final Report"""
        
        # Extract included agents from the orchestration context
        # This should come from orchestration.intent_analysis.required_agents but we need to derive it
        included_agents = self._extract_included_agents(agent_results)
        
        # Step 1: Normalize status labels for all included agents
        agent_status = self._normalize_agent_labels(agent_results, included_agents)
        
        # Step 2: Calculate overall risk
        overall_risk_level, overall_risk_score, overall_risk_reason = self._calculate_overall_risk(agent_status)
        
        # Step 3: Generate story-style narrative
        executive_summary = self._generate_story_narrative(alert, agent_results, agent_status, 
                                                         overall_risk_level, overall_risk_score, user_prompt)
        
        # Step 4: Extract highlights
        highlights = self._extract_highlights(agent_results, agent_status)
        
        return {
            "overall_risk_level": overall_risk_level,
            "overall_risk_score": overall_risk_score,
            "overall_risk_reason": overall_risk_reason,
            "executive_summary": executive_summary,
            "agent_status": agent_status,
            "highlights": highlights,
            "included_agents": included_agents,
            "errors": []
        }
    
    def _extract_included_agents(self, agent_results: List[Dict]) -> List[str]:
        """Extract list of agents that actually executed"""
        included = []
        
        # Always include summary first
        included.append("summary")
        
        # Add other agents based on results
        for result in agent_results:
            agent_type = result.get('agent_type')
            if agent_type and agent_type != 'summary' and result.get('status') != 'failed':
                if agent_type not in included:
                    included.append(agent_type)
        
        return included
    
    def _normalize_agent_labels(self, agent_results: List[Dict], included_agents: List[str]) -> Dict[str, Dict[str, str]]:
        """Normalize status labels for every included agent (no UNKNOWN ever)"""
        agent_status = {}
        
        # Create agent result lookup
        agent_lookup = {}
        for result in agent_results:
            agent_type = result.get('agent_type')
            if agent_type:
                agent_lookup[agent_type] = result
        
        for agent in included_agents:
            if agent == 'summary':
                # Summary always gets NORMAL unless there are critical findings
                agent_status[agent] = {
                    "status_label": "NORMAL",
                    "reason": "Summary compilation completed successfully."
                }
                continue
            
            result = agent_lookup.get(agent, {})
            
            if result.get('status') == 'failed':
                agent_status[agent] = {
                    "status_label": "N/A",
                    "reason": f"Agent failed to execute: {result.get('error', 'Unknown error')}"
                }
                continue
            
            # Apply precedence rules for label normalization
            status_label, reason = self._apply_label_precedence(result, agent)
            agent_status[agent] = {
                "status_label": status_label,
                "reason": reason
            }
        
        return agent_status
    
    def _apply_label_precedence(self, result: Dict, agent_type: str) -> tuple:
        """Apply precedence rules to derive status_label"""
        
        # Rule 1: If agent already provides string risk level, map aliases
        if 'risk_level' in result:
            risk_level = result['risk_level'].upper() if isinstance(result['risk_level'], str) else str(result['risk_level']).upper()
            
            # Map aliases
            risk_mapping = {
                'NONE': 'NORMAL', 'INFORMATIONAL': 'NORMAL', 'BENIGN': 'NORMAL',
                'LOW': 'LOW', 
                'MEDIUM': 'MODERATE', 'MODERATE': 'MODERATE', 'MED': 'MODERATE',
                'HIGH': 'HIGH',
                'CRITICAL': 'CRITICAL', 'SEVERE': 'CRITICAL'
            }
            
            if risk_level in risk_mapping:
                mapped_level = risk_mapping[risk_level]
                reason = self._generate_reason_from_result(result, agent_type)
                return mapped_level, reason
        
        # Rule 2: If agent provides numeric risk_score (0-100), bin it
        if 'risk_score' in result:
            score = result['risk_score']
            if isinstance(score, (int, float)):
                if score >= 80:
                    status = 'CRITICAL'
                elif score >= 60:
                    status = 'HIGH'
                elif score >= 40:
                    status = 'MODERATE'
                elif score >= 20:
                    status = 'LOW'
                else:
                    status = 'NORMAL'
                
                reason = self._generate_reason_from_result(result, agent_type)
                return status, reason
        
        # Rule 3: Infer qualitatively from text and fields
        status_label = self._infer_from_content(result, agent_type)
        reason = self._generate_reason_from_result(result, agent_type)
        
        return status_label, reason
    
    def _infer_from_content(self, result: Dict, agent_type: str) -> str:
        """Infer status qualitatively from text and fields"""
        
        # Check for explicit "no activity" patterns
        findings = []
        if 'analysis' in result and isinstance(result['analysis'], dict):
            findings = result['analysis'].get('findings', [])
            anomalies = result['analysis'].get('anomalies', [])
            patterns = result['analysis'].get('patterns', [])
        
        # Special handling for different agent types
        if agent_type == 'timeline':
            # Timeline: NORMAL unless explicit anomaly bursts/suspicious sequences
            timeline_analysis = result.get('timeline_analysis', {})
            if timeline_analysis.get('anomalies') or timeline_analysis.get('patterns'):
                return 'LOW'  # Minor timeline anomalies
            return 'NORMAL'
        
        elif agent_type == 'alert_history':
            # History: If no similar incidents, set NORMAL (don't escalate on "novel")
            historical_analysis = result.get('historical_analysis', {})
            threat_assessment = result.get('threat_assessment', '').upper()
            
            if 'NOVEL' in threat_assessment or 'NO_MATCHES' in threat_assessment:
                return 'NORMAL'
            elif 'RECURRENCE' in threat_assessment or 'PATTERN' in threat_assessment:
                return 'MODERATE'
            elif 'CRITICAL' in threat_assessment:
                return 'CRITICAL'
            elif 'HIGH' in threat_assessment:
                return 'HIGH'
            else:
                return 'NORMAL'
        
        # General inference rules
        content_text = str(result).lower()
        
        # Look for "no activity" patterns
        no_activity_patterns = ['no activity', 'no patterns', 'no matches', 'not found', 'no data', 'no results']
        if any(pattern in content_text for pattern in no_activity_patterns):
            return 'NORMAL'
        
        # Look for clear indicators of compromise
        high_risk_patterns = ['credential misuse', 'lateral movement', 'compromise', 'malicious', 'attack', 'breach']
        if any(pattern in content_text for pattern in high_risk_patterns):
            return 'HIGH'
        
        # Look for multiple anomalies
        if isinstance(findings, list) and len(findings) > 2:
            return 'MODERATE'
        elif isinstance(findings, list) and len(findings) > 0:
            return 'LOW'
        
        # Default to NORMAL
        return 'NORMAL'
    
    def _generate_reason_from_result(self, result: Dict, agent_type: str) -> str:
        """Generate one-line factual reason from agent findings"""
        
        # Try to extract key findings
        if 'analysis' in result and isinstance(result['analysis'], dict):
            findings = result['analysis'].get('findings', [])
            if findings:
                return f"{findings[0][:60]}..." if len(findings[0]) > 60 else findings[0]
        
        # Check for report content
        if 'report' in result:
            report_lines = str(result['report']).split('\n')
            for line in report_lines:
                if line.strip() and 'FINDINGS:' in line.upper():
                    # Find the next line with actual content
                    idx = report_lines.index(line)
                    if idx + 1 < len(report_lines):
                        next_line = report_lines[idx + 1].strip()
                        if next_line.startswith('•') or next_line.startswith('-'):
                            clean_line = next_line.lstrip('•-').strip()
                            return clean_line[:60] + "..." if len(clean_line) > 60 else clean_line
        
        # Fallback based on agent type
        if agent_type == 'user_context':
            return "User behavior analysis completed."
        elif agent_type == 'ip_agent':
            return "IP/network behavior analysis completed."
        elif agent_type == 'timeline':
            return "Timeline sequence analysis completed."
        elif agent_type == 'alert_history':
            return "Historical pattern analysis completed."
        else:
            return f"{agent_type} analysis completed."
    
    def _calculate_overall_risk(self, agent_status: Dict) -> tuple:
        """Calculate overall risk level, score, and reason"""
        
        # Risk level ordering: NORMAL < LOW < MODERATE < HIGH < CRITICAL
        risk_order = ['NORMAL', 'LOW', 'MODERATE', 'HIGH', 'CRITICAL']
        max_risk_level = 'NORMAL'
        contributing_agents = []
        
        for agent, status_info in agent_status.items():
            status_label = status_info['status_label']
            if status_label != 'N/A' and status_label in risk_order:
                if risk_order.index(status_label) > risk_order.index(max_risk_level):
                    max_risk_level = status_label
                    contributing_agents = [f"{agent} = {status_label}"]
                elif status_label == max_risk_level and agent not in [c.split(' = ')[0] for c in contributing_agents]:
                    contributing_agents.append(f"{agent} = {status_label}")
        
        # Calculate representative score
        score_mapping = {
            'NORMAL': 10,
            'LOW': 30, 
            'MODERATE': 50,
            'HIGH': 70,
            'CRITICAL': 90
        }
        
        overall_risk_score = score_mapping.get(max_risk_level, 10)
        
        # Generate reason
        if contributing_agents:
            if len(contributing_agents) == 1:
                overall_risk_reason = f"Driven by {contributing_agents[0]}; others lower."
            else:
                overall_risk_reason = f"Driven by {', '.join(contributing_agents[:2])}."
        else:
            overall_risk_reason = "No significant risk indicators detected."
        
        return max_risk_level, overall_risk_score, overall_risk_reason
    
    def _generate_story_narrative(self, alert: Dict, agent_results: List[Dict], 
                                 agent_status: Dict, overall_risk_level: str, 
                                 overall_risk_score: int, user_prompt: str) -> str:
        """Generate concise story-style narrative"""
        
        # Section 1: What triggered this
        alert_name = alert.get('alertName', 'Unknown Alert')
        severity = alert.get('severity', 'unknown')
        trigger_line = f"Alert '{alert_name}' (severity: {severity}) was triggered and required investigation."
        
        # Section 2: What we looked at
        agent_friendly_names = {
            'user_context': 'user behavior',
            'ip_agent': 'IP behavior', 
            'timeline': 'timeline analysis',
            'alert_history': 'historical patterns',
            'summary': 'summary compilation'
        }
        
        analyzed_areas = []
        for agent in agent_status.keys():
            if agent != 'summary' and agent_status[agent]['status_label'] != 'N/A':
                friendly_name = agent_friendly_names.get(agent, agent)
                analyzed_areas.append(friendly_name)
        
        if analyzed_areas:
            looked_at_line = f"We examined {', '.join(analyzed_areas[:-1])} and {analyzed_areas[-1]}." if len(analyzed_areas) > 1 else f"We examined {analyzed_areas[0]}."
        else:
            looked_at_line = "We conducted a preliminary analysis."
        
        # Section 3: What we found
        findings_lines = []
        for agent, status_info in agent_status.items():
            if agent != 'summary' and status_info['status_label'] != 'N/A':
                friendly_name = agent_friendly_names.get(agent, agent).capitalize()
                status_label = status_info['status_label']
                reason = status_info['reason']
                findings_lines.append(f"{friendly_name} analysis returned {status_label}: {reason}")
        
        findings_section = ' '.join(findings_lines[:4])  # Limit to 4 key findings
        
        # Section 4: Why it matters
        if overall_risk_level in ['HIGH', 'CRITICAL']:
            matters_line = f"This represents a {overall_risk_level.lower()} risk situation that requires immediate attention and response actions."
        elif overall_risk_level == 'MODERATE':
            matters_line = f"This represents a moderate risk that warrants monitoring and potential containment measures."
        elif overall_risk_level == 'LOW':
            matters_line = f"This appears to be a low-risk situation but should be documented for trend analysis."
        else:
            matters_line = f"This appears to be normal activity with no immediate security concerns."
        
        # Section 5: Verdict & next steps
        verdict_line = f"Overall assessment: {overall_risk_level} risk ({overall_risk_score}/100)."
        
        # Generate recommendations
        recommendations = self._generate_recommendations(overall_risk_level, agent_results)
        next_steps = f"Recommended actions: {'; '.join(recommendations[:3])}."
        
        # Combine all sections
        narrative = f"""{trigger_line} {looked_at_line} {findings_section} {matters_line} {verdict_line} {next_steps}"""
        
        return narrative
    
    def _extract_highlights(self, agent_results: List[Dict], agent_status: Dict) -> List[str]:
        """Extract key highlights from analysis"""
        highlights = []
        
        # Extract key points from each agent
        for result in agent_results:
            agent_type = result.get('agent_type')
            if not agent_type or agent_type == 'summary':
                continue
                
            status_info = agent_status.get(agent_type, {})
            if status_info.get('status_label') in ['HIGH', 'CRITICAL']:
                highlights.append(f"{agent_type.replace('_', ' ').title()}: {status_info.get('reason', 'High risk detected')}")
        
        # Add overall assessment if significant
        if len(highlights) == 0:
            highlights.append("No significant security concerns identified")
            highlights.append("All analysis completed within normal parameters")
        
        return highlights[:4]  # Limit to top 4 highlights
    
    def _generate_recommendations(self, risk_level: str, agent_results: List[Dict]) -> List[str]:
        """Generate role-appropriate recommendations based on evidence"""
        recommendations = []
        
        if risk_level == 'CRITICAL':
            recommendations.extend([
                "Initiate incident response procedures immediately",
                "Isolate affected systems and accounts",
                "Preserve forensic evidence",
                "Notify security team and stakeholders"
            ])
        elif risk_level == 'HIGH':
            recommendations.extend([
                "Investigate and contain potential threat",
                "Monitor affected entities closely",
                "Review security controls and access logs",
                "Update threat intelligence feeds"
            ])
        elif risk_level == 'MODERATE':
            recommendations.extend([
                "Continue monitoring for pattern development", 
                "Review alert tuning parameters",
                "Document findings for trend analysis"
            ])
        elif risk_level == 'LOW':
            recommendations.extend([
                "Monitor situation for changes",
                "Document incident details",
                "Consider alert threshold adjustments"
            ])
        else:  # NORMAL
            recommendations.extend([
                "Continue routine monitoring",
                "Update detection rules as needed",
                "Document for baseline establishment"
            ])
        
        return recommendations

# Usage/Testing
if __name__ == "__main__":
    print("=== Summary Agent ===")
    
    # Sample test
    sample_alert = {
        'alertId': 'test-alert',
        'alertName': 'Failed Logon from Unknown Host',
        'severity': 'high'
    }
    
    sample_agent_results = [
        {
            'agent_type': 'user_context',
            'risk_level': 'LOW',
            'analysis': {'findings': ['No authentication activity found']}
        },
        {
            'agent_type': 'ip_agent', 
            'risk_level': 'NORMAL',
            'analysis': {'findings': ['No traffic patterns found']}
        }
    ]
    
    agent = SummaryAgent()
    result = agent.compile_analysis(sample_alert, sample_agent_results, "investigate ip behaviour")
    
    print(json.dumps(result, indent=2))
