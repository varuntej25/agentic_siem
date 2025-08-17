# manual_test.py
from master_agent import MasterAgent

def manual_test():
    print("=== Manual Testing ===")
    
    # Test with your actual files
    master = MasterAgent()
    
    alerts_file = input("Enter alerts file path: ")
    alert_number = int(input("Enter alert number: "))
    user_prompt = input("Enter test prompt: ")
    
    # Run orchestration
    result = master.orchestrate_investigation(alerts_file, alert_number, user_prompt)
    
    print(f"\nRequired Agents: {result['intent_analysis']['required_agents']}")
    print("\nAgent Prompts:")
    for agent, prompt in result['agent_prompts'].items():
        print(f"{agent}: {prompt}")

if __name__ == '__main__':
    manual_test()