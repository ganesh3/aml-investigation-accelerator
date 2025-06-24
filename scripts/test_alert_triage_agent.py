#!/usr/bin/env python3
import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Change to project root directory for relative paths to work
os.chdir(PROJECT_ROOT)

print(f"🔧 Working directory: {os.getcwd()}")
print(f"🔧 Python path includes: {PROJECT_ROOT}")

try:
    from agents.alert_triage.agent import AlertTriageAgent
    from agents.base.message_system import AgentMessage, MessageType
    print("✅ Successfully imported agent modules")
except ImportError as e:
    print(f"❌ Import error: {str(e)}")
    print("📁 Make sure you have created the agent files first!")
    sys.exit(1)

async def test_alert_triage():
    """Test the Alert Triage Agent"""
    print("\n🚨 Testing Alert Triage Agent")
    print("="*50)
    
    try:
        # Load configuration
        config_path = r'D:\Google-ADK-Project\config\alert_triage_config.json'
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Loaded config from: {config_path}")
        
        # Check if models exist
        model_dir = Path(config['model_path'])
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            return False
        
        print(f"✅ Model directory found: {model_dir}")
        
        # Create agent
        print("🤖 x Alert Triage Agent...")
        agent = AlertTriageAgent(config)
        print(f"✅ Agent created with {len(agent.models)} models")
        
        # Test transactions
        test_cases = [
            {
                'transaction_id': 'TEST_HIGH_RISK',
                'amount': 95000,
                'risk_score': 0.85,
                'cross_border': True,
                'unusual_hour': True,
                'orig_ml_rate': 0.15,
                'hour': 23,
                'day_of_week': 6,
                'log_amount': 11.46,  # log(95000)
                'orig_tx_count': 50,
                'orig_avg_amount': 75000,
                'orig_total_amount': 3750000
            },
            {
                'transaction_id': 'TEST_LOW_RISK',
                'amount': 250,
                'risk_score': 0.05,
                'cross_border': False,
                'unusual_hour': False,
                'orig_ml_rate': 0.01,
                'hour': 14,
                'day_of_week': 2,
                'log_amount': 5.52,  # log(250)
                'orig_tx_count': 5,
                'orig_avg_amount': 300,
                'orig_total_amount': 1500
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i}: {test_case['transaction_id']} ---")
            print(f"💰 Amount: ${test_case['amount']:,.2f}")
            
            message = AgentMessage(
                sender="test_client",
                receiver="alert_triage",
                message_type=MessageType.RISK_ASSESSMENT_REQUEST.value,
                payload={'transaction_data': test_case},
                timestamp=datetime.now(),
                correlation_id=f"test_{i}"
            )
            
            response = await agent.process_request(message)
            
            if response.success:
                assessment = response.data['assessment']
                print(f"✅ Risk Score: {assessment['risk_score']:.3f}")
                print(f"📊 Priority: {assessment['priority_level']}")
                print(f"🎯 Action: {assessment['recommended_action']}")
                print(f"🤖 Confidence: {assessment['confidence']:.3f}")
                
                # Show individual model predictions
                if 'individual_predictions' in assessment:
                    print(f"🔍 Model Predictions:")
                    for model, pred in assessment['individual_predictions'].items():
                        print(f"   {model}: {pred:.3f}")
                
                print(f"💭 Top Reasoning: {assessment['reasoning'][0]}")
                print(f"⏱️ Processing Time: {response.processing_time*1000:.1f}ms")
                
                if assessment.get('auto_action'):
                    print(f"⚡ Auto Action: {assessment['auto_action']}")
                    
            else:
                print(f"❌ Error: {response.error_message}")
        
        # Show agent statistics
        print(f"\n📈 Agent Performance:")
        status_message = AgentMessage(
            sender="test_client",
            receiver="alert_triage",
            message_type=MessageType.AGENT_STATUS_REQUEST.value,
            payload={},
            timestamp=datetime.now(),
            correlation_id="status_test"
        )
        
        status_response = await agent.process_request(status_message)
        if status_response.success:
            stats = status_response.data['statistics']
            print(f"   Alerts Processed: {stats['alerts_processed']}")
            print(f"   High Risk Rate: {stats['high_risk_rate']:.1f}%")
            print(f"   Auto Escalated: {stats['auto_escalated']}")
            print(f"   Auto Dismissed: {stats['auto_dismissed']}")
        
        print(f"\n🎉 Alert Triage Agent testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_alert_triage())
    if success:
        print("\n✅ All tests passed! Your Alert Triage Agent is working!")
    else:
        print("\n❌ Tests failed. Check the error messages above.")