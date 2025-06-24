#!/usr/bin/env python3
"""
Test script for Google ADK AML Alert Triage Agent
Validates ML model integration and agent functionality
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"🔧 Project root: {PROJECT_ROOT}")
print(f"🔧 Working directory: {os.getcwd()}")

# Test data for validation
TEST_TRANSACTIONS = [
    {
        "transaction_id": "TXN_HIGH_RISK_001",
        "amount": 95000.0,
        "cross_border": True,
        "unusual_hour": True,
        "originator_account": "ACC_001",
        "beneficiary_account": "ACC_002",
        "hour": 23,
        "day_of_week": 6,
        "orig_tx_count": 50,
        "orig_ml_rate": 0.15
    },
    {
        "transaction_id": "TXN_LOW_RISK_002", 
        "amount": 250.0,
        "cross_border": False,
        "unusual_hour": False,
        "originator_account": "ACC_003",
        "beneficiary_account": "ACC_004",
        "hour": 14,
        "day_of_week": 2,
        "orig_tx_count": 5,
        "orig_ml_rate": 0.01
    },
    {
        "transaction_id": "TXN_CRITICAL_003",
        "amount": 150000.0,
        "cross_border": True,
        "unusual_hour": True,
        "originator_account": "ACC_005",
        "beneficiary_account": "ACC_006",
        "hour": 2,
        "day_of_week": 0,
        "orig_tx_count": 100,
        "orig_ml_rate": 0.25
    }
]

async def test_adk_alert_triage():
    """Test the Google ADK Alert Triage Agent"""
    
    print("\n" + "="*70)
    print("🤖 TESTING GOOGLE ADK ALERT TRIAGE AGENT")
    print("="*70)
    
    try:
        # Import the ADK agent
        from googleadk.alert_triage_agent.agent import alert_triage_agent
        print("✅ Successfully imported Google ADK Alert Triage Agent")
        
        # Test 1: Agent Status Check using ADK CLI interface
        print(f"\n📊 TEST 1: Agent Status Check")
        print("-" * 50)
        
        try:
            # Use the ADK CLI approach
            import subprocess
            import sys
            import os
            
            # Change to the agent directory
            agent_dir = "googleadk/alert_triage_agent"
            original_dir = os.getcwd()
            
            # Test using direct tool call instead of ADK run
            from googleadk.alert_triage_agent.agent import get_agent_status
            status_result = get_agent_status()
            
            print("Agent Status:")
            print(f"   Name: {status_result.get('agent_info', {}).get('name', 'Unknown')}")
            print(f"   Status: {status_result.get('agent_info', {}).get('status', 'Unknown')}")
            print(f"   Models: {status_result.get('model_status', {}).get('models_loaded', [])}")
            print(f"   Alerts Processed: {status_result.get('processing_statistics', {}).get('alerts_processed', 0)}")
            
        except Exception as status_error:
            print(f"Status test failed: {status_error}")
        
        # Test 2: Individual Transaction Assessment via Tools
        print(f"\n🎯 TEST 2: Individual Transaction Risk Assessment via Tools")
        print("-" * 50)
        
        from googleadk.alert_triage_agent.agent import assess_transaction_risk
        
        for i, transaction in enumerate(TEST_TRANSACTIONS, 1):
            print(f"\n--- Test Case {i}: {transaction['transaction_id']} ---")
            print(f"💰 Amount: ${transaction['amount']:,.2f}")
            print(f"🌍 Cross-border: {transaction['cross_border']}")
            print(f"⏰ Unusual hour: {transaction['unusual_hour']}")
            
            # Direct tool call
            result = assess_transaction_risk(**transaction)
            
            print(f"🤖 Assessment Results:")
            print(f"   Risk Score: {result.get('risk_score', 0):.3f}")
            print(f"   Priority: {result.get('priority_level', 'Unknown')}")
            print(f"   Action: {result.get('recommended_action', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            
            if result.get('reasoning'):
                print(f"   Reasoning: {result['reasoning'][0]}")
        
        # Test 3: Batch Processing
        print(f"\n📦 TEST 3: Batch Transaction Processing via Tools")
        print("-" * 50)
        
        from googleadk.alert_triage_agent.agent import batch_assess_transactions
        
        batch_result = batch_assess_transactions(TEST_TRANSACTIONS)
        
        if 'error' not in batch_result:
            summary = batch_result.get('batch_summary', {})
            print(f"🤖 Batch Processing Results:")
            print(f"   Total Processed: {summary.get('batch_info', {}).get('total_transactions', 0)}")
            print(f"   High Risk: {summary.get('risk_distribution', {}).get('high_count', 0)}")
            print(f"   Critical: {summary.get('risk_distribution', {}).get('critical_count', 0)}")
            print(f"   Auto Escalated: {summary.get('actions_summary', {}).get('auto_escalated', 0)}")
            print(f"   Average Risk: {summary.get('risk_statistics', {}).get('average_risk_score', 0):.3f}")
        else:
            print(f"   ❌ Batch processing error: {batch_result['error']}")
        
        # Test 4: ADK Agent Integration (if possible)
        print(f"\n📈 TEST 4: Google ADK Agent Integration")
        print("-" * 50)
        
        try:
            # Try to use ADK agent properly
            print(f"   Agent Type: {type(alert_triage_agent)}")
            print(f"   Agent Name: {alert_triage_agent.name}")
            print(f"   Agent Description Length: {len(alert_triage_agent.description)} chars")
            print(f"   Tools Available: {len(alert_triage_agent.tools)}")
            
            # List tools
            for tool in alert_triage_agent.tools:
                if hasattr(tool, '__name__'):
                    print(f"     - {tool.__name__}")
                elif hasattr(tool, 'function') and hasattr(tool.function, '__name__'):
                    print(f"     - {tool.function.__name__}")
            
            print("   ✅ ADK Agent structure looks correct")
            
        except Exception as adk_error:
            print(f"   ❌ ADK agent integration error: {adk_error}")
        
        # Test 5: CLI Testing Information
        print(f"\n🧠 TEST 5: ADK CLI Testing Instructions")
        print("-" * 50)
        
        print("   To test the full ADK agent interactively:")
        print("   1. Open terminal in project root")
        print("   2. Run: adk run googleadk/alert_triage_agent")
        print("   3. Or: adk ui (for web interface)")
        print("   4. Then ask: 'Assess risk for a $95,000 cross-border transaction'")
        
        print(f"\n🎉 GOOGLE ADK AGENT STRUCTURE VALIDATION COMPLETED!")
        print("="*70)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("📝 Make sure you have:")
        print("   1. Installed google-adk: pip install google-adk")
        print("   2. Created the agent.py file with proper content")
        print("   3. Set up your .env file with API keys")
        return False
        
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_direct_tools():
    """Test the underlying tools directly (bypasses ADK agent)"""
    
    print(f"\n🔧 DIRECT TOOL TESTING (Bypass ADK)")
    print("-" * 50)
    
    try:
        # Import tools directly
        from googleadk.alert_triage_agent.agent import (
            assess_transaction_risk, 
            get_agent_status, 
            batch_assess_transactions
        )
        
        print("✅ Successfully imported direct tools")
        
        # Test direct tool calls
        print(f"\n🎯 Testing assess_transaction_risk tool directly:")
        
        test_tx = TEST_TRANSACTIONS[0]
        result = assess_transaction_risk(**test_tx)
        
        print(f"   Transaction: {result.get('transaction_id')}")
        print(f"   Risk Score: {result.get('risk_score', 0):.3f}")
        print(f"   Priority: {result.get('priority_level', 'Unknown')}")
        print(f"   Confidence: {result.get('confidence', 0):.3f}")
        print(f"   Action: {result.get('recommended_action', 'Unknown')}")
        
        if result.get('error'):
            print(f"   ⚠️ Error: {result['error']}")
        
        # Test status tool
        print(f"\n📊 Testing get_agent_status tool directly:")
        status = get_agent_status()
        print(f"   Agent: {status.get('agent_info', {}).get('name', 'Unknown')}")
        print(f"   Models Loaded: {status.get('model_status', {}).get('models_loaded', [])}")
        print(f"   Status: {status.get('agent_info', {}).get('status', 'Unknown')}")
        
        # Test batch tool
        print(f"\n📦 Testing batch_assess_transactions tool directly:")
        batch_result = batch_assess_transactions(TEST_TRANSACTIONS)
        
        if 'error' in batch_result:
            print(f"   ⚠️ Batch Error: {batch_result['error']}")
        else:
            summary = batch_result.get('batch_summary', {})
            print(f"   Processed: {summary.get('batch_info', {}).get('total_transactions', 0)} transactions")
            print(f"   High Risk: {summary.get('risk_distribution', {}).get('high_count', 0)}")
            print(f"   Critical: {summary.get('risk_distribution', {}).get('critical_count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct tool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def validate_model_loading():
    """Validate that ML models are properly loaded"""
    
    print(f"\n🤖 MODEL VALIDATION")
    print("-" * 50)
    
    try:
        from googleadk.alert_triage_agent.agent import MODELS, SCALER, FEATURE_COLUMNS
        
        print(f"   Models loaded: {len(MODELS)}")
        print(f"   Model names: {list(MODELS.keys())}")
        print(f"   Scaler loaded: {SCALER is not None}")
        print(f"   Feature count: {len(FEATURE_COLUMNS)}")
        
        if FEATURE_COLUMNS:
            print(f"   Sample features: {FEATURE_COLUMNS[:5]}...")
        
        # Test model prediction if available
        if MODELS and SCALER is not None:
            print(f"   ✅ Models and scaler ready for predictions")
        else:
            print(f"   ⚠️ Models or scaler not loaded - check model path")
            
        return len(MODELS) > 0
        
    except Exception as e:
        print(f"   ❌ Model validation failed: {e}")
        return False

def check_environment():
    """Check environment setup"""
    
    print(f"\n🌍 ENVIRONMENT CHECK")
    print("-" * 50)
    
    # Check for .env file
    env_file = Path("googleadk/alert_triage_agent/.env")
    print(f"   .env file exists: {env_file.exists()}")
    
    # Load the specific .env file to check API key
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
    
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY', '').strip().strip('"\'')
    has_api_key = bool(api_key and api_key != 'your-actual-api-key-here')
    print(f"   Google API key configured: {has_api_key}")
    if has_api_key:
        print(f"   API key length: {len(api_key)} characters")
    
    # Check model path
    model_path = Path("models")  # Changed from scripts/models
    print(f"   Model directory exists: {model_path.exists()}")
    
    if model_path.exists():
        model_files = list(model_path.glob("*.pkl"))
        print(f"   Model files found: {len(model_files)}")
        if model_files:
            print(f"   Model files: {[f.name for f in model_files[:3]]}...")  # Show first 3
        
    # Check required packages
    try:
        import google.adk
        print(f"   Google ADK installed: ✅")
    except ImportError:
        print(f"   Google ADK installed: ❌")
        
    return has_api_key and model_path.exists()

async def main():
    """Main test function"""
    
    print("🚀 GOOGLE ADK AML ALERT TRIAGE AGENT TESTING")
    print("=" * 70)
    
    # Environment check
    env_ok = check_environment()
    
    # Model validation
    models_ok = await validate_model_loading()
    
    # Direct tool testing
    tools_ok = await test_direct_tools()
    
    # Full ADK agent testing (if environment is ready)
    if env_ok:
        adk_ok = await test_adk_alert_triage()
    else:
        print(f"\n⚠️ Skipping ADK agent test - environment not ready")
        adk_ok = False
    
    # Summary
    print(f"\n📋 TEST SUMMARY")
    print("-" * 30)
    print(f"   Environment: {'✅' if env_ok else '❌'}")
    print(f"   Models: {'✅' if models_ok else '❌'}")
    print(f"   Direct Tools: {'✅' if tools_ok else '❌'}")
    print(f"   ADK Agent: {'✅' if adk_ok else '❌'}")
    
    if all([env_ok, models_ok, tools_ok, adk_ok]):
        print(f"\n🎉 ALL TESTS PASSED! Your Google ADK Agent is ready!")
    else:
        print(f"\n⚠️ Some tests failed. Check the errors above.")
        
        if not env_ok:
            print(f"   👉 Set up your .env file with Google API key")
        if not models_ok:
            print(f"   👉 Ensure your ML models are in /models/")
        if not tools_ok:
            print(f"   👉 Check the agent.py implementation")

if __name__ == "__main__":
    asyncio.run(main())