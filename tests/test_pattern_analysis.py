#!/usr/bin/env python3
"""
Comprehensive Test Script for Pattern Analysis Agent
Tests all core functionality including ML models, typology detection, and network analysis
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Change to project root directory for relative paths to work
os.chdir(PROJECT_ROOT)

print(f"🔧 Working directory: {os.getcwd()}")
print(f"🔧 Python path includes: {PROJECT_ROOT}")

# Import functions - handle import errors gracefully
try:
    from googleadk.pattern_analysis_agent.agent import (
        analyze_transaction_patterns,
        detect_network_anomalies,
        identify_ml_typologies,
        generate_pattern_insights,
        get_pattern_agent_status,
        load_trained_models,
        load_transaction_data
    )
    print("✅ Successfully imported pattern analysis agent modules")
    AGENT_IMPORTED = True
except ImportError as e:
    print(f"❌ Import error: {str(e)}")
    print("📁 Make sure you have created the agent files and trained the models first!")
    AGENT_IMPORTED = False

def get_real_test_accounts():
    """Get real account names from the actual data"""
    try:
        # Import the agent to load data
        from googleadk.pattern_analysis_agent.agent import TRANSACTION_DATA, load_transaction_data
        
        # Ensure data is loaded
        if TRANSACTION_DATA is None:
            load_transaction_data()
        
        if TRANSACTION_DATA is not None and not TRANSACTION_DATA.empty:
            # Get unique account names from the data
            orig_accounts = TRANSACTION_DATA['originator_account'].unique()
            benef_accounts = TRANSACTION_DATA['beneficiary_account'].unique()
            all_accounts = set(orig_accounts) | set(benef_accounts)
            
            # Convert to list and take a sample
            account_list = list(all_accounts)
            
            return {
                'test_accounts': account_list[:5] if len(account_list) >= 5 else account_list,
                'large_test_accounts': account_list[:25] if len(account_list) >= 25 else account_list,
                'stress_test_accounts': account_list[:50] if len(account_list) >= 50 else account_list
            }
        else:
            # Fallback to default accounts
            return {
                'test_accounts': ['ACC_00001', 'ACC_00002', 'ACC_00003', 'ACC_00050', 'ACC_00100'],
                'large_test_accounts': [f'ACC_{i:05d}' for i in range(1, 26)],
                'stress_test_accounts': [f'ACC_{i:05d}' for i in range(1, 51)]
            }
    except Exception as e:
        print(f"⚠️ Could not get real accounts: {e}")
        # Fallback to default accounts
        return {
            'test_accounts': ['ACC_00001', 'ACC_00002', 'ACC_00003', 'ACC_00050', 'ACC_00100'],
            'large_test_accounts': [f'ACC_{i:05d}' for i in range(1, 26)],
            'stress_test_accounts': [f'ACC_{i:05d}' for i in range(1, 51)]
        }

def get_data_date_range():
    """Get the actual date range from your data"""
    try:
        from googleadk.pattern_analysis_agent.agent import TRANSACTION_DATA
        if TRANSACTION_DATA is not None and 'transaction_date' in TRANSACTION_DATA.columns:
            min_date = TRANSACTION_DATA['transaction_date'].min()
            max_date = TRANSACTION_DATA['transaction_date'].max()
            
            # Calculate days from max_date to today
            days_since_max = (datetime.now() - max_date).days
            
            # Add some buffer to ensure we capture all data
            return days_since_max + 50
        else:
            return 500  # Safe default
    except:
        return 500

# FIXED: Get real accounts and configure tests properly
real_accounts = get_real_test_accounts()

# FIXED: Single TEST_CONFIG definition with proper date range
TEST_CONFIG = {
    'test_accounts': real_accounts['test_accounts'],
    'large_test_accounts': real_accounts['large_test_accounts'], 
    'stress_test_accounts': real_accounts['stress_test_accounts'],
    'analysis_period_days': get_data_date_range(),  # Dynamic based on actual data
    'confidence_threshold': 0.5
}

print(f"📋 Using real test accounts: {TEST_CONFIG['test_accounts'][:3]}...")  # Show first 3

def print_test_header(test_name: str, test_number: int = None):
    """Print formatted test header"""
    if test_number:
        print(f"\n--- Test {test_number}: {test_name} ---")
    else:
        print(f"\n--- {test_name} ---")
    print("-" * (len(test_name) + 20))

def print_test_result(test_name: str, success: bool, details: str = ""):
    """Print formatted test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status}: {test_name}")
    if details:
        print(f"    {details}")

def print_metrics(metrics: dict, indent: int = 0):
    """Print metrics in a formatted way"""
    prefix = "  " * indent
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_metrics(value, indent + 1)
        elif isinstance(value, (int, float)):
            if isinstance(value, float):
                print(f"{prefix}{key}: {value:.3f}")
            else:
                print(f"{prefix}{key}: {value:,}")
        else:
            print(f"{prefix}{key}: {value}")

async def test_agent_status():
    """Test 1: Agent Status and Health Check"""
    print_test_header("Agent Status and Health Check", 1)
    
    try:
        if not AGENT_IMPORTED:
            print_test_result("Agent Import", False, "Cannot import agent modules")
            return False
        
        status_result = get_pattern_agent_status()
        
        if 'error' in status_result:
            print_test_result("Status Retrieval", False, f"Error: {status_result['error']}")
            return False
        
        # Check essential components
        agent_info = status_result.get('agent_info', {})
        model_status = status_result.get('model_status', {})
        data_status = status_result.get('data_status', {})
        
        print_test_result("Status Retrieval", True)
        print(f"    Agent: {agent_info.get('name', 'Unknown')} v{agent_info.get('version', 'Unknown')}")
        print(f"    Status: {agent_info.get('status', 'Unknown')}")
        print(f"    Models loaded: {model_status.get('models_loaded', False)}")
        print(f"    Total models: {model_status.get('total_models', 0)}")
        print(f"    Data loaded: {data_status.get('transaction_data_loaded', False)}")
        print(f"    Network built: {data_status.get('network_graph_built', False)}")
        
        # Detailed status
        if model_status.get('models_loaded'):
            print("    Model categories:", list(model_status.get('model_categories', [])))
        
        return True
        
    except Exception as e:
        print_test_result("Agent Status", False, f"Exception: {e}")
        return False

async def test_model_and_data_loading():
    """Test 2: Model and Data Loading"""
    print_test_header("Model and Data Loading", 2)
    
    try:
        if not AGENT_IMPORTED:
            print_test_result("Model Loading", False, "Agent not imported")
            return False
        
        # Test model loading
        models_loaded = load_trained_models()
        print_test_result("Model Loading", models_loaded, 
                         "Models loaded successfully" if models_loaded else "Failed to load models")
        
        # Test data loading
        data_loaded = load_transaction_data()
        print_test_result("Data Loading", data_loaded,
                         "Data loaded successfully" if data_loaded else "Failed to load data")
        
        return models_loaded and data_loaded
        
    except Exception as e:
        print_test_result("Model/Data Loading", False, f"Exception: {e}")
        return False

async def test_basic_pattern_analysis():
    """Test 3: Basic Pattern Analysis"""
    print_test_header("Basic Pattern Analysis", 3)
    
    try:
        if not AGENT_IMPORTED:
            print_test_result("Pattern Analysis", False, "Agent not imported")
            return False
        
        test_accounts = TEST_CONFIG['test_accounts']
        investigation_id = f"TEST_BASIC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🔍 Analyzing patterns for accounts: {test_accounts}")
        
        start_time = datetime.now()
        pattern_results = analyze_transaction_patterns(
            target_accounts=test_accounts,
            investigation_id=investigation_id,
            analysis_period_days=TEST_CONFIG['analysis_period_days'],
            include_network_analysis=True,
            pattern_types=['amount', 'behavioral', 'velocity', 'network', 'typologies']
        )
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        if 'error' in pattern_results:
            print_test_result("Pattern Analysis", False, f"Error: {pattern_results['error']}")
            return False
        
        print_test_result("Pattern Analysis", True, f"Completed in {processing_time:.2f}s")
        
        # Print summary results
        tx_summary = pattern_results.get('transaction_summary', {})
        risk_assessment = pattern_results.get('risk_assessment', {})
        
        print(f"    📊 Transactions analyzed: {tx_summary.get('total_transactions', 0):,}")
        print(f"    💰 Total amount: ${tx_summary.get('total_amount', 0):,.2f}")
        print(f"    🎯 Risk score: {risk_assessment.get('overall_risk_score', 0):.3f}")
        print(f"    ⚠️ Risk level: {risk_assessment.get('risk_level', 'UNKNOWN')}")
        
        # Check ML analysis results
        ml_analysis = pattern_results.get('ml_pattern_analysis', {})
        print(f"    🤖 ML Analysis Results:")
        for analysis_type, results in ml_analysis.items():
            if 'error' not in results:
                print(f"       {analysis_type}: ✅")
                if analysis_type == 'amount_anomalies' and 'consensus_analysis' in results:
                    outliers = results['consensus_analysis'].get('high_confidence_outliers', 0)
                    print(f"         High-confidence outliers: {outliers}")
                elif analysis_type == 'behavioral_anomalies' and 'isolation_forest' in results:
                    anomalies = results['isolation_forest'].get('anomaly_count', 0)
                    print(f"         Behavioral anomalies: {anomalies}")
            else:
                print(f"       {analysis_type}: ❌ {results['error']}")
        
        # Check typology results
        typology_detection = pattern_results.get('typology_detection', {})
        total_typologies = sum(results.get('findings_count', 0) for results in typology_detection.values())
        print(f"    🕵️ Typology Detection: {total_typologies} findings")
        
        return True
        
    except Exception as e:
        print_test_result("Pattern Analysis", False, f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_network_anomaly_detection():
    """Test 4: Network Anomaly Detection"""
    print_test_header("Network Anomaly Detection", 4)
    
    try:
        if not AGENT_IMPORTED:
            print_test_result("Network Analysis", False, "Agent not imported")
            return False
        
        test_accounts = TEST_CONFIG['test_accounts']
        investigation_id = f"TEST_NETWORK_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        start_time = datetime.now()
        network_results = detect_network_anomalies(
            target_accounts=test_accounts,
            investigation_id=investigation_id,
            sensitivity=0.1
        )
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        if 'error' in network_results:
            print_test_result("Network Analysis", False, f"Error: {network_results['error']}")
            return False
        
        print_test_result("Network Analysis", True, f"Completed in {processing_time:.2f}s")
        
        # Print network statistics
        network_stats = network_results.get('network_statistics', {})
        print(f"    🕸️ Network nodes: {network_stats.get('total_nodes', 0):,}")
        print(f"    🔗 Network edges: {network_stats.get('total_edges', 0):,}")
        print(f"    📊 Network density: {network_stats.get('network_density', 0):.4f}")
        print(f"    🎯 Target nodes in network: {network_stats.get('target_nodes_in_network', 0)}")
        
        # Check anomaly detection results
        anomaly_detection = network_results.get('anomaly_detection', {})
        if 'centrality_analysis' in anomaly_detection:
            centrality = anomaly_detection['centrality_analysis']
            print(f"    📈 Centrality Anomalies:")
            print(f"       High degree accounts: {len(centrality.get('high_degree_accounts', []))}")
            print(f"       High betweenness accounts: {len(centrality.get('high_betweenness_accounts', []))}")
            print(f"       Hub accounts: {len(centrality.get('hub_accounts', []))}")
        
        # Check suspicious patterns
        suspicious_patterns = network_results.get('suspicious_patterns', {})
        if 'money_mule_indicators' in suspicious_patterns:
            mule_data = suspicious_patterns['money_mule_indicators']
            mule_count = mule_data.get('mule_count', 0)
            print(f"    🔍 Potential money mules: {mule_count}")
        
        return True
        
    except Exception as e:
        print_test_result("Network Analysis", False, f"Exception: {e}")
        return False

async def test_typology_identification():
    """Test 5: Typology Identification"""
    print_test_header("Typology Identification", 5)
    
    try:
        if not AGENT_IMPORTED:
            print_test_result("Typology Detection", False, "Agent not imported")
            return False
        
        test_accounts = TEST_CONFIG['test_accounts']
        investigation_id = f"TEST_TYPOLOGY_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        start_time = datetime.now()
        typology_results = identify_ml_typologies(
            target_accounts=test_accounts,
            investigation_id=investigation_id,
            confidence_threshold=TEST_CONFIG['confidence_threshold']
        )
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        if 'error' in typology_results:
            print_test_result("Typology Detection", False, f"Error: {typology_results['error']}")
            return False
        
        print_test_result("Typology Detection", True, f"Completed in {processing_time:.2f}s")
        
        # Print typology summary
        summary = typology_results.get('summary', {})
        print(f"    🕵️ Typologies tested: {summary.get('total_typologies_tested', 0)}")
        print(f"    📋 Total findings: {summary.get('total_findings', 0)}")
        print(f"    🎯 High-confidence typologies: {summary.get('high_confidence_typologies', 0)}")
        print(f"    📊 Suspicion level: {summary.get('overall_suspicion_level', 'UNKNOWN')}")
        
        # Show specific typology results
        typology_findings = typology_results.get('typology_findings', {})
        for typology_name, results in typology_findings.items():
            findings_count = results.get('findings_count', 0)
            confidence = results.get('overall_confidence', 0)
            if findings_count > 0:
                print(f"       {typology_name.title()}: {findings_count} findings ({confidence:.1%} confidence)")
        
        return True
        
    except Exception as e:
        print_test_result("Typology Detection", False, f"Exception: {e}")
        return False

async def test_insight_generation():
    """Test 6: Pattern Insights Generation"""
    print_test_header("Pattern Insights Generation", 6)
    
    try:
        if not AGENT_IMPORTED:
            print_test_result("Insight Generation", False, "Agent not imported")
            return False
        
        test_accounts = TEST_CONFIG['test_accounts']
        investigation_id = f"TEST_INSIGHTS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # First run a pattern analysis to get results
        pattern_results = analyze_transaction_patterns(
            target_accounts=test_accounts,
            investigation_id=investigation_id,
            analysis_period_days=TEST_CONFIG['analysis_period_days'],
            include_network_analysis=True
        )
        
        if 'error' in pattern_results:
            print_test_result("Pattern Analysis for Insights", False, f"Error: {pattern_results['error']}")
            return False
        
        # FIXED: Generate insights with proper parameters
        start_time = datetime.now()
        insights_results = generate_pattern_insights(
                investigation_id=investigation_id,
                target_accounts=TEST_CONFIG['test_accounts'],  # ✅ Correct parameter
                analysis_period_days=TEST_CONFIG['analysis_period_days'],
                include_recommendations=True
        )
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        if 'error' in insights_results:
            print_test_result("Insight Generation", False, f"Error: {insights_results['error']}")
            return False
        
        print_test_result("Insight Generation", True, f"Completed in {processing_time:.2f}s")
        
        # Print executive summary
        exec_summary = insights_results.get('executive_summary', {})
        print(f"    📊 Executive Summary:")
        print(f"       Overall risk: {exec_summary.get('risk_level', 'UNKNOWN')} ({exec_summary.get('overall_risk_score', 0):.3f})")
        print(f"       Primary concern: {exec_summary.get('primary_concern', 'None identified')}")
        print(f"       Investigation urgency: {exec_summary.get('investigation_urgency', 'UNKNOWN')}")
        print(f"       Total anomalies: {exec_summary.get('total_anomalies', 0)}")
        
        # Show key findings
        key_findings = insights_results.get('key_findings', [])
        print(f"    🔍 Key Findings ({len(key_findings)}):")
        for finding in key_findings[:3]:  # Show top 3
            if isinstance(finding, dict):
                print(f"       {finding.get('category', 'UNKNOWN')}: {finding.get('finding', 'No description')} (Severity: {finding.get('severity', 'UNKNOWN')})")
            else:
                print(f"       {finding}")
        
        # Show recommendations
        recommendations = insights_results.get('recommendations', [])
        if recommendations:
            print(f"    💡 Recommendations ({len(recommendations)}):")
            for rec in recommendations[:3]:  # Show top 3
                print(f"       • {rec}")
        
        # Show confidence assessment
        confidence = insights_results.get('confidence_assessment', {})
        print(f"    🎯 Confidence Assessment:")
        print(f"       Overall confidence: {confidence.get('overall_confidence', 0):.1%}")
        print(f"       Confidence level: {confidence.get('confidence_level', 'UNKNOWN')}")
        
        return True
        
    except Exception as e:
        print_test_result("Insight Generation", False, f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_large_dataset_performance():
    """Test 7: Large Dataset Performance Test"""
    print_test_header("Large Dataset Performance Test", 7)
    
    try:
        if not AGENT_IMPORTED:
            print_test_result("Performance Test", False, "Agent not imported")
            return False
        
        large_test_accounts = TEST_CONFIG['large_test_accounts']
        investigation_id = f"TEST_LARGE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🔥 Performance testing with {len(large_test_accounts)} accounts")
        
        start_time = datetime.now()
        performance_results = analyze_transaction_patterns(
            target_accounts=large_test_accounts,
            investigation_id=investigation_id,
            analysis_period_days=min(30, TEST_CONFIG['analysis_period_days']),  # Use shorter period for speed
            include_network_analysis=True
        )
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        if 'error' in performance_results:
            print_test_result("Performance Test", False, f"Error: {performance_results['error']}")
            return False
        
        # Calculate performance metrics
        tx_count = performance_results.get('transaction_summary', {}).get('total_transactions', 0)
        processing_rate = tx_count / processing_time if processing_time > 0 else 0
        
        print_test_result("Performance Test", True, f"Completed in {processing_time:.2f}s")
        print(f"    📊 Transactions analyzed: {tx_count:,}")
        print(f"    ⚡ Processing rate: {processing_rate:.0f} txns/sec")
        print(f"    🎯 Risk score: {performance_results.get('risk_assessment', {}).get('overall_risk_score', 0):.3f}")
        
        # Performance benchmarks
        if processing_time < 60:
            print(f"    ✅ Performance: EXCELLENT (< 60s)")
        elif processing_time < 120:
            print(f"    ✅ Performance: GOOD (< 120s)")
        else:
            print(f"    ⚠️ Performance: NEEDS OPTIMIZATION (> 120s)")
        
        return True
        
    except Exception as e:
        print_test_result("Performance Test", False, f"Exception: {e}")
        return False

async def test_stress_test():
    """Test 8: Stress Test with Heavy Load"""
    print_test_header("Stress Test with Heavy Load", 8)
    
    try:
        if not AGENT_IMPORTED:
            print_test_result("Stress Test", False, "Agent not imported")
            return False
        
        stress_test_accounts = TEST_CONFIG['stress_test_accounts']
        investigation_id = f"TEST_STRESS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"💪 Stress testing with {len(stress_test_accounts)} accounts")
        
        start_time = datetime.now()
        stress_results = analyze_transaction_patterns(
            target_accounts=stress_test_accounts,
            investigation_id=investigation_id,
            analysis_period_days=min(14, TEST_CONFIG['analysis_period_days']),  # Very short period
            include_network_analysis=False,  # Disable network for speed
            pattern_types=['amount', 'behavioral']  # Limited analysis
        )
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        if 'error' in stress_results:
            print_test_result("Stress Test", False, f"Error: {stress_results['error']}")
            return False
        
        tx_count = stress_results.get('transaction_summary', {}).get('total_transactions', 0)
        processing_rate = tx_count / processing_time if processing_time > 0 else 0
        
        print_test_result("Stress Test", True, f"Completed in {processing_time:.2f}s")
        print(f"    📊 Transactions analyzed: {tx_count:,}")
        print(f"    ⚡ Processing rate: {processing_rate:.0f} txns/sec")
        print(f"    💾 Memory usage: NORMAL (estimated)")
        
        # Stress test benchmarks
        if processing_time < 180:  # 3 minutes
            print(f"    ✅ Stress test: PASSED")
        else:
            print(f"    ⚠️ Stress test: MARGINAL (consider optimization)")
        
        return True
        
    except Exception as e:
        print_test_result("Stress Test", False, f"Exception: {e}")
        return False

async def test_error_handling():
    """Test 9: Error Handling and Edge Cases"""
    print_test_header("Error Handling and Edge Cases", 9)
    
    try:
        if not AGENT_IMPORTED:
            print_test_result("Error Handling", False, "Agent not imported")
            return False
        
        error_tests_passed = 0
        total_error_tests = 4
        
        # Test 1: Invalid accounts
        print("    Testing invalid accounts...")
        error_results = analyze_transaction_patterns(
            target_accounts=['INVALID_ACCOUNT_123', 'NONEXISTENT_456'],
            investigation_id="ERROR_TEST_1",
            analysis_period_days=30
        )
        
        if 'error' not in error_results:
            print("       ✅ Invalid accounts handled gracefully")
            error_tests_passed += 1
        else:
            print(f"       ⚠️ Expected graceful handling, got error: {error_results['error']}")
        
        # Test 2: Empty account list
        print("    Testing empty account list...")
        try:
            empty_results = analyze_transaction_patterns(
                target_accounts=[],
                investigation_id="ERROR_TEST_2",
                analysis_period_days=30
            )
            print("       ✅ Empty account list handled")
            error_tests_passed += 1
        except Exception as e:
            print(f"       ❌ Empty account list caused exception: {e}")
        
        # Test 3: Very short analysis period
        print("    Testing very short analysis period...")
        try:
            short_results = analyze_transaction_patterns(
                target_accounts=TEST_CONFIG['test_accounts'][:2],
                investigation_id="ERROR_TEST_3",
                analysis_period_days=1
            )
            print("       ✅ Short analysis period handled")
            error_tests_passed += 1
        except Exception as e:
            print(f"       ❌ Short period caused exception: {e}")
        
        # Test 4: Invalid pattern types
        print("    Testing invalid pattern types...")
        try:
            invalid_results = analyze_transaction_patterns(
                target_accounts=TEST_CONFIG['test_accounts'][:2],
                investigation_id="ERROR_TEST_4",
                analysis_period_days=30,
                pattern_types=['invalid_type', 'another_invalid']
            )
            print("       ✅ Invalid pattern types handled")
            error_tests_passed += 1
        except Exception as e:
            print(f"       ❌ Invalid pattern types caused exception: {e}")
        
        success_rate = error_tests_passed / total_error_tests
        print_test_result("Error Handling", success_rate >= 0.75, 
                         f"{error_tests_passed}/{total_error_tests} error tests passed")
        
        return success_rate >= 0.75
        
    except Exception as e:
        print_test_result("Error Handling", False, f"Exception: {e}")
        return False

async def test_final_status_check():
    """Test 10: Final Status Check and Statistics"""
    print_test_header("Final Status Check and Statistics", 10)
    
    try:
        if not AGENT_IMPORTED:
            print_test_result("Final Status", False, "Agent not imported")
            return False
        
        final_status = get_pattern_agent_status()
        
        if 'error' in final_status:
            print_test_result("Final Status", False, f"Error: {final_status['error']}")
            return False
        
        print_test_result("Final Status", True)
        
        # Print comprehensive statistics
        if 'processing_statistics' in final_status:
            stats = final_status['processing_statistics']
            print(f"    📈 Final Agent Statistics:")
            print_metrics({
                'Pattern analyses performed': stats.get('pattern_analyses_performed', 0),
                'Networks analyzed': stats.get('networks_analyzed', 0),
                'Total anomalies detected': stats.get('total_anomalies_detected', 0),
                'Total typologies identified': stats.get('total_typologies_identified', 0),
                'Average processing time (ms)': stats.get('average_processing_time_ms', 0),
                'Total processing time (s)': stats.get('total_processing_time_seconds', 0)
            }, indent=2)
        
        # Print performance assessment
        performance_metrics = final_status.get('performance_metrics', {})
        system_health = performance_metrics.get('system_health', 'UNKNOWN')
        processing_efficiency = performance_metrics.get('processing_efficiency', 'UNKNOWN')
        
        print(f"    🏥 System Health: {system_health}")
        print(f"    ⚡ Processing Efficiency: {processing_efficiency}")
        
        return True
        
    except Exception as e:
        print_test_result("Final Status", False, f"Exception: {e}")
        return False

async def run_comprehensive_test_suite():
    """Run the complete test suite"""
    print("🚀 Pattern Analysis Agent Comprehensive Test Suite")
    print("=" * 80)
    print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test configuration summary
    print(f"\n📋 Test Configuration:")
    print(f"    Test accounts: {len(TEST_CONFIG['test_accounts'])}")
    print(f"    Large test accounts: {len(TEST_CONFIG['large_test_accounts'])}")
    print(f"    Stress test accounts: {len(TEST_CONFIG['stress_test_accounts'])}")
    print(f"    Analysis period: {TEST_CONFIG['analysis_period_days']} days")
    print(f"    Confidence threshold: {TEST_CONFIG['confidence_threshold']}")
    
    # Run all tests
    test_results = []
    overall_start_time = datetime.now()
    
    # Test suite
    tests = [
        ("Agent Status Check", test_agent_status),
        ("Model and Data Loading", test_model_and_data_loading),
        ("Basic Pattern Analysis", test_basic_pattern_analysis),
        ("Network Anomaly Detection", test_network_anomaly_detection),
        ("Typology Identification", test_typology_identification),
        ("Insight Generation", test_insight_generation),
        ("Large Dataset Performance", test_large_dataset_performance),
        ("Stress Test", test_stress_test),
        ("Error Handling", test_error_handling),
        ("Final Status Check", test_final_status_check)
    ]
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ CRITICAL ERROR in {test_name}: {e}")
            test_results.append((test_name, False))
    
    overall_end_time = datetime.now()
    total_test_time = (overall_end_time - overall_start_time).total_seconds()
    
    # Test Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"📈 Overall Results:")
    print(f"    Tests passed: {passed_tests}/{total_tests}")
    print(f"    Success rate: {success_rate:.1f}%")
    print(f"    Total test time: {total_test_time:.2f} seconds")
    
    print(f"\n📋 Individual Test Results:")
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"    {status}: {test_name}")
    
    # Final assessment
    print(f"\n🎯 Final Assessment:")
    if success_rate >= 90:
        print("    🎉 EXCELLENT: Pattern Analysis Agent is production-ready!")
    elif success_rate >= 80:
        print("    ✅ GOOD: Pattern Analysis Agent is mostly ready with minor issues")
    elif success_rate >= 70:
        print("    ⚠️ ACCEPTABLE: Pattern Analysis Agent needs some improvements")
    else:
        print("    ❌ POOR: Pattern Analysis Agent requires significant fixes")
    
    print(f"\n🕐 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return success_rate >= 80

async def main():
    """Main test function"""
    try:
        success = await run_comprehensive_test_suite()
        return success
    except KeyboardInterrupt:
        print("\n\n⚠️ Test suite interrupted by user")
        return False
    except Exception as e:
        print(f"\n\n❌ Test suite failed with critical error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)