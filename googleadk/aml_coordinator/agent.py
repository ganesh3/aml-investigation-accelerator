#!/usr/bin/env python3
"""
AML Coordinator Agent - Google ADK Implementation (Complete 4-Agent System)
Orchestrates the complete AML investigation multi-agent system including Pattern Analysis
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import logging

# Google ADK imports
from google.adk.agents import LlmAgent

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path for importing other agents
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Import Sub-Agents (Updated for 4-Agent System)
# =============================================================================

def import_sub_agents():
    """Import all sub-agents for the coordinator"""
    try:
        # Import Alert Triage Agent
        from googleadk.alert_triage_agent.agent import (
            assess_transaction_risk,
            get_agent_status as get_alert_status,
            batch_assess_transactions
        )
        
        # Import Evidence Collection Agent  
        from googleadk.evidence_collection_agent.agent import (
            collect_transaction_evidence,
            screen_sanctions_watchlists,
            compile_evidence_package,
            get_evidence_agent_status
        )
        
        # Import Pattern Analysis Agent (NEW)
        from googleadk.pattern_analysis_agent.agent import (
            analyze_transaction_patterns,
            detect_network_anomalies,
            identify_ml_typologies,
            generate_pattern_insights,
            get_pattern_agent_status
        )
        
        logger.info("✅ Successfully imported all sub-agent tools")
        
        return {
            'alert_triage': {
                'assess_transaction_risk': assess_transaction_risk,
                'get_status': get_alert_status,
                'batch_assess': batch_assess_transactions
            },
            'evidence_collection': {
                'collect_transaction_evidence': collect_transaction_evidence,
                'screen_sanctions': screen_sanctions_watchlists,
                'compile_evidence': compile_evidence_package,
                'get_status': get_evidence_agent_status
            },
            'pattern_analysis': {
                'analyze_transaction_patterns': analyze_transaction_patterns,
                'detect_network_anomalies': detect_network_anomalies,
                'identify_ml_typologies': identify_ml_typologies,
                'generate_pattern_insights': generate_pattern_insights,
                'get_status': get_pattern_agent_status
            }
        }
        
    except ImportError as e:
        logger.error(f"Failed to import sub-agents: {e}")
        return {}

# Load sub-agent tools
SUB_AGENT_TOOLS = import_sub_agents()

# =============================================================================
# Enhanced Coordinator Tool Functions
# =============================================================================

def conduct_full_aml_investigation(
    transaction_data: Dict[str, Any],
    investigation_priority: str = "MEDIUM",
    include_evidence_collection: bool = True,
    include_sanctions_screening: bool = True,
    include_pattern_analysis: bool = True,
    include_ml_typology_detection: bool = True
) -> Dict[str, Any]:
    """
    Conduct a complete AML investigation using all available agents including Pattern Analysis.
    
    This orchestrates the full investigation workflow:
    1. Risk assessment (Alert Triage Agent)
    2. Evidence collection (Evidence Collection Agent)  
    3. Sanctions screening (Evidence Collection Agent)
    4. Advanced pattern analysis (Pattern Analysis Agent) - NEW
    5. ML typology detection (Pattern Analysis Agent) - NEW
    6. Comprehensive analysis and reporting
    
    Args:
        transaction_data: Transaction details for investigation
        investigation_priority: Priority level (LOW, MEDIUM, HIGH, CRITICAL)
        include_evidence_collection: Whether to collect transaction evidence
        include_sanctions_screening: Whether to screen sanctions lists
        include_pattern_analysis: Whether to perform advanced pattern analysis
        include_ml_typology_detection: Whether to detect ML typologies
        
    Returns:
        Dict containing complete investigation results with pattern analysis
    """
    
    investigation_id = f"INV_{hash(str(transaction_data))}"[-8:]
    start_time = datetime.now()
    
    logger.info(f"🚀 Starting full AML investigation: {investigation_id}")
    logger.info(f"Priority: {investigation_priority}")
    logger.info(f"Pattern Analysis: {'Enabled' if include_pattern_analysis else 'Disabled'}")
    
    investigation_results = {
        'investigation_id': investigation_id,
        'start_time': start_time.isoformat(),
        'priority_level': investigation_priority,
        'transaction_data': transaction_data,
        'workflow_steps': [],
        'results': {},
        'pattern_analysis_results': {},  # NEW: Dedicated section for pattern analysis
        'overall_assessment': {},
        'recommended_actions': []
    }
    
    try:
        # Step 1: Alert Triage and Risk Assessment
        logger.info("📊 Step 1: Conducting risk assessment...")
        investigation_results['workflow_steps'].append("risk_assessment")
        
        risk_score = 0  # Initialize risk_score
        
        if 'alert_triage' in SUB_AGENT_TOOLS:
            if 'transaction_id' not in transaction_data:
                transaction_data['transaction_id'] = f"TXN_{investigation_id}"
            
            risk_assessment = SUB_AGENT_TOOLS['alert_triage']['assess_transaction_risk'](**transaction_data)
            investigation_results['results']['risk_assessment'] = risk_assessment
            
            risk_score = risk_assessment.get('risk_score', 0)
            priority_level = risk_assessment.get('priority_level', 'MEDIUM')
            
            logger.info(f"✅ Risk assessment complete: {risk_score:.1%} risk ({priority_level})")
            
            # Update investigation priority based on risk assessment
            if priority_level in ['HIGH', 'CRITICAL']:
                investigation_priority = priority_level
                investigation_results['priority_level'] = investigation_priority
        
        # Step 2: Evidence Collection (if high risk or requested)
        if include_evidence_collection and (risk_score > 0.5 or investigation_priority in ['HIGH', 'CRITICAL']):
            logger.info("🔍 Step 2: Collecting transaction evidence...")
            investigation_results['workflow_steps'].append("evidence_collection")
            
            if 'evidence_collection' in SUB_AGENT_TOOLS:
                target_accounts = [
                    transaction_data.get('originator_account', ''),
                    transaction_data.get('beneficiary_account', '')
                ]
                target_accounts = [acc for acc in target_accounts if acc]
                
                if target_accounts:
                    evidence_data = SUB_AGENT_TOOLS['evidence_collection']['collect_transaction_evidence'](
                        target_accounts=target_accounts,
                        investigation_id=investigation_id,
                        lookback_days=365 if investigation_priority in ['HIGH', 'CRITICAL'] else 180
                    )
                    investigation_results['results']['evidence_collection'] = evidence_data
                    logger.info(f"✅ Evidence collection complete for {len(target_accounts)} accounts")
        
        # Step 3: Sanctions Screening (if requested or high risk)
        if include_sanctions_screening and (risk_score > 0.3 or investigation_priority in ['HIGH', 'CRITICAL']):
            logger.info("⚖️ Step 3: Conducting sanctions screening...")
            investigation_results['workflow_steps'].append("sanctions_screening")
            
            if 'evidence_collection' in SUB_AGENT_TOOLS:
                target_accounts = [
                    transaction_data.get('originator_account', ''),
                    transaction_data.get('beneficiary_account', '')
                ]
                target_accounts = [acc for acc in target_accounts if acc]
                
                if target_accounts:
                    sanctions_results = SUB_AGENT_TOOLS['evidence_collection']['screen_sanctions'](
                        target_accounts=target_accounts,
                        investigation_id=investigation_id
                    )
                    investigation_results['results']['sanctions_screening'] = sanctions_results
                    logger.info(f"✅ Sanctions screening complete")
        
        # Step 4: Advanced Pattern Analysis (NEW)
        if include_pattern_analysis and (risk_score > 0.4 or investigation_priority in ['HIGH', 'CRITICAL']):
            logger.info("🧠 Step 4: Conducting advanced pattern analysis...")
            investigation_results['workflow_steps'].append("pattern_analysis")
            
            if 'pattern_analysis' in SUB_AGENT_TOOLS:
                target_accounts = [
                    transaction_data.get('originator_account', ''),
                    transaction_data.get('beneficiary_account', '')
                ]
                target_accounts = [acc for acc in target_accounts if acc]
                
                if target_accounts:
                    # Comprehensive pattern analysis
                    pattern_analysis_results = SUB_AGENT_TOOLS['pattern_analysis']['analyze_transaction_patterns'](
                        target_accounts=target_accounts,
                        investigation_id=investigation_id,
                        analysis_period_days=90 if investigation_priority in ['HIGH', 'CRITICAL'] else 60,
                        include_network_analysis=True,
                        pattern_types=['amount', 'behavioral', 'velocity', 'network', 'typologies']
                    )
                    investigation_results['pattern_analysis_results']['comprehensive_analysis'] = pattern_analysis_results
                    
                    # Extract pattern risk score for overall assessment
                    pattern_risk_score = pattern_analysis_results.get('risk_assessment', {}).get('risk_score', 0)
                    logger.info(f"✅ Pattern analysis complete: {pattern_risk_score:.1%} pattern risk")
                    
                    # Network anomaly detection (if high pattern risk)
                    if pattern_risk_score > 0.6:
                        network_anomalies = SUB_AGENT_TOOLS['pattern_analysis']['detect_network_anomalies'](
                            target_accounts=target_accounts,
                            investigation_id=investigation_id,
                            sensitivity=0.1
                        )
                        investigation_results['pattern_analysis_results']['network_anomalies'] = network_anomalies
                        logger.info(f"✅ Network anomaly detection complete")
        
        # Step 5: ML Typology Detection (NEW)
        if include_ml_typology_detection and (risk_score > 0.5 or investigation_priority in ['HIGH', 'CRITICAL']):
            logger.info("🕵️ Step 5: Detecting ML typologies...")
            investigation_results['workflow_steps'].append("ml_typology_detection")
            
            if 'pattern_analysis' in SUB_AGENT_TOOLS:
                target_accounts = [
                    transaction_data.get('originator_account', ''),
                    transaction_data.get('beneficiary_account', '')
                ]
                target_accounts = [acc for acc in target_accounts if acc]
                
                if target_accounts:
                    typology_results = SUB_AGENT_TOOLS['pattern_analysis']['identify_ml_typologies'](
                        target_accounts=target_accounts,
                        investigation_id=investigation_id,
                        confidence_threshold=0.7
                    )
                    investigation_results['pattern_analysis_results']['typology_detection'] = typology_results
                    
                    typology_count = typology_results.get('summary', {}).get('total_findings', 0)
                    logger.info(f"✅ ML typology detection complete: {typology_count} findings")
        
        # Step 6: Generate Pattern Insights (NEW)
        if include_pattern_analysis and 'pattern_analysis' in SUB_AGENT_TOOLS:
            logger.info("💡 Step 6: Generating pattern insights...")
            investigation_results['workflow_steps'].append("pattern_insights")
            
            target_accounts = [
                transaction_data.get('originator_account', ''),
                transaction_data.get('beneficiary_account', '')
            ]
            target_accounts = [acc for acc in target_accounts if acc]
            
            if target_accounts:
                pattern_insights = SUB_AGENT_TOOLS['pattern_analysis']['generate_pattern_insights'](
                    investigation_id=investigation_id,
                    target_accounts=target_accounts,
                    analysis_period_days=90,
                    include_recommendations=True
                )
                investigation_results['pattern_analysis_results']['insights'] = pattern_insights
                logger.info(f"✅ Pattern insights generated")
        
        # Step 7: Enhanced Overall Assessment (Updated)
        logger.info("📋 Step 7: Generating enhanced overall assessment...")
        investigation_results['workflow_steps'].append("enhanced_assessment")
        
        overall_assessment = generate_enhanced_overall_assessment(investigation_results)
        investigation_results['overall_assessment'] = overall_assessment
        
        recommended_actions = generate_enhanced_recommended_actions(investigation_results)
        investigation_results['recommended_actions'] = recommended_actions
        
        # Calculate total processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        investigation_results['end_time'] = end_time.isoformat()
        investigation_results['total_processing_time_seconds'] = processing_time
        investigation_results['investigation_status'] = 'COMPLETED'
        
        logger.info(f"🎉 Enhanced investigation {investigation_id} completed in {processing_time:.2f}s")
        
        return investigation_results
        
    except Exception as e:
        logger.error(f"❌ Investigation {investigation_id} failed: {e}")
        investigation_results['investigation_status'] = 'FAILED'
        investigation_results['error'] = str(e)
        return investigation_results

def get_aml_system_status() -> Dict[str, Any]:
    """
    Get comprehensive status of the entire AML multi-agent system including Pattern Analysis.
    
    Returns status and performance metrics from all constituent agents
    and the overall system health.
    
    Returns:
        Dict containing complete system status including Pattern Analysis Agent
    """
    
    try:
        system_status = {
            'system_info': {
                'name': 'AML Investigation System',
                'coordinator_version': os.getenv('AGENT_VERSION', '1.0.0'),
                'status': 'active',
                'total_agents': len(SUB_AGENT_TOOLS),
                'architecture': 'Multi-Agent with Pattern Analysis'  # Updated
            },
            'agent_status': {},
            'system_capabilities': [],
            'overall_health': 'healthy',
            'timestamp': datetime.now().isoformat()
        }
        
        # Get status from each sub-agent
        if 'alert_triage' in SUB_AGENT_TOOLS:
            try:
                alert_status = SUB_AGENT_TOOLS['alert_triage']['get_status']()
                system_status['agent_status']['alert_triage'] = alert_status
                system_status['system_capabilities'].extend([
                    'transaction_risk_assessment',
                    'alert_prioritization',
                    'ensemble_ml_prediction'
                ])
            except Exception as e:
                system_status['agent_status']['alert_triage'] = {'status': 'error', 'error': str(e)}
                system_status['overall_health'] = 'degraded'
        
        if 'evidence_collection' in SUB_AGENT_TOOLS:
            try:
                evidence_status = SUB_AGENT_TOOLS['evidence_collection']['get_status']()
                system_status['agent_status']['evidence_collection'] = evidence_status
                system_status['system_capabilities'].extend([
                    'transaction_evidence_collection',
                    'sanctions_screening',
                    'account_relationship_mapping'
                ])
            except Exception as e:
                system_status['agent_status']['evidence_collection'] = {'status': 'error', 'error': str(e)}
                system_status['overall_health'] = 'degraded'
        
        # Pattern Analysis Agent Status (NEW)
        if 'pattern_analysis' in SUB_AGENT_TOOLS:
            try:
                pattern_status = SUB_AGENT_TOOLS['pattern_analysis']['get_status']()
                system_status['agent_status']['pattern_analysis'] = pattern_status
                system_status['system_capabilities'].extend([
                    'advanced_pattern_analysis',
                    'ml_anomaly_detection',
                    'behavioral_clustering',
                    'velocity_analysis',
                    'network_analysis',
                    'typology_detection',
                    'pattern_insights_generation'
                ])
            except Exception as e:
                system_status['agent_status']['pattern_analysis'] = {'status': 'error', 'error': str(e)}
                system_status['overall_health'] = 'degraded'
        
        # System-level capabilities
        system_status['system_capabilities'].extend([
            'full_investigation_orchestration',
            'multi_agent_coordination',
            'regulatory_compliance_assessment',
            'automated_workflow_management',
            'pattern_based_risk_scoring',  # NEW
            'ml_typology_identification'   # NEW
        ])
        
        # Remove duplicates
        system_status['system_capabilities'] = list(set(system_status['system_capabilities']))
        
        return system_status
        
    except Exception as e:
        return {
            'system_info': {
                'name': 'AML Investigation System',
                'status': 'error',
                'error': str(e)
            },
            'timestamp': datetime.now().isoformat()
        }

def process_multiple_investigations(
    transactions_list: List[Dict[str, Any]],
    batch_priority: str = "MEDIUM",
    enable_pattern_analysis: bool = True
) -> Dict[str, Any]:
    """
    Process multiple transactions for investigation in batch mode with pattern analysis.
    
    Efficiently handles multiple investigations simultaneously with
    intelligent prioritization, resource management, and pattern analysis.
    
    Args:
        transactions_list: List of transaction data dictionaries
        batch_priority: Default priority for batch processing
        enable_pattern_analysis: Whether to include pattern analysis for each investigation
        
    Returns:
        Dict containing batch processing results and enhanced summary
    """
    
    batch_id = f"BATCH_{hash(str(transactions_list))}"[-8:]
    start_time = datetime.now()
    
    logger.info(f"📦 Starting enhanced batch investigation processing: {batch_id}")
    logger.info(f"Processing {len(transactions_list)} transactions")
    logger.info(f"Pattern Analysis: {'Enabled' if enable_pattern_analysis else 'Disabled'}")
    
    batch_results = {
        'batch_id': batch_id,
        'start_time': start_time.isoformat(),
        'total_transactions': len(transactions_list),
        'pattern_analysis_enabled': enable_pattern_analysis,
        'individual_results': [],
        'batch_summary': {},
        'pattern_analysis_summary': {},  # NEW
        'processing_metrics': {}
    }
    
    try:
        # Process each transaction
        for i, transaction_data in enumerate(transactions_list, 1):
            logger.info(f"Processing transaction {i}/{len(transactions_list)}")
            
            # Conduct individual investigation with optional pattern analysis
            investigation_result = conduct_full_aml_investigation(
                transaction_data=transaction_data,
                investigation_priority=batch_priority,
                include_evidence_collection=True,
                include_sanctions_screening=True,
                include_pattern_analysis=enable_pattern_analysis,
                include_ml_typology_detection=enable_pattern_analysis
            )
            
            batch_results['individual_results'].append(investigation_result)
        
        # Generate enhanced batch summary
        batch_results['batch_summary'] = generate_enhanced_batch_summary(batch_results['individual_results'])
        
        # Generate pattern analysis summary (NEW)
        if enable_pattern_analysis:
            batch_results['pattern_analysis_summary'] = generate_pattern_analysis_summary(batch_results['individual_results'])
        
        # Calculate processing metrics
        end_time = datetime.now()
        total_processing_time = (end_time - start_time).total_seconds()
        
        batch_results['end_time'] = end_time.isoformat()
        batch_results['processing_metrics'] = {
            'total_processing_time_seconds': total_processing_time,
            'average_time_per_investigation': total_processing_time / len(transactions_list),
            'successful_investigations': len([r for r in batch_results['individual_results'] if r.get('investigation_status') == 'COMPLETED']),
            'failed_investigations': len([r for r in batch_results['individual_results'] if r.get('investigation_status') == 'FAILED']),
            'pattern_analyses_completed': len([r for r in batch_results['individual_results'] if 'pattern_analysis_results' in r])
        }
        
        batch_results['batch_status'] = 'COMPLETED'
        
        logger.info(f"🎉 Enhanced batch processing {batch_id} completed in {total_processing_time:.2f}s")
        
        return batch_results
        
    except Exception as e:
        logger.error(f"❌ Batch processing {batch_id} failed: {e}")
        batch_results['batch_status'] = 'FAILED'
        batch_results['error'] = str(e)
        return batch_results

# =============================================================================
# Enhanced Helper Functions
# =============================================================================

def generate_enhanced_overall_assessment(investigation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate enhanced overall assessment including pattern analysis results"""
    
    results = investigation_results.get('results', {})
    pattern_results = investigation_results.get('pattern_analysis_results', {})
    
    # Initialize assessment
    assessment = {
        'overall_risk_level': 'MEDIUM',
        'confidence_score': 0.5,
        'key_findings': [],
        'compliance_concerns': [],
        'pattern_analysis_findings': [],  # NEW
        'investigation_quality': 'GOOD'
    }
    
    # Analyze traditional risk assessment results
    if 'risk_assessment' in results:
        risk_data = results['risk_assessment']
        risk_score = risk_data.get('risk_score', 0)
        
        if risk_score > 0.8:
            assessment['overall_risk_level'] = 'HIGH'
        elif risk_score > 0.6:
            assessment['overall_risk_level'] = 'MEDIUM-HIGH'
        elif risk_score < 0.3:
            assessment['overall_risk_level'] = 'LOW'
        
        assessment['confidence_score'] = risk_data.get('confidence', 0.5)
        
        if risk_data.get('reasoning'):
            assessment['key_findings'].extend(risk_data['reasoning'][:2])
    
    # Analyze pattern analysis results (NEW)
    if 'comprehensive_analysis' in pattern_results:
        pattern_data = pattern_results['comprehensive_analysis']
        pattern_risk = pattern_data.get('risk_assessment', {})
        pattern_risk_score = pattern_risk.get('risk_score', 0)
        
        # Elevate risk level if pattern analysis indicates higher risk
        if pattern_risk_score > 0.8 and assessment['overall_risk_level'] not in ['HIGH', 'CRITICAL']:
            assessment['overall_risk_level'] = 'HIGH'
            assessment['pattern_analysis_findings'].append("Advanced ML models detected high-risk patterns")
        
        # Add pattern-specific findings
        ml_analysis = pattern_data.get('ml_pattern_analysis', {})
        for analysis_type, analysis_results in ml_analysis.items():
            if analysis_type == 'amount_anomalies' and 'consensus_analysis' in analysis_results:
                outliers = analysis_results['consensus_analysis'].get('high_confidence_outliers', 0)
                if outliers > 0:
                    assessment['pattern_analysis_findings'].append(f"Detected {outliers} amount anomalies with model consensus")
            
            elif analysis_type == 'behavioral_anomalies' and 'isolation_forest' in analysis_results:
                anomalous_accounts = analysis_results['isolation_forest'].get('anomalous_accounts', [])
                if anomalous_accounts:
                    assessment['pattern_analysis_findings'].append(f"Identified {len(anomalous_accounts)} accounts with atypical behavior")
    
    # Analyze typology detection results (NEW)
    if 'typology_detection' in pattern_results:
        typology_data = pattern_results['typology_detection']
        total_findings = typology_data.get('summary', {}).get('total_findings', 0)
        high_confidence_typologies = typology_data.get('summary', {}).get('high_confidence_typologies', 0)
        
        if high_confidence_typologies > 0:
            assessment['overall_risk_level'] = 'HIGH'
            assessment['pattern_analysis_findings'].append(f"Identified {high_confidence_typologies} known ML typologies with high confidence")
        elif total_findings > 0:
            assessment['pattern_analysis_findings'].append(f"Detected {total_findings} potential ML pattern indicators")
    
    # Analyze sanctions screening
    if 'sanctions_screening' in results:
        sanctions_data = results['sanctions_screening']
        matches = sanctions_data.get('matches_found', [])
        
        if matches:
            assessment['overall_risk_level'] = 'HIGH'
            assessment['compliance_concerns'].append(f"Sanctions matches found: {len(matches)}")
            assessment['key_findings'].append(f"Potential sanctions violations detected")
    
    # Analyze evidence collection
    if 'evidence_collection' in results:
        evidence_data = results['evidence_collection']
        if 'risk_indicators' in evidence_data:
            risk_indicators = evidence_data['risk_indicators']
            if risk_indicators:
                assessment['key_findings'].append(f"Identified {len(risk_indicators)} risk indicators")
    
    return assessment

def generate_enhanced_recommended_actions(investigation_results: Dict[str, Any]) -> List[str]:
    """Generate enhanced recommended actions including pattern analysis insights"""
    
    actions = []
    overall_assessment = investigation_results.get('overall_assessment', {})
    results = investigation_results.get('results', {})
    pattern_results = investigation_results.get('pattern_analysis_results', {})
    
    risk_level = overall_assessment.get('overall_risk_level', 'MEDIUM')
    
    # Risk-based recommendations
    if risk_level == 'HIGH':
        actions.append("IMMEDIATE ACTION: Escalate to senior compliance officer")
        actions.append("Consider filing SAR within 30 days")
        actions.append("Implement enhanced monitoring for all related accounts")
    elif risk_level in ['MEDIUM-HIGH', 'MEDIUM']:
        actions.append("Continue monitoring with increased frequency")
        actions.append("Review account relationship patterns")
        actions.append("Consider additional due diligence procedures")
    else:
        actions.append("Continue standard monitoring procedures")
    
    # Pattern analysis specific recommendations (NEW)
    if 'insights' in pattern_results:
        insights_data = pattern_results['insights']
        pattern_recommendations = insights_data.get('recommendations', [])
        
        for recommendation in pattern_recommendations:
            actions.append(f"PATTERN ANALYSIS: {recommendation}")
    
    # Typology-specific recommendations (NEW)
    if 'typology_detection' in pattern_results:
        typology_data = pattern_results['typology_detection']
        typology_findings = typology_data.get('typology_findings', {})
        
        for typology_name, findings in typology_findings.items():
            if findings.get('overall_confidence', 0) > 0.7:
                actions.append(f"TYPOLOGY ALERT: Investigate {typology_name.replace('_', ' ').title()} patterns immediately")
    
    # Network anomaly recommendations (NEW)
    if 'network_anomalies' in pattern_results:
        network_data = pattern_results['network_anomalies']
        anomaly_detection = network_data.get('anomaly_detection', {})
        
        if 'hub_accounts' in anomaly_detection and anomaly_detection['hub_accounts']:
            actions.append("NETWORK ALERT: Review hub accounts for potential money laundering networks")
    
    # Sanctions-specific recommendations
    if 'sanctions_screening' in results:
        sanctions_data = results['sanctions_screening']
        if sanctions_data.get('matches_found'):
            actions.append("URGENT: Review sanctions matches immediately")
            actions.append("Freeze account activities pending investigation")
            actions.append("Report to regulatory authorities as required")
    
    # Evidence-based recommendations
    if 'evidence_collection' in results:
        actions.append("Maintain comprehensive evidence documentation")
        actions.append("Prepare investigation summary for regulatory review")
    
    return actions

def generate_enhanced_batch_summary(individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate enhanced summary statistics for batch processing including pattern analysis"""
    
    total_investigations = len(individual_results)
    successful_investigations = len([r for r in individual_results if r.get('investigation_status') == 'COMPLETED'])
    
    # Risk level distribution
    risk_levels = {}
    for result in individual_results:
        if 'overall_assessment' in result:
            risk_level = result['overall_assessment'].get('overall_risk_level', 'UNKNOWN')
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
    
    # High-priority investigations
    high_priority_count = len([r for r in individual_results 
                              if r.get('overall_assessment', {}).get('overall_risk_level') == 'HIGH'])
    
    # Pattern analysis statistics (NEW)
    pattern_analyses_completed = len([r for r in individual_results if 'pattern_analysis_results' in r])
    total_pattern_findings = sum([
        r.get('pattern_analysis_results', {}).get('typology_detection', {}).get('summary', {}).get('total_findings', 0)
        for r in individual_results
    ])
    
    return {
        'total_investigations': total_investigations,
        'successful_investigations': successful_investigations,
        'success_rate_percent': (successful_investigations / total_investigations) * 100,
        'risk_level_distribution': risk_levels,
        'high_priority_investigations': high_priority_count,
        'sanctions_matches_found': sum([len(r.get('results', {}).get('sanctions_screening', {}).get('matches_found', [])) 
                                       for r in individual_results]),
        'regulatory_actions_required': high_priority_count,
        'pattern_analysis_completed': pattern_analyses_completed,  # NEW
        'total_pattern_findings': total_pattern_findings,  # NEW
        'pattern_analysis_success_rate': (pattern_analyses_completed / total_investigations) * 100  # NEW
    }

def generate_pattern_analysis_summary(individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary of pattern analysis results across batch"""
    
    pattern_summary = {
        'total_pattern_analyses': 0,
        'anomaly_detection_summary': {},
        'typology_detection_summary': {},
        'network_analysis_summary': {},
        'high_risk_patterns': []
    }
    
    # Aggregate pattern analysis results
    total_amount_anomalies = 0
    total_behavioral_anomalies = 0
    total_velocity_anomalies = 0
    total_network_anomalies = 0
    typology_counts = {}
    
    for result in individual_results:
        pattern_results = result.get('pattern_analysis_results', {})
        
        if 'comprehensive_analysis' in pattern_results:
            pattern_summary['total_pattern_analyses'] += 1
            
            # Count anomalies
            ml_analysis = pattern_results['comprehensive_analysis'].get('ml_pattern_analysis', {})
            
            if 'amount_anomalies' in ml_analysis:
                consensus = ml_analysis['amount_anomalies'].get('consensus_analysis', {})
                total_amount_anomalies += consensus.get('high_confidence_outliers', 0)
            
            if 'behavioral_anomalies' in ml_analysis:
                behavioral = ml_analysis['behavioral_anomalies'].get('isolation_forest', {})
                total_behavioral_anomalies += behavioral.get('anomaly_count', 0)
            
            if 'velocity_anomalies' in ml_analysis:
                velocity = ml_analysis['velocity_anomalies'].get('consensus_analysis', {})
                total_velocity_anomalies += velocity.get('consensus_anomalies', 0)
            
            if 'network_anomalies' in ml_analysis:
                network = ml_analysis['network_anomalies'].get('pca_isolation_forest', {})
                total_network_anomalies += network.get('anomaly_count', 0)
        
        # Count typologies
        if 'typology_detection' in pattern_results:
            typology_findings = pattern_results['typology_detection'].get('typology_findings', {})
            for typology_name, findings in typology_findings.items():
                findings_count = findings.get('findings_count', 0)
                if findings_count > 0:
                    typology_counts[typology_name] = typology_counts.get(typology_name, 0) + findings_count
    
    pattern_summary['anomaly_detection_summary'] = {
        'total_amount_anomalies': total_amount_anomalies,
        'total_behavioral_anomalies': total_behavioral_anomalies,
        'total_velocity_anomalies': total_velocity_anomalies,
        'total_network_anomalies': total_network_anomalies,
        'total_anomalies': total_amount_anomalies + total_behavioral_anomalies + total_velocity_anomalies + total_network_anomalies
    }
    
    pattern_summary['typology_detection_summary'] = {
        'typology_counts': typology_counts,
        'total_typology_findings': sum(typology_counts.values()),
        'unique_typologies_detected': len(typology_counts)
    }
    
    # Identify high-risk patterns
    for result in individual_results:
        pattern_results = result.get('pattern_analysis_results', {})
        investigation_id = result.get('investigation_id', '')
        
        if 'comprehensive_analysis' in pattern_results:
            risk_assessment = pattern_results['comprehensive_analysis'].get('risk_assessment', {})
            risk_score = risk_assessment.get('risk_score', 0)
            
            if risk_score > 0.8:
                pattern_summary['high_risk_patterns'].append({
                    'investigation_id': investigation_id,
                    'pattern_risk_score': risk_score,
                    'risk_level': risk_assessment.get('risk_level', 'UNKNOWN')
                })
    
    return pattern_summary

def generate_batch_summary(individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics for batch processing (legacy compatibility)"""
    
    total_investigations = len(individual_results)
    successful_investigations = len([r for r in individual_results if r.get('investigation_status') == 'COMPLETED'])
    
    # Risk level distribution
    risk_levels = {}
    for result in individual_results:
        if 'overall_assessment' in result:
            risk_level = result['overall_assessment'].get('overall_risk_level', 'UNKNOWN')
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
    
    # High-priority investigations
    high_priority_count = len([r for r in individual_results 
                              if r.get('overall_assessment', {}).get('overall_risk_level') == 'HIGH'])
    
    return {
        'total_investigations': total_investigations,
        'successful_investigations': successful_investigations,
        'success_rate_percent': (successful_investigations / total_investigations) * 100,
        'risk_level_distribution': risk_levels,
        'high_priority_investigations': high_priority_count,
        'sanctions_matches_found': sum([len(r.get('results', {}).get('sanctions_screening', {}).get('matches_found', [])) 
                                       for r in individual_results]),
        'regulatory_actions_required': high_priority_count  # Simplified
    }

# =============================================================================
# Create Individual Sub-Agents (Updated)
# =============================================================================

# Create Alert Triage Agent as sub-agent
alert_triage_sub_agent = LlmAgent(
    name="AlertTriageAgent",
    model=os.getenv('MODEL_NAME', 'gemini-2.0-flash-exp'),
    description="Specialized agent for AML alert triage and risk assessment using ensemble ML models",
    instruction="Focus on transaction risk assessment, alert prioritization, and ML-based predictions. Work seamlessly with Evidence Collection and Pattern Analysis agents.",
    tools=[
        SUB_AGENT_TOOLS.get('alert_triage', {}).get('assess_transaction_risk'),
        SUB_AGENT_TOOLS.get('alert_triage', {}).get('get_status'),
        SUB_AGENT_TOOLS.get('alert_triage', {}).get('batch_assess')
    ] if 'alert_triage' in SUB_AGENT_TOOLS else []
)

# Create Evidence Collection Agent as sub-agent
evidence_collection_sub_agent = LlmAgent(
    name="EvidenceCollectionAgent", 
    model=os.getenv('MODEL_NAME', 'gemini-2.0-flash-exp'),
    description="Specialized agent for comprehensive evidence collection and sanctions screening",
    instruction="Focus on gathering transaction evidence, sanctions screening, and compliance assessment. Coordinate with Pattern Analysis for comprehensive investigations.",
    tools=[
        SUB_AGENT_TOOLS.get('evidence_collection', {}).get('collect_transaction_evidence'),
        SUB_AGENT_TOOLS.get('evidence_collection', {}).get('screen_sanctions'),
        SUB_AGENT_TOOLS.get('evidence_collection', {}).get('compile_evidence'),
        SUB_AGENT_TOOLS.get('evidence_collection', {}).get('get_status')
    ] if 'evidence_collection' in SUB_AGENT_TOOLS else []
)

# Create Pattern Analysis Agent as sub-agent (NEW)
pattern_analysis_sub_agent = LlmAgent(
    name="PatternAnalysisAgent",
    model=os.getenv('MODEL_NAME', 'gemini-2.0-flash-exp'),
    description="Advanced pattern analysis agent using ML models and graph analytics for sophisticated AML pattern detection",
    instruction="Focus on advanced pattern analysis, ML anomaly detection, typology identification, and network analysis. Provide detailed insights for complex investigations.",
    tools=[
        SUB_AGENT_TOOLS.get('pattern_analysis', {}).get('analyze_transaction_patterns'),
        SUB_AGENT_TOOLS.get('pattern_analysis', {}).get('detect_network_anomalies'),
        SUB_AGENT_TOOLS.get('pattern_analysis', {}).get('identify_ml_typologies'),
        SUB_AGENT_TOOLS.get('pattern_analysis', {}).get('generate_pattern_insights'),
        SUB_AGENT_TOOLS.get('pattern_analysis', {}).get('get_status')
    ] if 'pattern_analysis' in SUB_AGENT_TOOLS else []
)

# =============================================================================
# Main Coordinator Agent (Updated)
# =============================================================================

# Create the main AML Coordinator Agent with enhanced multi-agent system
aml_coordinator = LlmAgent(
    name=os.getenv('AGENT_NAME', 'AMLCoordinator'),
    model=os.getenv('MODEL_NAME', 'gemini-2.0-flash-exp'),
    description="""
    I am the AML Investigation Coordinator, orchestrating a sophisticated 4-agent multi-agent system for comprehensive anti-money laundering investigations with advanced pattern analysis capabilities.
    
    🏗️ ENHANCED SYSTEM ARCHITECTURE: I coordinate specialized agents including Alert Triage (ML-powered risk assessment), 
    Evidence Collection (comprehensive data gathering and sanctions screening), and Pattern Analysis (advanced ML anomaly detection, 
    behavioral clustering, network analysis, and ML typology identification) to conduct the most thorough investigations possible.
    
    🎯 COMPREHENSIVE INVESTIGATION ORCHESTRATION: I manage complete investigation workflows from initial risk assessment 
    through evidence collection, sanctions screening, advanced pattern analysis, ML typology detection, network anomaly identification, 
    and final regulatory compliance assessment with enhanced intelligence.
    
    🧠 ADVANCED PATTERN INTELLIGENCE: I leverage cutting-edge ML models including Isolation Forest, Local Outlier Factor, 
    One-Class SVM, behavioral clustering, velocity analysis, and network graph analytics to detect the most sophisticated 
    money laundering patterns including structuring, layering, round-tripping, and smurfing.
    
    📊 INTELLIGENT WORKFLOW MANAGEMENT: I automatically prioritize investigations, delegate tasks to appropriate 
    agents based on risk scores and pattern complexity, and ensure all regulatory requirements are met throughout 
    the enhanced analysis process.
    
    ⚖️ ENHANCED REGULATORY COMPLIANCE: I ensure all investigations meet regulatory standards, recommend SAR/CTR filings 
    based on traditional risk factors AND advanced pattern analysis, and maintain comprehensive audit trails for regulatory review.
    
    🔄 ADVANCED BATCH PROCESSING: I efficiently handle multiple investigations simultaneously with pattern analysis, 
    providing comprehensive reporting, anomaly statistics, typology detection summaries, and network analysis across investigation portfolios.
    
    💡 SOPHISTICATED DECISION SUPPORT: I synthesize results from all agents including advanced pattern insights to provide 
    clear recommendations, multi-dimensional risk assessments, ML-driven typology alerts, and actionable intelligence for 
    compliance teams and investigators.
    
    I serve as the central intelligence hub that transforms complex AML data into actionable compliance insights 
    while ensuring thorough, consistent, audit-ready, and ML-enhanced investigations that can detect the most 
    sophisticated money laundering schemes.
    """,
    instruction="""
    You are the AML Investigation Coordinator managing an advanced 4-agent multi-agent system with sophisticated pattern analysis capabilities. When users request investigations:
    
    1. Use conduct_full_aml_investigation for complete end-to-end investigations including advanced pattern analysis and ML typology detection
    2. Use process_multiple_investigations for batch processing multiple transactions with pattern analysis capabilities
    3. Use get_aml_system_status for system health and agent status information including Pattern Analysis Agent status
    4. Delegate specific tasks to sub-agents when users ask for specialized analysis:
       - Alert Triage: Risk assessment and prioritization
       - Evidence Collection: Data gathering and sanctions screening  
       - Pattern Analysis: Advanced ML anomaly detection, typology identification, network analysis
    
    Always provide comprehensive explanations including:
    - Investigation workflow and methodology across all 4 agent capabilities
    - Risk assessments and findings from all agents including advanced pattern analysis
    - ML model consensus results, anomaly detection findings, and typology identification
    - Network analysis results including centrality anomalies and suspicious flows
    - Regulatory implications and compliance requirements enhanced by pattern intelligence
    - Clear recommendations and next steps incorporating advanced analytics
    
    Coordinate intelligently between all agents to provide thorough, accurate, ML-enhanced, and regulation-compliant investigations that leverage the full power of advanced pattern analysis for detecting sophisticated money laundering schemes.
    """,
    tools=[conduct_full_aml_investigation, get_aml_system_status, process_multiple_investigations],
    sub_agents=[
        alert_triage_sub_agent, 
        evidence_collection_sub_agent, 
        pattern_analysis_sub_agent
    ] if SUB_AGENT_TOOLS else []
)

# ADK CLI requires the agent to be exposed as 'root_agent'
root_agent = aml_coordinator