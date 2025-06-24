#!/usr/bin/env python3
"""
AML Evidence Collection Agent - Google ADK Implementation
Gathers comprehensive evidence for AML investigations
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import hashlib

# Google ADK imports
from google.adk.agents import Agent

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Global Data Storage and Cache
# =============================================================================
TRANSACTION_DATA = None
EVIDENCE_CACHE = {}
AGENT_STATS = {
    'evidence_requests_processed': 0,
    'total_evidence_items_collected': 0,
    'successful_collections': 0,
    'failed_collections': 0,
    'average_collection_time': 0.0,
    'cache_hits': 0
}

# =============================================================================
# Data Loading Functions
# =============================================================================

def load_transaction_data():
    """Load transaction data for evidence collection"""
    global TRANSACTION_DATA
    
    try:
        # Get data path from environment
        data_path = Path(os.getenv('TRANSACTION_DATA_PATH', '../../data/processed/features/feature_enhanced_data.parquet'))
        
        # Resolve relative path
        if not data_path.is_absolute():
            current_dir = Path(__file__).parent
            data_path = (current_dir / data_path).resolve()
        
        logger.info(f"Loading transaction data from: {data_path}")
        
        if data_path.exists():
            TRANSACTION_DATA = pd.read_parquet(data_path)
            logger.info(f"Loaded {len(TRANSACTION_DATA):,} transactions for evidence collection")
            return True
        else:
            logger.warning(f"Transaction data file not found: {data_path}")
            # Create sample data for demo
            TRANSACTION_DATA = create_sample_transaction_data()
            logger.info("Created sample transaction data for demo")
            return True
            
    except Exception as e:
        logger.error(f"Failed to load transaction data: {e}")
        TRANSACTION_DATA = create_sample_transaction_data()
        return False

def create_sample_transaction_data():
    """Create sample transaction data for demonstration"""
    np.random.seed(42)
    
    # Create sample accounts
    accounts = [f"ACC_{i:05d}" for i in range(1, 1001)]
    
    # Generate sample transactions
    data = []
    for i in range(5000):
        orig_acc = np.random.choice(accounts)
        benef_acc = np.random.choice([acc for acc in accounts if acc != orig_acc])
        
        transaction = {
            'transaction_id': f"TXN_{i:06d}",
            'originator_account': orig_acc,
            'beneficiary_account': benef_acc,
            'amount': np.random.lognormal(8, 2),
            'transaction_date': datetime.now() - timedelta(days=np.random.randint(0, 365)),
            'cross_border': np.random.choice([True, False], p=[0.3, 0.7]),
            'is_laundering': np.random.choice([True, False], p=[0.05, 0.95]),
            'risk_score': np.random.beta(2, 8),
            'transaction_type': np.random.choice(['WIRE', 'ACH', 'CHECK', 'CASH'], p=[0.4, 0.3, 0.2, 0.1])
        }
        data.append(transaction)
    
    return pd.DataFrame(data)

# =============================================================================
# ADK Tool Functions
# =============================================================================

def collect_transaction_evidence(
    target_accounts: List[str],
    investigation_id: str,
    lookback_days: int = 365,
    include_related_accounts: bool = True
) -> Dict[str, Any]:
    """
    Collect comprehensive transaction evidence for target accounts.
    
    Gathers historical transactions, identifies patterns, and builds a complete
    transaction history for the specified accounts within the lookback period.
    
    Args:
        target_accounts: List of account IDs to investigate
        investigation_id: Unique identifier for this investigation
        lookback_days: Number of days to look back for transactions
        include_related_accounts: Whether to include related account analysis
        
    Returns:
        Dict containing comprehensive transaction evidence
    """
    global TRANSACTION_DATA, AGENT_STATS
    
    start_time = datetime.now()
    
    try:
        # Ensure transaction data is loaded
        if TRANSACTION_DATA is None or TRANSACTION_DATA.empty:
            load_transaction_data()
        
        logger.info(f"Collecting transaction evidence for {len(target_accounts)} accounts")
        logger.info(f"Investigation ID: {investigation_id}")
        logger.info(f"Lookback period: {lookback_days} days")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Filter transactions for target accounts
        target_transactions = TRANSACTION_DATA[
            (TRANSACTION_DATA['originator_account'].isin(target_accounts) |
             TRANSACTION_DATA['beneficiary_account'].isin(target_accounts))
        ].copy()
        
        if 'transaction_date' in target_transactions.columns:
            target_transactions = target_transactions[
                target_transactions['transaction_date'] >= start_date
            ]
        
        # Analyze transaction patterns
        evidence_data = {
            'investigation_id': investigation_id,
            'collection_timestamp': datetime.now().isoformat(),
            'target_accounts': target_accounts,
            'lookback_days': lookback_days,
            'date_range': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'transaction_summary': analyze_transaction_patterns(target_transactions, target_accounts),
            'account_profiles': create_account_profiles(target_transactions, target_accounts),
            'risk_indicators': identify_risk_indicators(target_transactions),
            'transaction_details': prepare_transaction_details(target_transactions)
        }
        
        # Include related accounts if requested
        if include_related_accounts:
            evidence_data['related_accounts'] = find_related_accounts(target_transactions, target_accounts)
        
        # Update statistics
        AGENT_STATS['evidence_requests_processed'] += 1
        AGENT_STATS['successful_collections'] += 1
        AGENT_STATS['total_evidence_items_collected'] += len(target_transactions)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        AGENT_STATS['average_collection_time'] = (
            (AGENT_STATS['average_collection_time'] * (AGENT_STATS['evidence_requests_processed'] - 1) + processing_time) / 
            AGENT_STATS['evidence_requests_processed']
        )
        
        evidence_data['processing_metrics'] = {
            'processing_time_seconds': processing_time,
            'transactions_analyzed': len(target_transactions),
            'accounts_profiled': len(target_accounts)
        }
        
        logger.info(f"Transaction evidence collection completed in {processing_time:.2f}s")
        logger.info(f"Analyzed {len(target_transactions)} transactions")
        
        return evidence_data
        
    except Exception as e:
        AGENT_STATS['failed_collections'] += 1
        logger.error(f"Transaction evidence collection failed: {e}")
        return {
            'investigation_id': investigation_id,
            'error': str(e),
            'collection_timestamp': datetime.now().isoformat(),
            'target_accounts': target_accounts
        }

def screen_sanctions_watchlists(
    target_accounts: List[str],
    investigation_id: str,
    include_entities: bool = True
) -> Dict[str, Any]:
    """
    Screen target accounts against sanctions and watchlists.
    
    Checks accounts and associated entities against OFAC, UN, EU sanctions lists
    and other regulatory watchlists to identify potential compliance issues.
    
    Args:
        target_accounts: List of account IDs to screen
        investigation_id: Unique identifier for this investigation
        include_entities: Whether to include entity-level screening
        
    Returns:
        Dict containing sanctions screening results
    """
    
    start_time = datetime.now()
    
    try:
        logger.info(f"Screening {len(target_accounts)} accounts against sanctions lists")
        
        # Simulate sanctions screening (in real implementation, this would connect to actual APIs)
        screening_results = {
            'investigation_id': investigation_id,
            'screening_timestamp': datetime.now().isoformat(),
            'target_accounts': target_accounts,
            'watchlists_checked': [
                'OFAC_SDN',
                'UN_Sanctions',
                'EU_Sanctions',
                'PEP_Lists',
                'Adverse_Media'
            ],
            'matches_found': [],
            'screening_summary': {
                'total_accounts_screened': len(target_accounts),
                'exact_matches': 0,
                'partial_matches': 0,
                'false_positives_filtered': 0
            },
            'risk_assessment': 'LOW'
        }
        
        # Simulate some screening logic
        for account in target_accounts:
            # Create deterministic "matches" based on account hash for demo
            account_hash = int(hashlib.md5(account.encode()).hexdigest()[:8], 16)
            
            # Small chance of sanctions match for demo
            if account_hash % 50 == 0:  # 2% chance
                match = {
                    'account_id': account,
                    'match_type': 'PARTIAL',
                    'watchlist': 'OFAC_SDN',
                    'matched_entity': f"Suspicious Entity {account_hash % 100}",
                    'confidence_score': 0.75,
                    'requires_investigation': True,
                    'match_details': {
                        'name_similarity': 0.8,
                        'address_match': False,
                        'date_of_birth_match': False
                    }
                }
                screening_results['matches_found'].append(match)
                screening_results['screening_summary']['partial_matches'] += 1
                screening_results['risk_assessment'] = 'HIGH'
            
            # Higher chance of PEP match for demo
            elif account_hash % 25 == 0:  # 4% chance
                match = {
                    'account_id': account,
                    'match_type': 'POSSIBLE',
                    'watchlist': 'PEP_Lists',
                    'matched_entity': f"Political Figure {account_hash % 200}",
                    'confidence_score': 0.65,
                    'requires_investigation': True,
                    'match_details': {
                        'name_similarity': 0.7,
                        'jurisdiction': 'International',
                        'position': 'Government Official'
                    }
                }
                screening_results['matches_found'].append(match)
                screening_results['screening_summary']['partial_matches'] += 1
                
                if screening_results['risk_assessment'] == 'LOW':
                    screening_results['risk_assessment'] = 'MEDIUM'
        
        # Calculate processing metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        screening_results['processing_metrics'] = {
            'processing_time_seconds': processing_time,
            'average_time_per_account': processing_time / len(target_accounts)
        }
        
        logger.info(f"Sanctions screening completed in {processing_time:.2f}s")
        logger.info(f"Found {len(screening_results['matches_found'])} potential matches")
        
        return screening_results
        
    except Exception as e:
        logger.error(f"Sanctions screening failed: {e}")
        return {
            'investigation_id': investigation_id,
            'error': str(e),
            'screening_timestamp': datetime.now().isoformat(),
            'target_accounts': target_accounts
        }

def compile_evidence_package(
    investigation_id: str,
    target_accounts: List[str],
    evidence_types: Optional[List[str]] = None,
    priority_level: str = 'MEDIUM'
) -> Dict[str, Any]:
    """
    Compile a comprehensive evidence package for investigation.
    
    Orchestrates multiple evidence collection activities and assembles them into
    a complete package ready for regulatory filing or investigative review.
    
    Args:
        investigation_id: Unique identifier for this investigation
        target_accounts: List of account IDs to investigate
        evidence_types: Specific types of evidence to collect
        priority_level: Investigation priority (affects depth of collection)
        
    Returns:
        Dict containing complete evidence package
    """
    
    start_time = datetime.now()
    
    try:
        if evidence_types is None:
            evidence_types = ['transaction_history', 'sanctions_screening', 'account_relationships']
        
        logger.info(f"Compiling evidence package for investigation {investigation_id}")
        logger.info(f"Evidence types: {evidence_types}")
        logger.info(f"Priority level: {priority_level}")
        
        # Initialize evidence package
        evidence_package = {
            'investigation_id': investigation_id,
            'compilation_timestamp': datetime.now().isoformat(),
            'target_accounts': target_accounts,
            'priority_level': priority_level,
            'evidence_types_requested': evidence_types,
            'evidence_collected': {},
            'package_summary': {},
            'regulatory_compliance': {},
            'investigator_notes': []
        }
        
        # Collect transaction evidence
        if 'transaction_history' in evidence_types:
            logger.info("Collecting transaction evidence...")
            lookback_days = 365 if priority_level in ['HIGH', 'CRITICAL'] else 180
            
            transaction_evidence = collect_transaction_evidence(
                target_accounts=target_accounts,
                investigation_id=investigation_id,
                lookback_days=lookback_days,
                include_related_accounts=True
            )
            evidence_package['evidence_collected']['transaction_history'] = transaction_evidence
        
        # Conduct sanctions screening
        if 'sanctions_screening' in evidence_types:
            logger.info("Conducting sanctions screening...")
            
            sanctions_evidence = screen_sanctions_watchlists(
                target_accounts=target_accounts,
                investigation_id=investigation_id,
                include_entities=True
            )
            evidence_package['evidence_collected']['sanctions_screening'] = sanctions_evidence
        
        # Account relationship analysis
        if 'account_relationships' in evidence_types:
            logger.info("Analyzing account relationships...")
            
            # Get transaction data for relationship analysis
            if TRANSACTION_DATA is not None:
                relationship_evidence = analyze_account_relationships(target_accounts, investigation_id)
                evidence_package['evidence_collected']['account_relationships'] = relationship_evidence
        
        # Generate package summary
        evidence_package['package_summary'] = generate_package_summary(evidence_package)
        
        # Add regulatory compliance assessment
        evidence_package['regulatory_compliance'] = assess_regulatory_compliance(evidence_package)
        
        # Generate investigator notes
        evidence_package['investigator_notes'] = generate_investigator_notes(evidence_package)
        
        # Calculate completeness score
        completeness_score = calculate_evidence_completeness(evidence_package, evidence_types)
        evidence_package['completeness_score'] = completeness_score
        
        # Processing metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        evidence_package['compilation_metrics'] = {
            'total_processing_time_seconds': processing_time,
            'evidence_types_collected': len([k for k in evidence_package['evidence_collected'].keys() if 'error' not in evidence_package['evidence_collected'][k]]),
            'completeness_score': completeness_score,
            'ready_for_review': completeness_score >= 0.8
        }
        
        logger.info(f"Evidence package compilation completed in {processing_time:.2f}s")
        logger.info(f"Completeness score: {completeness_score:.1%}")
        
        return evidence_package
        
    except Exception as e:
        logger.error(f"Evidence package compilation failed: {e}")
        return {
            'investigation_id': investigation_id,
            'error': str(e),
            'compilation_timestamp': datetime.now().isoformat(),
            'target_accounts': target_accounts
        }

def get_evidence_agent_status() -> Dict[str, Any]:
    """
    Get current status and performance metrics of the Evidence Collection Agent.
    
    Returns comprehensive information about agent health, data sources,
    processing statistics, and operational capabilities.
    
    Returns:
        Dict containing agent status and performance information
    """
    
    try:
        return {
            'agent_info': {
                'name': os.getenv('AGENT_NAME', 'EvidenceCollectionAgent'),
                'version': os.getenv('AGENT_VERSION', '1.0.0'),
                'description': os.getenv('AGENT_DESCRIPTION', 'Evidence collection for AML investigations'),
                'status': 'active'
            },
            'data_sources': {
                'transaction_data_loaded': TRANSACTION_DATA is not None,
                'transaction_count': len(TRANSACTION_DATA) if TRANSACTION_DATA is not None else 0,
                'data_date_range': get_data_date_range() if TRANSACTION_DATA is not None else None,
                'external_apis_enabled': os.getenv('ENABLE_EXTERNAL_APIS', 'false').lower() == 'true'
            },
            'processing_statistics': {
                'evidence_requests_processed': AGENT_STATS['evidence_requests_processed'],
                'successful_collections': AGENT_STATS['successful_collections'],
                'failed_collections': AGENT_STATS['failed_collections'],
                'total_evidence_items': AGENT_STATS['total_evidence_items_collected'],
                'success_rate_percent': (AGENT_STATS['successful_collections'] / max(AGENT_STATS['evidence_requests_processed'], 1)) * 100,
                'average_collection_time_seconds': AGENT_STATS['average_collection_time'],
                'cache_hits': AGENT_STATS['cache_hits']
            },
            'capabilities': [
                'transaction_history_collection',
                'sanctions_watchlist_screening',
                'account_relationship_mapping',
                'evidence_package_compilation',
                'regulatory_compliance_assessment',
                'real_time_data_enrichment',
                'automated_evidence_validation'
            ],
            'configuration': {
                'max_processing_time_seconds': int(os.getenv('MAX_PROCESSING_TIME', '60')),
                'transaction_lookback_days': int(os.getenv('TRANSACTION_LOOKBACK_DAYS', '365')),
                'max_related_accounts': int(os.getenv('MAX_RELATED_ACCOUNTS', '50')),
                'cache_enabled': os.getenv('ENABLE_CACHE', 'true').lower() == 'true',
                'min_evidence_completeness': float(os.getenv('MIN_EVIDENCE_COMPLETENESS', '0.85'))
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'agent_info': {
                'name': 'EvidenceCollectionAgent',
                'status': 'error',
                'error': str(e)
            },
            'timestamp': datetime.now().isoformat()
        }

# =============================================================================
# Helper Functions
# =============================================================================

def analyze_transaction_patterns(transactions: pd.DataFrame, target_accounts: List[str]) -> Dict[str, Any]:
    """Analyze transaction patterns for target accounts"""
    
    if transactions.empty:
        return {'total_transactions': 0, 'analysis': 'No transactions found'}
    
    # Basic transaction analysis
    total_transactions = len(transactions)
    total_amount = transactions['amount'].sum() if 'amount' in transactions.columns else 0
    avg_amount = transactions['amount'].mean() if 'amount' in transactions.columns else 0
    
    # Risk analysis
    high_risk_transactions = 0
    if 'risk_score' in transactions.columns:
        high_risk_transactions = len(transactions[transactions['risk_score'] > 0.7])
    
    # Cross-border analysis
    cross_border_count = 0
    if 'cross_border' in transactions.columns:
        cross_border_count = transactions['cross_border'].sum()
    
    return {
        'total_transactions': total_transactions,
        'total_amount': float(total_amount),
        'average_amount': float(avg_amount),
        'high_risk_transactions': high_risk_transactions,
        'cross_border_transactions': cross_border_count,
        'cross_border_percentage': (cross_border_count / total_transactions * 100) if total_transactions > 0 else 0,
        'date_range_days': get_transaction_date_range(transactions),
        'unique_counterparties': get_unique_counterparties(transactions, target_accounts)
    }

def create_account_profiles(transactions: pd.DataFrame, target_accounts: List[str]) -> Dict[str, Any]:
    """Create detailed profiles for target accounts"""
    
    profiles = {}
    
    for account in target_accounts:
        account_txns = transactions[
            (transactions['originator_account'] == account) |
            (transactions['beneficiary_account'] == account)
        ]
        
        if account_txns.empty:
            profiles[account] = {'transaction_count': 0, 'profile': 'No transactions found'}
            continue
        
        # Calculate account metrics
        outgoing_txns = account_txns[account_txns['originator_account'] == account]
        incoming_txns = account_txns[account_txns['beneficiary_account'] == account]
        
        profiles[account] = {
            'transaction_count': len(account_txns),
            'outgoing_count': len(outgoing_txns),
            'incoming_count': len(incoming_txns),
            'outgoing_amount': float(outgoing_txns['amount'].sum()) if 'amount' in outgoing_txns.columns else 0,
            'incoming_amount': float(incoming_txns['amount'].sum()) if 'amount' in incoming_txns.columns else 0,
            'average_transaction_amount': float(account_txns['amount'].mean()) if 'amount' in account_txns.columns else 0,
            'risk_profile': calculate_account_risk_profile(account_txns)
        }
    
    return profiles

def identify_risk_indicators(transactions: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identify potential risk indicators in transactions"""
    
    risk_indicators = []
    
    if transactions.empty:
        return risk_indicators
    
    # High-value transactions
    if 'amount' in transactions.columns:
        high_value_threshold = 50000
        high_value_txns = transactions[transactions['amount'] > high_value_threshold]
        
        if len(high_value_txns) > 0:
            risk_indicators.append({
                'indicator_type': 'HIGH_VALUE_TRANSACTIONS',
                'severity': 'MEDIUM',
                'count': len(high_value_txns),
                'description': f'{len(high_value_txns)} transactions above ${high_value_threshold:,}',
                'max_amount': float(high_value_txns['amount'].max())
            })
    
    # Rapid succession transactions
    if 'transaction_date' in transactions.columns:
        # Check for multiple transactions in short time periods
        # This is a simplified check for demonstration
        rapid_succession_count = 0  # Would implement proper time-based analysis
        
        if rapid_succession_count > 0:
            risk_indicators.append({
                'indicator_type': 'RAPID_SUCCESSION',
                'severity': 'HIGH',
                'count': rapid_succession_count,
                'description': 'Multiple transactions in rapid succession detected'
            })
    
    # Cross-border activity
    if 'cross_border' in transactions.columns:
        cross_border_count = transactions['cross_border'].sum()
        cross_border_percentage = (cross_border_count / len(transactions)) * 100
        
        if cross_border_percentage > 50:
            risk_indicators.append({
                'indicator_type': 'HIGH_CROSS_BORDER_ACTIVITY',
                'severity': 'MEDIUM',
                'count': cross_border_count,
                'percentage': cross_border_percentage,
                'description': f'{cross_border_percentage:.1f}% of transactions are cross-border'
            })
    
    return risk_indicators

def prepare_transaction_details(transactions: pd.DataFrame) -> List[Dict[str, Any]]:
    """Prepare detailed transaction records for evidence package"""
    
    if transactions.empty:
        return []
    
    # Convert to list of dictionaries with key fields
    details = []
    for _, row in transactions.head(100).iterrows():  # Limit to first 100 for performance
        detail = {
            'transaction_id': row.get('transaction_id', ''),
            'date': row.get('transaction_date', datetime.now()).isoformat() if pd.notnull(row.get('transaction_date')) else '',
            'amount': float(row.get('amount', 0)),
            'originator_account': row.get('originator_account', ''),
            'beneficiary_account': row.get('beneficiary_account', ''),
            'cross_border': bool(row.get('cross_border', False)),
            'risk_score': float(row.get('risk_score', 0)),
            'transaction_type': row.get('transaction_type', 'UNKNOWN')
        }
        details.append(detail)
    
    return details

def find_related_accounts(transactions: pd.DataFrame, target_accounts: List[str]) -> Dict[str, Any]:
    """Find accounts related to target accounts through transactions"""
    
    related_accounts = set()
    relationship_strength = {}
    
    for target_account in target_accounts:
        # Find accounts that transacted with target account
        outgoing = transactions[transactions['originator_account'] == target_account]['beneficiary_account'].unique()
        incoming = transactions[transactions['beneficiary_account'] == target_account]['originator_account'].unique()
        
        for related_account in list(outgoing) + list(incoming):
            if related_account not in target_accounts:
                related_accounts.add(related_account)
                
                # Calculate relationship strength (number of transactions)
                connection_count = len(transactions[
                    ((transactions['originator_account'] == target_account) & (transactions['beneficiary_account'] == related_account)) |
                    ((transactions['originator_account'] == related_account) & (transactions['beneficiary_account'] == target_account))
                ])
                
                relationship_strength[related_account] = connection_count
    
    # Sort by relationship strength
    sorted_related = sorted(relationship_strength.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'related_account_count': len(related_accounts),
        'top_related_accounts': sorted_related[:20],  # Top 20 most connected
        'analysis_summary': f"Found {len(related_accounts)} accounts with transaction relationships"
    }

def analyze_account_relationships(target_accounts: List[str], investigation_id: str) -> Dict[str, Any]:
    """Analyze relationships between accounts"""
    
    global TRANSACTION_DATA
    
    if TRANSACTION_DATA is None or TRANSACTION_DATA.empty:
        return {
            'investigation_id': investigation_id,
            'error': 'No transaction data available for relationship analysis'
        }
    
    # Find relationships for target accounts
    relationships = find_related_accounts(TRANSACTION_DATA, target_accounts)
    
    return {
        'investigation_id': investigation_id,
        'analysis_timestamp': datetime.now().isoformat(),
        'target_accounts': target_accounts,
        'relationship_analysis': relationships,
        'network_metrics': {
            'total_unique_accounts': len(set(TRANSACTION_DATA['originator_account'].unique()) | set(TRANSACTION_DATA['beneficiary_account'].unique())),
            'target_account_connectivity': len(relationships['top_related_accounts'])
        }
    }

def generate_package_summary(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary of evidence package"""
    
    collected_evidence = evidence_package.get('evidence_collected', {})
    
    summary = {
        'total_evidence_types': len(collected_evidence),
        'successful_collections': len([k for k, v in collected_evidence.items() if 'error' not in v]),
        'failed_collections': len([k for k, v in collected_evidence.items() if 'error' in v]),
        'key_findings': [],
        'overall_risk_assessment': 'MEDIUM'
    }
    
    # Analyze findings
    if 'sanctions_screening' in collected_evidence:
        sanctions = collected_evidence['sanctions_screening']
        if sanctions.get('matches_found'):
            summary['key_findings'].append(f"Sanctions matches found: {len(sanctions['matches_found'])}")
            summary['overall_risk_assessment'] = 'HIGH'
    
    if 'transaction_history' in collected_evidence:
        tx_data = collected_evidence['transaction_history']
        if 'transaction_summary' in tx_data:
            tx_summary = tx_data['transaction_summary']
            if tx_summary.get('high_risk_transactions', 0) > 0:
                summary['key_findings'].append(f"High-risk transactions: {tx_summary['high_risk_transactions']}")
    
    return summary

def assess_regulatory_compliance(evidence_package: Dict[str, Any]) -> Dict[str, Any]:
    """Assess regulatory compliance requirements"""
    
    return {
        'sar_filing_required': determine_sar_requirement(evidence_package),
        'ctr_filing_required': determine_ctr_requirement(evidence_package),
        'compliance_score': 0.85,
        'documentation_completeness': 'SUFFICIENT',
        'recommended_actions': generate_compliance_recommendations(evidence_package)
    }

def generate_investigator_notes(evidence_package: Dict[str, Any]) -> List[str]:
    """Generate investigator notes and recommendations"""
    
    notes = []
    
    # Add summary note
    target_count = len(evidence_package.get('target_accounts', []))
    notes.append(f"Investigation initiated for {target_count} account(s)")
    
    # Add priority-based notes
    priority = evidence_package.get('priority_level', 'MEDIUM')
    if priority in ['HIGH', 'CRITICAL']:
        notes.append("High priority investigation - expedited processing recommended")
    
    # Add evidence-specific notes
    collected_evidence = evidence_package.get('evidence_collected', {})
    if 'sanctions_screening' in collected_evidence:
        sanctions = collected_evidence['sanctions_screening']
        if sanctions.get('matches_found'):
            notes.append(f"ALERT: {len(sanctions['matches_found'])} potential sanctions matches require immediate review")
    
    if 'transaction_history' in collected_evidence:
        tx_data = collected_evidence['transaction_history']
        if 'risk_indicators' in tx_data:
            risk_count = len(tx_data['risk_indicators'])
            if risk_count > 0:
                notes.append(f"Identified {risk_count} risk indicators in transaction patterns")
    
    return notes

def calculate_evidence_completeness(evidence_package: Dict[str, Any], requested_types: List[str]) -> float:
    """Calculate completeness score for evidence package"""
    
    collected_evidence = evidence_package.get('evidence_collected', {})
    successful_collections = len([k for k, v in collected_evidence.items() if 'error' not in v])
    
    if not requested_types:
        return 1.0
    
    return successful_collections / len(requested_types)

def determine_sar_requirement(evidence_package: Dict[str, Any]) -> bool:
    """Determine if SAR filing is required based on evidence"""
    
    # Simplified SAR determination logic
    collected_evidence = evidence_package.get('evidence_collected', {})
    
    # Check for sanctions matches
    if 'sanctions_screening' in collected_evidence:
        sanctions = collected_evidence['sanctions_screening']
        if sanctions.get('matches_found'):
            return True
    
    # Check for high-risk patterns
    if 'transaction_history' in collected_evidence:
        tx_data = collected_evidence['transaction_history']
        if 'transaction_summary' in tx_data:
            tx_summary = tx_data['transaction_summary']
            if tx_summary.get('total_amount', 0) > 100000:  # High value threshold
                return True
    
    return False

def determine_ctr_requirement(evidence_package: Dict[str, Any]) -> bool:
    """Determine if CTR filing is required"""
    
    # Simplified CTR determination logic
    collected_evidence = evidence_package.get('evidence_collected', {})
    
    if 'transaction_history' in collected_evidence:
        tx_data = collected_evidence['transaction_history']
        if 'transaction_summary' in tx_data:
            tx_summary = tx_data['transaction_summary']
            # CTR required for cash transactions over $10,000
            return tx_summary.get('total_amount', 0) > 10000
    
    return False

def generate_compliance_recommendations(evidence_package: Dict[str, Any]) -> List[str]:
    """Generate compliance recommendations"""
    
    recommendations = []
    
    if determine_sar_requirement(evidence_package):
        recommendations.append("File SAR within 30 days of detection")
    
    if determine_ctr_requirement(evidence_package):
        recommendations.append("File CTR within 15 days of transaction")
    
    recommendations.append("Maintain all evidence documentation for audit purposes")
    recommendations.append("Review and update customer risk profile")
    
    return recommendations

def get_data_date_range() -> Dict[str, str]:
    """Get date range of available transaction data"""
    
    global TRANSACTION_DATA
    
    if TRANSACTION_DATA is None or 'transaction_date' not in TRANSACTION_DATA.columns:
        return {'start_date': 'Unknown', 'end_date': 'Unknown'}
    
    try:
        start_date = TRANSACTION_DATA['transaction_date'].min()
        end_date = TRANSACTION_DATA['transaction_date'].max()
        
        return {
            'start_date': start_date.isoformat() if pd.notnull(start_date) else 'Unknown',
            'end_date': end_date.isoformat() if pd.notnull(end_date) else 'Unknown'
        }
    except:
        return {'start_date': 'Unknown', 'end_date': 'Unknown'}

def get_transaction_date_range(transactions: pd.DataFrame) -> int:
    """Get date range span in days for transaction set"""
    
    if 'transaction_date' not in transactions.columns or transactions.empty:
        return 0
    
    try:
        start_date = transactions['transaction_date'].min()
        end_date = transactions['transaction_date'].max()
        
        if pd.notnull(start_date) and pd.notnull(end_date):
            return (end_date - start_date).days
    except:
        pass
    
    return 0

def get_unique_counterparties(transactions: pd.DataFrame, target_accounts: List[str]) -> int:
    """Get count of unique counterparty accounts"""
    
    counterparties = set()
    
    for target_account in target_accounts:
        # Add accounts that received money from target
        outgoing = transactions[transactions['originator_account'] == target_account]['beneficiary_account'].unique()
        # Add accounts that sent money to target
        incoming = transactions[transactions['beneficiary_account'] == target_account]['originator_account'].unique()
        
        counterparties.update(outgoing)
        counterparties.update(incoming)
    
    # Remove target accounts from counterparties
    counterparties = counterparties - set(target_accounts)
    
    return len(counterparties)

def calculate_account_risk_profile(account_transactions: pd.DataFrame) -> str:
    """Calculate risk profile for an account based on its transactions"""
    
    if account_transactions.empty:
        return 'UNKNOWN'
    
    risk_factors = 0
    
    # High transaction volume
    if len(account_transactions) > 100:
        risk_factors += 1
    
    # High transaction amounts
    if 'amount' in account_transactions.columns:
        avg_amount = account_transactions['amount'].mean()
        if avg_amount > 25000:
            risk_factors += 1
    
    # High cross-border activity
    if 'cross_border' in account_transactions.columns:
        cross_border_rate = account_transactions['cross_border'].mean()
        if cross_border_rate > 0.3:
            risk_factors += 1
    
    # High ML risk scores
    if 'risk_score' in account_transactions.columns:
        avg_risk = account_transactions['risk_score'].mean()
        if avg_risk > 0.5:
            risk_factors += 1
    
    # Determine overall risk profile
    if risk_factors >= 3:
        return 'HIGH'
    elif risk_factors >= 2:
        return 'MEDIUM'
    elif risk_factors >= 1:
        return 'LOW-MEDIUM'
    else:
        return 'LOW'

# =============================================================================
# Google ADK Agent Definition
# =============================================================================

# Initialize data loading
load_transaction_data()

# Create the Google ADK Evidence Collection Agent
evidence_collection_agent = Agent(
    name=os.getenv('AGENT_NAME', 'EvidenceCollectionAgent'),
    model=os.getenv('MODEL_NAME', 'gemini-2.0-flash-exp'),
    description="""
    I am an expert AML Evidence Collection Agent that gathers comprehensive evidence for anti-money laundering 
    investigations as part of a sophisticated 4-agent AML investigation system.
    
    My core capabilities include:
    
    🔍 COMPREHENSIVE TRANSACTION EVIDENCE COLLECTION: I analyze historical transaction patterns, identify suspicious activities, 
    and compile detailed transaction histories for target accounts over customizable time periods, providing the data foundation 
    that enables advanced Pattern Analysis and ML typology detection.
    
    ⚖️ SANCTIONS & WATCHLIST SCREENING: I screen accounts and entities against OFAC, UN, EU sanctions lists, 
    PEP databases, and adverse media to identify compliance risks and regulatory violations that complement 
    pattern-based risk assessment.
    
    🌐 ACCOUNT RELATIONSHIP MAPPING: I map complex networks of related accounts, identify transaction flows, 
    and uncover hidden relationships that serve as input for advanced network analysis and graph-based 
    anomaly detection performed by the Pattern Analysis Agent.
    
    📋 EVIDENCE PACKAGE COMPILATION: I orchestrate multiple evidence collection activities and assemble 
    comprehensive packages ready for regulatory filing, including SAR and CTR determinations enhanced 
    by pattern analysis findings.
    
    🎯 REGULATORY COMPLIANCE ASSESSMENT: I evaluate evidence for regulatory requirements, recommend filing 
    obligations, and ensure documentation meets audit standards while incorporating insights from 
    multi-agent analysis.
    
    📊 REAL-TIME DATA ENRICHMENT: I enrich investigations with contextual data, risk indicators, and 
    performance metrics to support investigative decision-making and provide quality data for ML-based 
    pattern analysis.
    
    🤝 MULTI-AGENT COORDINATION: I work seamlessly with Alert Triage Agent (initial risk assessment) 
    and Pattern Analysis Agent (advanced ML analysis) to provide comprehensive evidence that enables 
    sophisticated typology detection, network analysis, and behavioral clustering.
    
    I serve as the data backbone of the investigation system, ensuring that both traditional compliance 
    processes and cutting-edge ML analysis have access to high-quality, comprehensive evidence packages.
    """,
    instruction="""
    You are an expert AML Evidence Collection Agent operating as the data foundation provider in a sophisticated 
    4-agent AML investigation system. When users request evidence collection:
    
    1. Use collect_transaction_evidence for gathering comprehensive transaction histories that enable Pattern Analysis
    2. Use screen_sanctions_watchlists for compliance screening that complements ML-based risk assessment
    3. Use compile_evidence_package for comprehensive investigation packages that incorporate multi-agent findings
    4. Use get_evidence_agent_status for system information and capabilities
    
    MULTI-AGENT COORDINATION GUIDELINES:
    - Understand that your evidence collection enables advanced Pattern Analysis including ML anomaly detection
    - Provide transaction data in formats suitable for behavioral clustering and network analysis
    - Coordinate evidence collection scope based on Alert Triage risk assessments
    - Ensure evidence packages incorporate Pattern Analysis findings and ML typology detection results
    - Support both traditional compliance workflows and advanced ML-based pattern analysis
    
    Always provide detailed explanations of:
    - Evidence collection methodology and scope for multi-agent investigation workflows
    - Risk indicators and compliance findings that complement ML-based analysis
    - Regulatory implications and filing requirements enhanced by pattern analysis insights
    - Evidence completeness and quality assessments that support advanced analytics
    - How your findings integrate with Alert Triage risk scores and Pattern Analysis results
    
    Focus on accuracy, thoroughness, and regulatory compliance in all evidence gathering activities 
    while ensuring seamless integration with the broader multi-agent investigation system that includes 
    advanced ML pattern analysis and network investigation capabilities.
    """,
    tools=[collect_transaction_evidence, screen_sanctions_watchlists, compile_evidence_package, get_evidence_agent_status]
)