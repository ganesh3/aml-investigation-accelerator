#!/usr/bin/env python3
"""
Pattern Analysis Agent Utilities
Helper functions for pattern detection, feature extraction, and analysis
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

# =============================================================================
# Feature Extraction Functions
# =============================================================================

def extract_behavioral_features_for_accounts(transactions: pd.DataFrame, target_accounts: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Extract behavioral features for specific accounts"""
    
    features = []
    valid_accounts = []
    
    for account in target_accounts:
        account_txns = transactions[
            (transactions['originator_account'] == account) |
            (transactions['beneficiary_account'] == account)
        ]
        
        if len(account_txns) < 2:
            continue
        
        outgoing = account_txns[account_txns['originator_account'] == account]
        incoming = account_txns[account_txns['beneficiary_account'] == account]
        
        outgoing_total = float(outgoing['amount'].sum()) if len(outgoing) > 0 else 0
        incoming_total = float(incoming['amount'].sum()) if len(incoming) > 0 else 0
        
        if outgoing_total == 0 and incoming_total == 0:
            continue
        
        total_amount = outgoing_total + incoming_total
        total_count = len(outgoing) + len(incoming)
        
        flow_ratio = (outgoing_total / max(incoming_total, 1)) if incoming_total > 0 else 100
        count_ratio = (len(outgoing) / max(len(incoming), 1)) if len(incoming) > 0 else 100
        avg_amount = total_amount / max(total_count, 1)
        
        cross_border_ratio = float(account_txns['cross_border'].mean()) if 'cross_border' in account_txns.columns else 0
        
        # Count unique counterparties
        counterparties = set()
        counterparties.update(outgoing['beneficiary_account'].unique())
        counterparties.update(incoming['originator_account'].unique())
        counterparties.discard(account)
        
        high_risk_ratio = float(account_txns['risk_score'].mean()) if 'risk_score' in account_txns.columns else 0
        
        time_span_days = (account_txns['transaction_date'].max() - account_txns['transaction_date'].min()).days
        
        feature_vector = [
            np.log(total_amount + 1),
            np.log(total_count + 1),
            np.log(flow_ratio + 1),
            np.log(count_ratio + 1),
            np.log(avg_amount + 1),
            cross_border_ratio,
            len(counterparties),
            high_risk_ratio,
            time_span_days,
            len(account_txns)
        ]
        
        features.append(feature_vector)
        valid_accounts.append(account)
    
    return np.array(features), valid_accounts

def extract_velocity_features_for_accounts(transactions: pd.DataFrame, target_accounts: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Extract velocity features for specific accounts"""
    
    features = []
    valid_accounts = []
    
    for account in target_accounts:
        account_txns = transactions[
            (transactions['originator_account'] == account) |
            (transactions['beneficiary_account'] == account)
        ]
        
        if len(account_txns) < 5:
            continue
        
        account_txns = account_txns.sort_values('transaction_date')
        
        # Calculate velocity metrics
        daily_counts = account_txns.groupby(account_txns['transaction_date'].dt.date).size()
        daily_amounts = account_txns.groupby(account_txns['transaction_date'].dt.date)['amount'].sum()
        hourly_counts = account_txns.groupby(account_txns['transaction_date'].dt.hour).size()
        
        time_diffs = account_txns['transaction_date'].diff().dt.total_seconds().dropna()
        
        if len(daily_counts) == 0 or len(time_diffs) == 0:
            continue
        
        # Weekend vs weekday activity
        account_txns['day_of_week'] = account_txns['transaction_date'].dt.dayofweek
        weekend_ratio = float((account_txns['day_of_week'] >= 5).mean())
        
        # Business hours vs off-hours
        business_hours_ratio = float(((account_txns['transaction_date'].dt.hour >= 9) & 
                                    (account_txns['transaction_date'].dt.hour <= 17)).mean())
        
        feature_vector = [
            np.log(len(account_txns) + 1),
            float(daily_counts.max()),
            float(daily_counts.mean()),
            float(daily_counts.std()),
            np.log(daily_amounts.max() + 1),
            float(daily_amounts.mean()),
            int((daily_counts > daily_counts.quantile(0.95)).sum()),
            int((time_diffs < 3600).sum()),
            np.log(time_diffs.mean() + 1),
            float(time_diffs.std() / max(time_diffs.mean(), 1)),
            len(daily_counts),
            float(hourly_counts.std()),
            weekend_ratio,
            business_hours_ratio
        ]
        
        features.append(feature_vector)
        valid_accounts.append(account)
    
    return np.array(features), valid_accounts

def extract_network_features_for_accounts(network_graph: nx.DiGraph, target_accounts: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Extract network features for specific accounts"""
    
    if not network_graph:
        return np.array([]), []
    
    features = []
    valid_accounts = []
    
    # Calculate centralities for efficiency
    degree_centrality = nx.degree_centrality(network_graph)
    
    # Use subset for expensive calculations
    subgraph_nodes = list(network_graph.nodes())[:1000] if len(network_graph.nodes()) > 1000 else list(network_graph.nodes())
    subgraph = network_graph.subgraph(subgraph_nodes)
    
    try:
        closeness_centrality = nx.closeness_centrality(subgraph)
        betweenness_centrality = nx.betweenness_centrality(subgraph, k=min(100, len(subgraph)))
    except:
        closeness_centrality = {node: 0 for node in subgraph_nodes}
        betweenness_centrality = {node: 0 for node in subgraph_nodes}
    
    clustering = nx.clustering(network_graph.to_undirected())
    
    try:
        core_numbers = nx.core_number(network_graph.to_undirected())
    except:
        core_numbers = {node: 0 for node in network_graph.nodes()}
    
    for account in target_accounts:
        if account not in network_graph:
            continue
        
        try:
            feature_vector = [
                network_graph.degree(account),
                network_graph.in_degree(account),
                network_graph.out_degree(account),
                clustering.get(account, 0),
                core_numbers.get(account, 0),
                degree_centrality[account],
                closeness_centrality.get(account, 0),
                betweenness_centrality.get(account, 0),
                np.log(network_graph.nodes[account].get('total_sent', 0) + 1),
                np.log(network_graph.nodes[account].get('total_received', 0) + 1),
                network_graph.nodes[account].get('transaction_count', 0),
                len(list(network_graph.neighbors(account)))
            ]
            
            features.append(feature_vector)
            valid_accounts.append(account)
            
        except Exception as e:
            logger.warning(f"Failed to extract network features for {account}: {e}")
            continue
    
    return np.array(features), valid_accounts

# =============================================================================
# Typology Detection Functions
# =============================================================================

def detect_structuring_patterns(transactions: pd.DataFrame, target_accounts: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Detect structuring/smurfing patterns"""
    
    structuring_findings = []
    
    threshold_amount = config.get('threshold_amount', 10000)
    time_window_hours = config.get('time_window_hours', 24)
    min_transactions = config.get('min_transactions', 3)
    proximity_percentage = config.get('proximity_percentage', 10)
    
    for account in target_accounts:
        account_txns = transactions[
            (transactions['originator_account'] == account) |
            (transactions['beneficiary_account'] == account)
        ].sort_values('transaction_date')
        
        if len(account_txns) < min_transactions:
            continue
        
        # Group transactions by time windows
        for i in range(len(account_txns) - min_transactions + 1):
            window_start = account_txns.iloc[i]['transaction_date']
            window_end = window_start + timedelta(hours=time_window_hours)
            
            window_txns = account_txns[
                (account_txns['transaction_date'] >= window_start) &
                (account_txns['transaction_date'] <= window_end)
            ]
            
            if len(window_txns) < min_transactions:
                continue
            
            # Check for amounts just below thresholds
            near_threshold_count = 0
            for amount in window_txns['amount']:
                for threshold in [threshold_amount, 15000, 25000]:
                    if (threshold - amount) / threshold <= (proximity_percentage / 100):
                        near_threshold_count += 1
                        break
            
            if near_threshold_count >= min_transactions:
                confidence_score = min(0.9, near_threshold_count / len(window_txns))
                
                structuring_findings.append({
                    'account': account,
                    'pattern_type': 'STRUCTURING',
                    'time_window': {
                        'start': window_start.isoformat(),
                        'end': window_end.isoformat()
                    },
                    'transaction_count': len(window_txns),
                    'near_threshold_count': near_threshold_count,
                    'confidence_score': confidence_score,
                    'total_amount': float(window_txns['amount'].sum()),
                    'evidence': {
                        'transaction_ids': window_txns['transaction_id'].tolist()[:5],
                        'amounts': window_txns['amount'].tolist()[:5]
                    }
                })
    
    return {
        'typology': 'STRUCTURING',
        'findings_count': len(structuring_findings),
        'findings': structuring_findings,
        'overall_confidence': np.mean([f['confidence_score'] for f in structuring_findings]) if structuring_findings else 0
    }

def detect_layering_patterns(transactions: pd.DataFrame, target_accounts: List[str], network_graph: nx.DiGraph, config: Dict[str, Any]) -> Dict[str, Any]:
    """Detect layering patterns (complex transaction chains)"""
    
    layering_findings = []
    
    min_chain_length = config.get('min_chain_length', 3)
    max_chain_length = config.get('max_chain_length', 8)
    time_window_hours = config.get('time_window_hours', 72)
    amount_similarity_threshold = config.get('amount_similarity_threshold', 0.15)
    
    for account in target_accounts:
        if account not in network_graph:
            continue
        
        # Find transaction chains starting from this account
        try:
            # Simple path finding (limited depth for performance)
            paths = []
            for target in list(network_graph.neighbors(account))[:10]:  # Limit neighbors
                try:
                    simple_paths = list(nx.all_simple_paths(
                        network_graph, account, target, 
                        cutoff=min(max_chain_length, 4)
                    ))
                    paths.extend([path for path in simple_paths if len(path) >= min_chain_length])
                except:
                    continue
            
            # Analyze each path for layering characteristics
            for path in paths[:10]:  # Limit analysis for performance
                path_transactions = []
                path_amounts = []
                
                for i in range(len(path) - 1):
                    orig_acc = path[i]
                    benef_acc = path[i + 1]
                    
                    # Find transactions between these accounts
                    chain_txns = transactions[
                        (transactions['originator_account'] == orig_acc) &
                        (transactions['beneficiary_account'] == benef_acc)
                    ]
                    
                    if len(chain_txns) > 0:
                        path_transactions.extend(chain_txns['transaction_id'].tolist())
                        path_amounts.extend(chain_txns['amount'].tolist())
                
                if len(path_amounts) >= min_chain_length:
                    # Check for amount similarity (indicating coordinated layering)
                    amount_variance = np.std(path_amounts) / max(np.mean(path_amounts), 1)
                    
                    if amount_variance <= amount_similarity_threshold:
                        confidence_score = min(0.9, (len(path) - min_chain_length) / (max_chain_length - min_chain_length))
                        
                        layering_findings.append({
                            'account': account,
                            'pattern_type': 'LAYERING',
                            'chain_length': len(path),
                            'chain_accounts': path,
                            'confidence_score': confidence_score,
                            'amount_consistency': 1 - amount_variance,
                            'total_amount': sum(path_amounts),
                            'evidence': {
                                'transaction_ids': path_transactions[:5],
                                'amounts': path_amounts[:5]
                            }
                        })
        
        except Exception as e:
            logger.warning(f"Layering detection failed for {account}: {e}")
            continue
    
    return {
        'typology': 'LAYERING',
        'findings_count': len(layering_findings),
        'findings': layering_findings,
        'overall_confidence': np.mean([f['confidence_score'] for f in layering_findings]) if layering_findings else 0
    }

def detect_round_tripping_patterns(transactions: pd.DataFrame, target_accounts: List[str], network_graph: nx.DiGraph, config: Dict[str, Any]) -> Dict[str, Any]:
    """Detect round-tripping patterns (circular flows)"""
    
    round_trip_findings = []
    
    min_cycle_length = config.get('min_cycle_length', 2)
    max_cycle_length = config.get('max_cycle_length', 6)
    max_net_change_ratio = config.get('max_net_change_ratio', 0.1)
    
    for account in target_accounts:
        if account not in network_graph:
            continue
        
        try:
            # Find cycles starting and ending at this account
            cycles = []
            for neighbor in list(network_graph.neighbors(account))[:5]:  # Limit for performance
                try:
                    paths = list(nx.all_simple_paths(
                        network_graph, neighbor, account,
                        cutoff=max_cycle_length - 1
                    ))
                    for path in paths:
                        if len(path) >= min_cycle_length - 1:
                            full_cycle = [account] + path
                            cycles.append(full_cycle)
                except:
                    continue
            
            # Analyze each cycle
            for cycle in cycles[:5]:  # Limit for performance
                cycle_amounts = []
                outgoing_amount = 0
                incoming_amount = 0
                
                for i in range(len(cycle)):
                    orig_acc = cycle[i]
                    benef_acc = cycle[(i + 1) % len(cycle)]
                    
                    cycle_txns = transactions[
                        (transactions['originator_account'] == orig_acc) &
                        (transactions['beneficiary_account'] == benef_acc)
                    ]
                    
                    if len(cycle_txns) > 0:
                        amount = cycle_txns['amount'].sum()
                        cycle_amounts.append(amount)
                        
                        if orig_acc == account:
                            outgoing_amount += amount
                        elif benef_acc == account:
                            incoming_amount += amount
                
                if cycle_amounts and outgoing_amount > 0:
                    net_change_ratio = abs(outgoing_amount - incoming_amount) / outgoing_amount
                    
                    if net_change_ratio <= max_net_change_ratio:
                        confidence_score = min(0.9, 1 - net_change_ratio)
                        
                        round_trip_findings.append({
                            'account': account,
                            'pattern_type': 'ROUND_TRIPPING',
                            'cycle_length': len(cycle),
                            'cycle_accounts': cycle,
                            'confidence_score': confidence_score,
                            'net_change_ratio': net_change_ratio,
                            'outgoing_amount': outgoing_amount,
                            'incoming_amount': incoming_amount,
                            'evidence': {
                                'cycle_amounts': cycle_amounts
                            }
                        })
        
        except Exception as e:
            logger.warning(f"Round-tripping detection failed for {account}: {e}")
            continue
    
    return {
        'typology': 'ROUND_TRIPPING',
        'findings_count': len(round_trip_findings),
        'findings': round_trip_findings,
        'overall_confidence': np.mean([f['confidence_score'] for f in round_trip_findings]) if round_trip_findings else 0
    }

def detect_smurfing_patterns(transactions: pd.DataFrame, target_accounts: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Detect smurfing patterns (coordinated multi-account transactions)"""
    
    smurfing_findings = []
    
    coordination_window_minutes = config.get('coordination_window_minutes', 30)
    min_coordinated_accounts = config.get('min_coordinated_accounts', 3)
    amount_similarity_threshold = config.get('amount_similarity_threshold', 0.1)
    
    # Group transactions by time windows
    time_windows = {}
    for _, txn in transactions.iterrows():
        # Round timestamp to coordination window
        window_start = txn['transaction_date'].replace(second=0, microsecond=0)
        window_key = window_start.strftime('%Y%m%d_%H%M')
        
        if window_key not in time_windows:
            time_windows[window_key] = []
        time_windows[window_key].append(txn)
    
    # Analyze each time window for coordination
    for window_key, window_txns in time_windows.items():
        if len(window_txns) < min_coordinated_accounts:
            continue
        
        window_df = pd.DataFrame(window_txns)
        
        # Check if target accounts are involved
        involved_accounts = set(window_df['originator_account']) | set(window_df['beneficiary_account'])
        target_involved = [acc for acc in target_accounts if acc in involved_accounts]
        
        if not target_involved:
            continue
        
        # Check for amount similarity
        amounts = window_df['amount'].values
        if len(amounts) >= min_coordinated_accounts:
            amount_variance = np.std(amounts) / max(np.mean(amounts), 1)
            
            if amount_variance <= amount_similarity_threshold:
                confidence_score = min(0.9, 1 - amount_variance)
                
                smurfing_findings.append({
                    'pattern_type': 'SMURFING',
                    'time_window': window_key,
                    'involved_accounts': list(involved_accounts),
                    'target_accounts_involved': target_involved,
                    'coordinated_transaction_count': len(window_txns),
                    'confidence_score': confidence_score,
                    'amount_similarity': 1 - amount_variance,
                    'total_amount': float(window_df['amount'].sum()),
                    'evidence': {
                        'transaction_ids': window_df['transaction_id'].tolist()[:5],
                        'amounts': amounts.tolist()[:5]
                    }
                })
    
    return {
        'typology': 'SMURFING',
        'findings_count': len(smurfing_findings),
        'findings': smurfing_findings,
        'overall_confidence': np.mean([f['confidence_score'] for f in smurfing_findings]) if smurfing_findings else 0
    }

# =============================================================================
# Network Analysis Functions
# =============================================================================

def detect_centrality_anomalies(graph: nx.DiGraph, target_accounts: List[str], sensitivity: float) -> Dict[str, Any]:
    """Detect accounts with unusual centrality measures"""
    
    results = {
        'high_degree_accounts': [],
        'high_betweenness_accounts': [],
        'isolated_accounts': [],
        'hub_accounts': []
    }
    
    try:
        # Calculate centralities
        degree_centrality = nx.degree_centrality(graph)
        
        # Use subgraph for expensive calculations
        subgraph_nodes = list(graph.nodes())[:500] if len(graph.nodes()) > 500 else list(graph.nodes())
        subgraph = graph.subgraph(subgraph_nodes)
        
        betweenness_centrality = nx.betweenness_centrality(subgraph, k=100)
        
        # Define thresholds based on sensitivity
        degree_threshold = np.percentile(list(degree_centrality.values()), (1 - sensitivity) * 100)
        betweenness_threshold = np.percentile(list(betweenness_centrality.values()), (1 - sensitivity) * 100)
        
        for account in target_accounts:
            if account not in graph:
                continue
            
            degree_cent = degree_centrality.get(account, 0)
            betweenness_cent = betweenness_centrality.get(account, 0)
            degree = graph.degree(account)
            
            # High degree centrality
            if degree_cent > degree_threshold:
                results['high_degree_accounts'].append({
                    'account': account,
                    'degree_centrality': degree_cent,
                    'total_connections': degree
                })
            
            # High betweenness centrality
            if betweenness_cent > betweenness_threshold:
                results['high_betweenness_accounts'].append({
                    'account': account,
                    'betweenness_centrality': betweenness_cent
                })
            
            # Isolated accounts
            if degree <= 2:
                results['isolated_accounts'].append({
                    'account': account,
                    'degree': degree
                })
            
            # Hub accounts (high degree and high betweenness)
            if degree_cent > degree_threshold and betweenness_cent > betweenness_threshold:
                results['hub_accounts'].append({
                    'account': account,
                    'degree_centrality': degree_cent,
                    'betweenness_centrality': betweenness_cent
                })
        
    except Exception as e:
        logger.warning(f"Centrality anomaly detection failed: {e}")
        results['error'] = str(e)
    
    return results

def detect_suspicious_flows(graph: nx.DiGraph, transactions: pd.DataFrame, target_accounts: List[str]) -> Dict[str, Any]:
    """Detect suspicious transaction flow patterns"""
    
    results = {
        'flow_imbalances': [],
        'circular_flows': [],
        'rapid_flows': []
    }
    
    try:
        for account in target_accounts:
            if account not in graph:
                continue
            
            # Flow imbalance analysis
            total_sent = graph.nodes[account].get('total_sent', 0)
            total_received = graph.nodes[account].get('total_received', 0)
            
            if total_received > 0:
                flow_ratio = total_sent / total_received
                if flow_ratio > 10 or flow_ratio < 0.1:  # Extreme imbalance
                    results['flow_imbalances'].append({
                        'account': account,
                        'flow_ratio': flow_ratio,
                        'total_sent': total_sent,
                        'total_received': total_received,
                        'imbalance_type': 'SENDER_DOMINANT' if flow_ratio > 10 else 'RECEIVER_DOMINANT'
                    })
        
    except Exception as e:
        logger.warning(f"Suspicious flow detection failed: {e}")
        results['error'] = str(e)
    
    return results

def detect_money_mules(graph: nx.DiGraph, transactions: pd.DataFrame, target_accounts: List[str]) -> Dict[str, Any]:
    """Detect potential money mule indicators"""
    
    mule_indicators = []
    
    try:
        for account in target_accounts:
            if account not in graph:
                continue
            
            # Money mule characteristics
            in_degree = graph.in_degree(account)
            out_degree = graph.out_degree(account)
            total_received = graph.nodes[account].get('total_received', 0)
            total_sent = graph.nodes[account].get('total_sent', 0)
            
            # High incoming, rapid outgoing pattern
            if (in_degree > 5 and out_degree > 3 and 
                total_received > total_sent * 0.8 and
                total_received > 50000):
                
                mule_score = calculate_mule_score(in_degree, out_degree, total_received, total_sent)
                
                mule_indicators.append({
                    'account': account,
                    'mule_score': mule_score,
                    'in_degree': in_degree,
                    'out_degree': out_degree,
                    'total_received': total_received,
                    'total_sent': total_sent,
                    'indicators': ['high_incoming_volume', 'rapid_outgoing', 'amount_similarity']
                })
        
    except Exception as e:
        logger.warning(f"Money mule detection failed: {e}")
        return {'error': str(e)}
    
    return {
        'potential_mules': mule_indicators,
        'mule_count': len(mule_indicators)
    }

def calculate_mule_score(in_degree: int, out_degree: int, total_received: float, total_sent: float) -> float:
    """Calculate money mule probability score"""
    
    # Factors that indicate money mule behavior
    flow_ratio = total_sent / max(total_received, 1)
    degree_ratio = out_degree / max(in_degree, 1)
    volume_factor = min(1.0, total_received / 100000)  # Normalize by 100k
    
    # Weighted score
    mule_score = (
        0.4 * min(1.0, flow_ratio) +      # How much flows out vs in
        0.3 * min(1.0, degree_ratio) +    # Distribution pattern
        0.3 * volume_factor                # Volume factor
    )
    
    return min(1.0, mule_score)

# =============================================================================
# Risk Scoring and Analysis Functions
# =============================================================================

def calculate_ml_risk_score(ml_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall risk score based on ML analysis results"""
    
    risk_factors = []
    total_anomalies = 0
    confidence_scores = []
    
    # Amount anomalies
    if 'amount_anomalies' in ml_analysis:
        amount_data = ml_analysis['amount_anomalies']
        if 'consensus_analysis' in amount_data:
            high_conf_outliers = amount_data['consensus_analysis'].get('high_confidence_outliers', 0)
            if high_conf_outliers > 0:
                risk_factors.append(f"High-confidence amount anomalies: {high_conf_outliers}")
                total_anomalies += high_conf_outliers
    
    # Behavioral anomalies
    if 'behavioral_anomalies' in ml_analysis:
        behavioral_data = ml_analysis['behavioral_anomalies']
        if 'isolation_forest' in behavioral_data:
            anomaly_count = behavioral_data['isolation_forest'].get('anomaly_count', 0)
            if anomaly_count > 0:
                risk_factors.append(f"Behavioral anomalies detected: {anomaly_count}")
                total_anomalies += anomaly_count
                if 'mean_anomaly_score' in behavioral_data['isolation_forest']:
                    confidence_scores.append(abs(behavioral_data['isolation_forest']['mean_anomaly_score']))
    
    # Velocity anomalies
    if 'velocity_anomalies' in ml_analysis:
        velocity_data = ml_analysis['velocity_anomalies']
        if 'consensus_analysis' in velocity_data:
            consensus_anomalies = velocity_data['consensus_analysis'].get('consensus_anomalies', 0)
            if consensus_anomalies > 0:
                risk_factors.append(f"Velocity consensus anomalies: {consensus_anomalies}")
                total_anomalies += consensus_anomalies
    
    # Network anomalies
    if 'network_anomalies' in ml_analysis:
        network_data = ml_analysis['network_anomalies']
        if 'pca_isolation_forest' in network_data:
            network_anomaly_count = network_data['pca_isolation_forest'].get('anomaly_count', 0)
            if network_anomaly_count > 0:
                risk_factors.append(f"Network structure anomalies: {network_anomaly_count}")
                total_anomalies += network_anomaly_count
    
    # Calculate overall risk score
    base_risk = min(0.7, total_anomalies * 0.1)  # Each anomaly adds 10% risk, max 70%
    
    # Boost risk if multiple model types agree
    model_types_with_anomalies = len([k for k in ml_analysis.keys() if 'anomalies' in k and ml_analysis[k]])
    consensus_boost = model_types_with_anomalies * 0.05  # 5% boost per model type
    
    # Confidence boost from model certainty
    confidence_boost = np.mean(confidence_scores) * 0.1 if confidence_scores else 0
    
    final_risk_score = min(0.95, base_risk + consensus_boost + confidence_boost)
    
    # Calculate confidence (based on model agreement)
    confidence_score = min(0.95, 0.5 + (model_types_with_anomalies * 0.15))
    
    return {
        'risk_score': float(final_risk_score),
        'confidence_score': float(confidence_score),
        'risk_factors': risk_factors,
        'total_anomalies': total_anomalies,
        'model_consensus': model_types_with_anomalies
    }

def get_risk_level(risk_score: float) -> str:
    """Convert risk score to risk level"""
    if risk_score >= 0.9:
        return 'CRITICAL'
    elif risk_score >= 0.7:
        return 'HIGH'
    elif risk_score >= 0.5:
        return 'MEDIUM'
    elif risk_score >= 0.3:
        return 'LOW'
    else:
        return 'MINIMAL'

def get_counterparties(transactions: pd.DataFrame, target_accounts: List[str]) -> set:
    """Get unique counterparty accounts for target accounts"""
    
    counterparties = set()
    
    for target_account in target_accounts:
        # Outgoing transactions
        outgoing = transactions[transactions['originator_account'] == target_account]['beneficiary_account'].unique()
        # Incoming transactions
        incoming = transactions[transactions['beneficiary_account'] == target_account]['originator_account'].unique()
        
        counterparties.update(outgoing)
        counterparties.update(incoming)
    
    # Remove target accounts from counterparties
    counterparties = counterparties - set(target_accounts)
    
    return counterparties

def generate_pattern_insights(analysis_results: Dict[str, Any]) -> List[str]:
    """Generate human-readable insights from analysis results"""
    
    insights = []
    
    # ML insights
    ml_analysis = analysis_results.get('ml_pattern_analysis', {})
    
    if 'amount_anomalies' in ml_analysis:
        amount_data = ml_analysis['amount_anomalies']
        if 'consensus_analysis' in amount_data:
            outliers = amount_data['consensus_analysis'].get('high_confidence_outliers', 0)
            if outliers > 0:
                insights.append(f"Detected {outliers} transactions with unusual amounts using ensemble ML models")
                    
    if 'behavioral_anomalies' in ml_analysis:
        behavioral_data = ml_analysis['behavioral_anomalies']
        if 'isolation_forest' in behavioral_data:
            anomalous_accounts = behavioral_data['isolation_forest'].get('anomalous_accounts', [])
            if anomalous_accounts:
                insights.append(f"Identified {len(anomalous_accounts)} accounts with atypical behavioral patterns")
    
    # Typology insights
    typology_detection = analysis_results.get('typology_detection', {})
    for typology_name, results in typology_detection.items():
        findings_count = results.get('findings_count', 0)
        if findings_count > 0:
            confidence = results.get('overall_confidence', 0)
            insights.append(f"Found {findings_count} instances of {typology_name} patterns (confidence: {confidence:.1%})")
    
    return insights

# =============================================================================
# Insight Generation Functions
# =============================================================================

def get_primary_concern(analysis_results: Dict[str, Any]) -> str:
    """Identify primary concern from analysis results"""
    
    risk_score = analysis_results.get('risk_assessment', {}).get('overall_risk_score', 0)
    typology_detection = analysis_results.get('typology_detection', {})
    
    # Check for high-confidence typologies
    high_conf_typologies = []
    for typology_name, results in typology_detection.items():
        if results.get('overall_confidence', 0) > 0.7:
            high_conf_typologies.append(typology_name)
    
    if high_conf_typologies:
        return f"Known ML typology detected: {', '.join(high_conf_typologies)}"
    elif risk_score > 0.8:
        return "Multiple anomaly indicators"
    elif risk_score > 0.6:
        return "Suspicious transaction patterns"
    else:
        return "Low-level irregularities"

def get_investigation_urgency(risk_score: float) -> str:
    """Determine investigation urgency based on risk score"""
    
    if risk_score >= 0.9:
        return "IMMEDIATE"
    elif risk_score >= 0.7:
        return "HIGH"
    elif risk_score >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"

def sum_total_anomalies(ml_analysis: Dict[str, Any]) -> int:
    """Sum total anomalies across all ML analyses"""
    
    total = 0
    
    if 'amount_anomalies' in ml_analysis:
        total += ml_analysis['amount_anomalies'].get('consensus_analysis', {}).get('high_confidence_outliers', 0)
    
    if 'behavioral_anomalies' in ml_analysis:
        total += ml_analysis['behavioral_anomalies'].get('isolation_forest', {}).get('anomaly_count', 0)
    
    if 'velocity_anomalies' in ml_analysis:
        total += ml_analysis['velocity_anomalies'].get('consensus_analysis', {}).get('consensus_anomalies', 0)
    
    if 'network_anomalies' in ml_analysis:
        total += ml_analysis['network_anomalies'].get('pca_isolation_forest', {}).get('anomaly_count', 0)
    
    return total

def sum_typology_findings(typology_detection: Dict[str, Any]) -> int:
    """Sum total typology findings"""
    
    return sum(results.get('findings_count', 0) for results in typology_detection.values())

def calculate_overall_suspicion_level(typologies_detected: Dict[str, Any]) -> str:
    """Calculate overall suspicion level from typology detection"""
    
    max_confidence = max((results.get('overall_confidence', 0) for results in typologies_detected.values()), default=0)
    total_findings = sum(results.get('findings_count', 0) for results in typologies_detected.values())
    
    if max_confidence > 0.8 or total_findings > 5:
        return 'VERY_HIGH'
    elif max_confidence > 0.6 or total_findings > 2:
        return 'HIGH'
    elif max_confidence > 0.4 or total_findings > 0:
        return 'MEDIUM'
    else:
        return 'LOW'

def assess_data_quality(analysis_results: Dict[str, Any]) -> float:
    """Assess data quality for the analysis"""
    
    transaction_summary = analysis_results.get('transaction_summary', {})
    total_transactions = transaction_summary.get('total_transactions', 0)
    
    # Simple data quality assessment
    if total_transactions > 100:
        return 0.9
    elif total_transactions > 50:
        return 0.8
    elif total_transactions > 10:
        return 0.7
    else:
        return 0.6

def assess_analysis_completeness(analysis_results: Dict[str, Any]) -> float:
    """Assess completeness of the analysis"""
    
    ml_analysis = analysis_results.get('ml_pattern_analysis', {})
    typology_detection = analysis_results.get('typology_detection', {})
    
    components = 0
    total_components = 5  # amount, behavioral, velocity, network, typologies
    
    if 'amount_anomalies' in ml_analysis:
        components += 1
    if 'behavioral_anomalies' in ml_analysis:
        components += 1
    if 'velocity_anomalies' in ml_analysis:
        components += 1
    if 'network_anomalies' in ml_analysis:
        components += 1
    if typology_detection:
        components += 1
    
    return components / total_components

def check_ctr_requirement(analysis_results: Dict[str, Any]) -> bool:
    """Check if CTR filing is required"""
    
    transaction_summary = analysis_results.get('transaction_summary', {})
    total_amount = transaction_summary.get('total_amount', 0)
    
    # CTR required for cash transactions over $10,000
    return total_amount > 10000

def generate_compliance_actions(analysis_results: Dict[str, Any]) -> List[str]:
    """Generate compliance actions based on analysis"""
    
    actions = []
    risk_score = analysis_results.get('risk_assessment', {}).get('overall_risk_score', 0)
    
    if risk_score > 0.8:
        actions.extend([
            'File SAR within 30 days',
            'Implement enhanced monitoring',
            'Escalate to compliance officer'
        ])
    elif risk_score > 0.6:
        actions.extend([
            'Continue monitoring with increased frequency',
            'Document findings for potential SAR'
        ])
    
    return actions

def analyze_account_relationships(graph: nx.DiGraph, target_accounts: List[str]) -> Dict[str, Any]:
    """Analyze relationships between target accounts"""
    
    relationships = {
        'direct_connections': [],
        'indirect_connections': [],
        'common_counterparties': [],
        'relationship_strength': {}
    }
    
    try:
        # Direct connections between target accounts
        for i, account1 in enumerate(target_accounts):
            for account2 in target_accounts[i+1:]:
                if account1 in graph and account2 in graph:
                    if graph.has_edge(account1, account2) or graph.has_edge(account2, account1):
                        weight = 0
                        if graph.has_edge(account1, account2):
                            weight += graph[account1][account2].get('weight', 0)
                        if graph.has_edge(account2, account1):
                            weight += graph[account2][account1].get('weight', 0)
                        
                        relationships['direct_connections'].append({
                            'account1': account1,
                            'account2': account2,
                            'total_amount': weight,
                            'bidirectional': graph.has_edge(account1, account2) and graph.has_edge(account2, account1)
                        })
        
        # Common counterparties
        for i, account1 in enumerate(target_accounts):
            for account2 in target_accounts[i+1:]:
                if account1 in graph and account2 in graph:
                    neighbors1 = set(graph.neighbors(account1)) | set(graph.predecessors(account1))
                    neighbors2 = set(graph.neighbors(account2)) | set(graph.predecessors(account2))
                    
                    common = neighbors1 & neighbors2
                    if common:
                        relationships['common_counterparties'].append({
                            'account1': account1,
                            'account2': account2,
                            'common_accounts': list(common),
                            'common_count': len(common)
                        })
        
    except Exception as e:
        logger.warning(f"Relationship analysis failed: {e}")
        relationships['error'] = str(e)
    
    return relationships