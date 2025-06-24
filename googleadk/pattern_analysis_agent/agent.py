#!/usr/bin/env python3
"""
AML Pattern Analysis Agent - Google ADK Implementation (Production-Ready)
Advanced pattern analysis using pre-trained ML models and graph analytics
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter
import hashlib

# Google ADK imports
from google.adk.agents import Agent

# Import utility functions
from .utils import (
    extract_behavioral_features_for_accounts,
    extract_velocity_features_for_accounts,
    extract_network_features_for_accounts,
    detect_structuring_patterns,
    detect_layering_patterns,
    detect_round_tripping_patterns,
    detect_smurfing_patterns,
    detect_centrality_anomalies,
    detect_suspicious_flows,
    detect_money_mules,
    calculate_ml_risk_score,
    get_risk_level,
    get_counterparties,
    generate_pattern_insights,
    get_primary_concern,
    get_investigation_urgency,
    sum_total_anomalies,
    sum_typology_findings,
    calculate_overall_suspicion_level,
    assess_data_quality,
    assess_analysis_completeness,
    check_ctr_requirement,
    generate_compliance_actions,
    analyze_account_relationships
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Global Model Storage and Cache
# =============================================================================
TRAINED_MODELS = {}
SCALERS = {}
MODEL_METADATA = {}
NETWORK_GRAPH = None
TRANSACTION_DATA = None
AGENT_STATS = {
    'pattern_analyses_performed': 0,
    'networks_analyzed': 0,
    'anomalies_detected': 0,
    'typologies_identified': 0,
    'total_processing_time': 0.0,
    'models_loaded': False,
    'consensus_decisions': 0
}

# Known ML Typology Patterns (rule-based detection)
ML_TYPOLOGIES = {
    'STRUCTURING': {
        'description': 'Breaking large amounts into smaller transactions to avoid reporting',
        'threshold_amount': 10000,
        'time_window_hours': 24,
        'min_transactions': 3,
        'proximity_percentage': 10
    },
    'LAYERING': {
        'description': 'Complex series of transactions through multiple intermediaries to obscure money trail',
        'min_chain_length': 3,
        'max_chain_length': 8,
        'time_window_hours': 72,
        'amount_similarity_threshold': 0.15
    },
    'SMURFING': {
        'description': 'Using multiple accounts or individuals to conduct coordinated transactions',
        'coordination_window_minutes': 30,
        'min_coordinated_accounts': 3,
        'amount_similarity_threshold': 0.1,
        'timing_tolerance_minutes': 15
    },
    'ROUND_TRIPPING': {
        'description': 'Circular transaction flows that return funds to the originating account',
        'min_cycle_length': 2,
        'max_cycle_length': 6,
        'max_net_change_ratio': 0.1,
        'time_window_hours': 168
    }
}

# =============================================================================
# Model Loading Functions
# =============================================================================

def load_trained_models():
    """Load all pre-trained models from disk with corrected path resolution"""
    global TRAINED_MODELS, SCALERS, MODEL_METADATA, AGENT_STATS
    
    try:
        # Get model path from environment
        model_path = Path(os.getenv('PATTERN_MODEL_PATH', 'models/pattern_analysis'))
        
        # FIXED: Better path resolution logic
        if not model_path.is_absolute():
            # Start from the current file location
            current_file = Path(__file__).resolve()
            
            # Navigate to project root from googleadk/pattern_analysis_agent/agent.py
            # Go up: agent.py -> pattern_analysis_agent -> googleadk -> project_root
            project_root = current_file.parent.parent.parent
            
            # Construct the full path
            model_path = (project_root / model_path).resolve()
        
        logger.info(f"Loading pre-trained models from: {model_path}")
        
        if not model_path.exists():
            logger.error(f"Model directory not found: {model_path}")
            logger.error("Please train models first using scripts/train_pattern_models.py")
            AGENT_STATS['models_loaded'] = False
            return False
        
        # Load metadata
        metadata_path = model_path / 'model_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                MODEL_METADATA = json.load(f)
            logger.info("Loaded model metadata")
        else:
            logger.warning("Model metadata not found")
            MODEL_METADATA = {}
        
        # Load scalers
        scaler_dir = model_path / 'scalers'
        if scaler_dir.exists():
            for scaler_file in scaler_dir.glob('*.pkl'):
                scaler_name = scaler_file.stem.replace('_scaler', '')
                try:
                    SCALERS[scaler_name] = joblib.load(scaler_file)
                    logger.info(f"Loaded {scaler_name} scaler")
                except Exception as e:
                    logger.error(f"Failed to load scaler {scaler_name}: {e}")
        else:
            logger.error(f"Scalers directory not found: {scaler_dir}")
            return False
        
        # Load model groups
        model_types = ['amount_anomaly', 'behavioral_clustering', 'velocity_anomaly', 'network_anomaly']
        models_loaded = 0
        
        for model_type in model_types:
            model_type_dir = model_path / model_type
            if model_type_dir.exists():
                TRAINED_MODELS[model_type] = {}
                for model_file in model_type_dir.glob('*.pkl'):
                    model_name = model_file.stem
                    try:
                        TRAINED_MODELS[model_type][model_name] = joblib.load(model_file)
                        logger.info(f"Loaded {model_type}/{model_name}")
                        models_loaded += 1
                    except Exception as e:
                        logger.error(f"Failed to load {model_type}/{model_name}: {e}")
            else:
                logger.warning(f"Model type directory not found: {model_type_dir}")
        
        if models_loaded == 0:
            logger.error("No models were loaded successfully")
            logger.error("Please ensure models are trained and saved in the correct format")
            AGENT_STATS['models_loaded'] = False
            return False
        
        AGENT_STATS['models_loaded'] = True
        logger.info(f"Successfully loaded {models_loaded} models")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        AGENT_STATS['models_loaded'] = False
        return False

def load_transaction_data():
    """Load transaction data for analysis with corrected path resolution"""
    global TRANSACTION_DATA
    
    try:
        # Get data path from environment
        data_path = Path(os.getenv('TRANSACTION_DATA_PATH', 'data/processed/features/feature_enhanced_data.parquet'))
        
        # FIXED: Better path resolution logic
        if not data_path.is_absolute():
            # Start from the current file location
            current_file = Path(__file__).resolve()
            
            # Navigate to project root from googleadk/pattern_analysis_agent/agent.py
            # Go up: agent.py -> pattern_analysis_agent -> googleadk -> project_root
            project_root = current_file.parent.parent.parent
            
            # Construct the full path
            data_path = (project_root / data_path).resolve()
        
        logger.info(f"Loading transaction data from: {data_path}")
        
        if not data_path.exists():
            logger.error(f"Transaction data file not found: {data_path}")
            
            # Try alternative paths
            alternative_paths = [
                project_root / "data" / "processed" / "features" / "feature_enhanced_data.parquet",
                project_root / "data" / "processed" / "feature_enhanced_data.parquet",
                project_root / "data" / "feature_enhanced_data.parquet"
            ]
            
            for alt_path in alternative_paths:
                logger.info(f"Trying alternative path: {alt_path}")
                if alt_path.exists():
                    data_path = alt_path
                    logger.info(f"Found data at alternative path: {data_path}")
                    break
            else:
                logger.error("Data file not found in any expected location")
                logger.error(f"Project root: {project_root}")
                logger.error(f"Please ensure the data file exists at: {project_root / 'data' / 'processed' / 'features' / 'feature_enhanced_data.parquet'}")
                return False
        
        # Load the actual data
        TRANSACTION_DATA = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(TRANSACTION_DATA):,} transactions")
        logger.info(f"Columns available: {list(TRANSACTION_DATA.columns)}")
        
        # Handle date column - find and standardize
        date_columns = ['transaction_date', 'date', 'timestamp', 'tx_date', 'Date', 'TRANSACTION_DATE']
        date_column_found = None
        
        for col in date_columns:
            if col in TRANSACTION_DATA.columns:
                date_column_found = col
                break
        
        if date_column_found:
            try:
                # Convert to datetime
                TRANSACTION_DATA['transaction_date'] = pd.to_datetime(
                    TRANSACTION_DATA[date_column_found], 
                    errors='coerce'
                )
                
                # Check for invalid dates
                invalid_dates = TRANSACTION_DATA['transaction_date'].isna().sum()
                if invalid_dates > 0:
                    logger.warning(f"Found {invalid_dates} rows with invalid dates")
                    # Remove rows with invalid dates
                    TRANSACTION_DATA = TRANSACTION_DATA.dropna(subset=['transaction_date'])
                
                logger.info(f"Date column '{date_column_found}' standardized to 'transaction_date'")
                logger.info(f"Date range: {TRANSACTION_DATA['transaction_date'].min()} to {TRANSACTION_DATA['transaction_date'].max()}")
                
            except Exception as date_error:
                logger.error(f"Date conversion failed: {date_error}")
                logger.error("Transaction date column is required for temporal analysis")
                return False
        else:
            logger.error(f"No date column found in data. Available columns: {list(TRANSACTION_DATA.columns)}")
            logger.error("A date column is required for pattern analysis")
            return False
        
        # Verify required columns exist
        required_columns = ['originator_account', 'beneficiary_account', 'amount']
        missing_columns = [col for col in required_columns if col not in TRANSACTION_DATA.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            logger.error(f"Available columns: {list(TRANSACTION_DATA.columns)}")
            return False
        
        # Validate data types
        if not pd.api.types.is_numeric_dtype(TRANSACTION_DATA['amount']):
            logger.warning("Amount column is not numeric, attempting conversion")
            try:
                TRANSACTION_DATA['amount'] = pd.to_numeric(TRANSACTION_DATA['amount'], errors='coerce')
                # Remove rows with invalid amounts
                TRANSACTION_DATA = TRANSACTION_DATA.dropna(subset=['amount'])
            except Exception as e:
                logger.error(f"Failed to convert amount column to numeric: {e}")
                return False
        
        logger.info(f"Data validation successful: {len(TRANSACTION_DATA):,} transactions ready for analysis")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load transaction data: {e}")
        return False

def build_network_from_transactions(transactions: pd.DataFrame) -> nx.DiGraph:
    """Build network graph from transaction data"""
    
    G = nx.DiGraph()
    
    for _, row in transactions.iterrows():
        orig = row['originator_account']
        benef = row['beneficiary_account']
        amount = row['amount']
        
        if not G.has_node(orig):
            G.add_node(orig, total_sent=0, total_received=0, transaction_count=0)
        if not G.has_node(benef):
            G.add_node(benef, total_sent=0, total_received=0, transaction_count=0)
        
        if G.has_edge(orig, benef):
            G[orig][benef]['weight'] += amount
            G[orig][benef]['count'] += 1
        else:
            G.add_edge(orig, benef, weight=amount, count=1)
        
        G.nodes[orig]['total_sent'] += amount
        G.nodes[orig]['transaction_count'] += 1
        G.nodes[benef]['total_received'] += amount
        G.nodes[benef]['transaction_count'] += 1
    
    return G

# =============================================================================
# ML Inference Functions
# =============================================================================

def predict_amount_anomalies(amounts: np.ndarray) -> Dict[str, Any]:
    """Use pre-trained models to detect amount anomalies"""
    
    if 'amount_anomaly' not in TRAINED_MODELS or 'amount' not in SCALERS:
        logger.warning("Amount anomaly models not loaded")
        return {'error': 'Models not available'}
    
    models = TRAINED_MODELS['amount_anomaly']
    scaler = SCALERS['amount']
    results = {}
    
    try:
        # Prepare features
        log_amounts = np.log(amounts + 1)
        amount_features = np.column_stack([
            amounts,
            log_amounts,
            amounts / np.mean(amounts),
            (amounts - np.mean(amounts)) / np.std(amounts),
        ])
        
        # Scale features
        features_scaled = scaler.transform(amount_features)
        
        # Get predictions from each model
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict'):
                    predictions = model.predict(features_scaled)
                    outliers = np.where(predictions == -1)[0]
                else:
                    continue
                
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(features_scaled)
                elif hasattr(model, 'score_samples'):
                    scores = model.score_samples(features_scaled)
                else:
                    scores = predictions.astype(float)
                
                results[model_name] = {
                    'outlier_indices': outliers.tolist(),
                    'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / len(amounts)) * 100,
                    'outlier_amounts': amounts[outliers].tolist()[:10],  # Top 10
                    'mean_score': float(np.mean(scores))
                }
                
            except Exception as e:
                logger.warning(f"Amount anomaly prediction failed for {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # Create consensus results
        all_outlier_indices = set()
        for model_result in results.values():
            if 'outlier_indices' in model_result:
                all_outlier_indices.update(model_result['outlier_indices'])
        
        # Count how many models agree on each outlier
        outlier_consensus = {}
        for idx in all_outlier_indices:
            consensus_count = sum(1 for result in results.values() 
                                if 'outlier_indices' in result and idx in result['outlier_indices'])
            outlier_consensus[idx] = consensus_count
        
        # High confidence outliers (detected by multiple models)
        high_confidence_outliers = [idx for idx, count in outlier_consensus.items() if count >= 2]
        
        results['consensus_analysis'] = {
            'total_unique_outliers': len(all_outlier_indices),
            'high_confidence_outliers': len(high_confidence_outliers),
            'high_confidence_amounts': amounts[high_confidence_outliers].tolist()[:10],
            'outlier_consensus_scores': outlier_consensus
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Amount anomaly prediction failed: {e}")
        return {'error': str(e)}

def predict_behavioral_anomalies(behavioral_features: np.ndarray, account_names: List[str]) -> Dict[str, Any]:
    """Use pre-trained models to detect behavioral anomalies"""
    
    if 'behavioral_clustering' not in TRAINED_MODELS or 'behavioral' not in SCALERS:
        logger.warning("Behavioral models or scalers not loaded")
        return {'error': 'Models not available'}
    
    models = TRAINED_MODELS['behavioral_clustering']
    scaler = SCALERS['behavioral']
    
    try:
        # Scale features using trained scaler
        features_scaled = scaler.transform(behavioral_features)
        
        results = {}
        
        # DBSCAN clustering
        if 'dbscan' in models:
            cluster_labels = models['dbscan'].fit_predict(features_scaled)
            noise_accounts = [account_names[i] for i, label in enumerate(cluster_labels) if label == -1]
            
            results['dbscan'] = {
                'cluster_labels': cluster_labels.tolist(),
                'noise_accounts': noise_accounts,
                'noise_count': len(noise_accounts),
                'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            }
        
        # K-Means clustering
        if 'kmeans' in models:
            cluster_labels = models['kmeans'].predict(features_scaled)
            
            results['kmeans'] = {
                'cluster_labels': cluster_labels.tolist(),
                'n_clusters': models['kmeans'].n_clusters
            }
        
        # Isolation Forest anomalies
        if 'isolation_forest' in models:
            anomaly_predictions = models['isolation_forest'].predict(features_scaled)
            anomaly_scores = models['isolation_forest'].decision_function(features_scaled)
            
            anomalous_accounts = [account_names[i] for i, pred in enumerate(anomaly_predictions) if pred == -1]
            
            results['isolation_forest'] = {
                'anomalous_accounts': anomalous_accounts,
                'anomaly_count': len(anomalous_accounts),
                'anomaly_scores': anomaly_scores.tolist(),
                'mean_anomaly_score': float(np.mean(anomaly_scores))
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Behavioral anomaly prediction failed: {e}")
        return {'error': str(e)}

def predict_velocity_anomalies(velocity_features: np.ndarray, account_names: List[str]) -> Dict[str, Any]:
    """Use pre-trained models to detect velocity anomalies"""
    
    if 'velocity_anomaly' not in TRAINED_MODELS or 'velocity' not in SCALERS:
        logger.warning("Velocity models or scalers not loaded")
        return {'error': 'Models not available'}
    
    models = TRAINED_MODELS['velocity_anomaly']
    scaler = SCALERS['velocity']
    
    try:
        # Scale features
        features_scaled = scaler.transform(velocity_features)
        
        results = {}
        anomaly_methods = []
        
        # Test each model
        for model_name, model in models.items():
            if model_name == 'velocity_clustering':
                # Handle clustering separately
                cluster_labels = model.predict(features_scaled)
                results[model_name] = {
                    'cluster_labels': cluster_labels.tolist(),
                    'n_clusters': model.n_clusters
                }
                continue
            
            try:
                if hasattr(model, 'predict'):
                    predictions = model.predict(features_scaled)
                    anomalous_indices = np.where(predictions == -1)[0]
                    anomalous_accounts = [account_names[i] for i in anomalous_indices]
                    
                    if hasattr(model, 'decision_function'):
                        scores = model.decision_function(features_scaled)
                    else:
                        scores = predictions.astype(float)
                    
                    results[model_name] = {
                        'anomalous_accounts': anomalous_accounts,
                        'anomaly_count': len(anomalous_accounts),
                        'anomaly_scores': scores.tolist(),
                        'mean_score': float(np.mean(scores))
                    }
                    
                    anomaly_methods.append((model_name, set(anomalous_indices)))
                    
            except Exception as e:
                logger.warning(f"Velocity prediction failed for {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # Consensus analysis
        if len(anomaly_methods) >= 2:
            consensus_anomalies = set.intersection(*[indices for _, indices in anomaly_methods])
            consensus_accounts = [account_names[i] for i in consensus_anomalies]
            
            results['consensus_analysis'] = {
                'consensus_anomalies': len(consensus_anomalies),
                'consensus_accounts': consensus_accounts,
                'methods_used': len(anomaly_methods)
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Velocity anomaly prediction failed: {e}")
        return {'error': str(e)}

def predict_network_anomalies(network_features: np.ndarray, node_names: List[str]) -> Dict[str, Any]:
    """Use pre-trained models to detect network anomalies"""
    
    if 'network_anomaly' not in TRAINED_MODELS or 'network' not in SCALERS:
        logger.warning("Network models or scalers not loaded")
        return {'error': 'Models not available'}
    
    models = TRAINED_MODELS['network_anomaly']
    scaler = SCALERS['network']
    
    try:
        # Scale features
        features_scaled = scaler.transform(network_features)
        
        results = {}
        
        # PCA transformation
        if 'pca' in models:
            features_pca = models['pca'].transform(features_scaled)
            
            # Isolation Forest on PCA features
            if 'isolation_forest' in models:
                anomaly_predictions = models['isolation_forest'].predict(features_pca)
                anomaly_scores = models['isolation_forest'].decision_function(features_pca)
                
                anomalous_indices = np.where(anomaly_predictions == -1)[0]
                anomalous_nodes = [node_names[i] for i in anomalous_indices]
                
                results['pca_isolation_forest'] = {
                    'anomalous_nodes': anomalous_nodes,
                    'anomaly_count': len(anomalous_nodes),
                    'anomaly_scores': anomaly_scores.tolist(),
                    'mean_score': float(np.mean(anomaly_scores))
                }
        
        # Network clustering
        if 'network_clustering' in models:
            cluster_labels = models['network_clustering'].predict(features_scaled)
            
            results['network_clustering'] = {
                'cluster_labels': cluster_labels.tolist(),
                'n_clusters': models['network_clustering'].n_clusters
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Network anomaly prediction failed: {e}")
        return {'error': str(e)}

# =============================================================================
# ADK Tool Functions (Main Agent Capabilities)
# =============================================================================

def analyze_transaction_patterns(
    target_accounts: List[str],
    investigation_id: str,
    analysis_period_days: int = 90,
    include_network_analysis: bool = True,
    pattern_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze comprehensive transaction patterns for target accounts using ML models.
    """
    global TRANSACTION_DATA, NETWORK_GRAPH, AGENT_STATS
    
    start_time = datetime.now()
    
    try:
        # Ensure data is loaded
        if TRANSACTION_DATA is None or TRANSACTION_DATA.empty:
            success = load_transaction_data()
            if not success:
                return {
                    'investigation_id': investigation_id,
                    'error': 'Failed to load transaction data',
                    'analysis_timestamp': datetime.now().isoformat(),
                    'target_accounts': target_accounts
                }
        
        # Try to load models if not loaded
        if not AGENT_STATS['models_loaded']:
            success = load_trained_models()
            if not success:
                logger.warning("Models not available - providing basic analysis only")
        
        logger.info(f"Starting pattern analysis for {len(target_accounts)} accounts")
        logger.info(f"Investigation ID: {investigation_id}")
        logger.info(f"Analysis period: {analysis_period_days} days")
        
        # Default pattern types
        if pattern_types is None:
            pattern_types = ['amount', 'behavioral', 'velocity', 'network', 'typologies']
        
        # Filter transactions for analysis period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=analysis_period_days)
        
        # Ensure transaction_date column exists
        if 'transaction_date' not in TRANSACTION_DATA.columns:
            logger.error("transaction_date column not found in data")
            return {
                'investigation_id': investigation_id,
                'error': 'transaction_date column not found in data',
                'analysis_timestamp': datetime.now().isoformat(),
                'target_accounts': target_accounts
            }
        
        # Filter by date range
        try:
            relevant_transactions = TRANSACTION_DATA[
                (TRANSACTION_DATA['transaction_date'] >= start_date) &
                (TRANSACTION_DATA['transaction_date'] <= end_date)
            ].copy()
            logger.info(f"Filtered to {len(relevant_transactions)} transactions in date range")
        except Exception as filter_error:
            logger.error(f"Date filtering failed: {filter_error}")
            return {
                'investigation_id': investigation_id,
                'error': f'Date filtering failed: {filter_error}',
                'analysis_timestamp': datetime.now().isoformat(),
                'target_accounts': target_accounts
            }
        
        # Filter for target accounts
        target_transactions = relevant_transactions[
            (relevant_transactions['originator_account'].isin(target_accounts)) |
            (relevant_transactions['beneficiary_account'].isin(target_accounts))
        ]
        
        if len(target_transactions) == 0:
            logger.warning("No transactions found for target accounts in the specified period")
        
        analysis_results = {
            'investigation_id': investigation_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'target_accounts': target_accounts,
            'analysis_period_days': analysis_period_days,
            'pattern_types_performed': pattern_types,
            'transaction_summary': {},
            'ml_pattern_analysis': {},
            'typology_detection': {},
            'risk_assessment': {},
            'processing_metrics': {}
        }
        
        # Transaction summary
        all_counterparties = get_counterparties(relevant_transactions, target_accounts)
        
        analysis_results['transaction_summary'] = {
            'total_transactions': len(target_transactions),
            'total_amount': float(target_transactions['amount'].sum()) if len(target_transactions) > 0 else 0,
            'unique_counterparties': len(all_counterparties),
            'date_range': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            }
        }
        
        # Only proceed with ML analysis if models are loaded and we have sufficient data
        if AGENT_STATS['models_loaded'] and len(target_transactions) >= 10:
            
            # Amount anomaly analysis
            if 'amount' in pattern_types and 'amount' in target_transactions.columns:
                logger.info("Performing amount anomaly analysis...")
                amounts = target_transactions['amount'].values
                amount_anomalies = predict_amount_anomalies(amounts)
                analysis_results['ml_pattern_analysis']['amount_anomalies'] = amount_anomalies
                
                if 'consensus_analysis' in amount_anomalies:
                    AGENT_STATS['anomalies_detected'] += amount_anomalies['consensus_analysis'].get('high_confidence_outliers', 0)
            
            # Behavioral analysis
            if 'behavioral' in pattern_types:
                logger.info("Performing behavioral analysis...")
                try:
                    behavioral_features, valid_accounts = extract_behavioral_features_for_accounts(
                        relevant_transactions, target_accounts
                    )
                    
                    if len(behavioral_features) > 0:
                        behavioral_anomalies = predict_behavioral_anomalies(behavioral_features, valid_accounts)
                        analysis_results['ml_pattern_analysis']['behavioral_anomalies'] = behavioral_anomalies
                        
                        if 'isolation_forest' in behavioral_anomalies:
                            AGENT_STATS['anomalies_detected'] += behavioral_anomalies['isolation_forest'].get('anomaly_count', 0)
                except Exception as e:
                    logger.warning(f"Behavioral analysis failed: {e}")
                    analysis_results['ml_pattern_analysis']['behavioral_anomalies'] = {'error': str(e)}
            
            # Velocity analysis
            if 'velocity' in pattern_types:
                logger.info("Performing velocity analysis...")
                try:
                    velocity_features, valid_accounts = extract_velocity_features_for_accounts(
                        relevant_transactions, target_accounts
                    )
                    
                    if len(velocity_features) > 0:
                        velocity_anomalies = predict_velocity_anomalies(velocity_features, valid_accounts)
                        analysis_results['ml_pattern_analysis']['velocity_anomalies'] = velocity_anomalies
                        
                        if 'consensus_analysis' in velocity_anomalies:
                            AGENT_STATS['anomalies_detected'] += velocity_anomalies['consensus_analysis'].get('consensus_anomalies', 0)
                except Exception as e:
                    logger.warning(f"Velocity analysis failed: {e}")
                    analysis_results['ml_pattern_analysis']['velocity_anomalies'] = {'error': str(e)}
            
            # Network analysis
            if 'network' in pattern_types and include_network_analysis:
                logger.info("Performing network analysis...")
                try:
                    if NETWORK_GRAPH is None:
                        NETWORK_GRAPH = build_network_from_transactions(relevant_transactions)
                        AGENT_STATS['networks_analyzed'] += 1
                    
                    network_features, valid_accounts = extract_network_features_for_accounts(
                        NETWORK_GRAPH, target_accounts
                    )
                    
                    if len(network_features) > 0:
                        network_anomalies = predict_network_anomalies(network_features, valid_accounts)
                        analysis_results['ml_pattern_analysis']['network_anomalies'] = network_anomalies
                        
                        if 'pca_isolation_forest' in network_anomalies:
                            AGENT_STATS['anomalies_detected'] += network_anomalies['pca_isolation_forest'].get('anomaly_count', 0)
                except Exception as e:
                    logger.warning(f"Network analysis failed: {e}")
                    analysis_results['ml_pattern_analysis']['network_anomalies'] = {'error': str(e)}
        
        else:
            if not AGENT_STATS['models_loaded']:
                analysis_results['warning'] = 'ML models not loaded - basic analysis only'
                logger.warning("ML models not loaded - skipping ML-based analysis")
            if len(target_transactions) < 10:
                analysis_results['warning'] = f'Insufficient data ({len(target_transactions)} transactions) for ML analysis'
                logger.warning(f"Insufficient data for ML analysis: {len(target_transactions)} transactions")
        
        # Typology detection (rule-based, doesn't require trained models)
        if 'typologies' in pattern_types and len(target_transactions) >= 5:
            logger.info("Performing typology detection...")
            try:
                typology_results = {}
                
                # Structuring detection
                structuring_results = detect_structuring_patterns(
                    relevant_transactions, target_accounts, ML_TYPOLOGIES['STRUCTURING']
                )
                typology_results['structuring'] = structuring_results
                AGENT_STATS['typologies_identified'] += structuring_results.get('findings_count', 0)
                
                # Network-based typologies (if network available)
                if NETWORK_GRAPH is not None:
                    layering_results = detect_layering_patterns(
                        relevant_transactions, target_accounts, NETWORK_GRAPH, ML_TYPOLOGIES['LAYERING']
                    )
                    typology_results['layering'] = layering_results
                    AGENT_STATS['typologies_identified'] += layering_results.get('findings_count', 0)
                    
                    round_trip_results = detect_round_tripping_patterns(
                        relevant_transactions, target_accounts, NETWORK_GRAPH, ML_TYPOLOGIES['ROUND_TRIPPING']
                    )
                    typology_results['round_tripping'] = round_trip_results
                    AGENT_STATS['typologies_identified'] += round_trip_results.get('findings_count', 0)
                
                # Smurfing detection
                smurfing_results = detect_smurfing_patterns(
                    relevant_transactions, target_accounts, ML_TYPOLOGIES['SMURFING']
                )
                typology_results['smurfing'] = smurfing_results
                AGENT_STATS['typologies_identified'] += smurfing_results.get('findings_count', 0)
                
                analysis_results['typology_detection'] = typology_results
            except Exception as e:
                logger.warning(f"Typology detection failed: {e}")
                analysis_results['typology_detection'] = {'error': str(e)}
        
        # Risk assessment
        ml_analysis = analysis_results.get('ml_pattern_analysis', {})
        risk_assessment = calculate_ml_risk_score(ml_analysis)
        risk_assessment['risk_level'] = get_risk_level(risk_assessment['risk_score'])
        analysis_results['risk_assessment'] = risk_assessment
        
        # Processing metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        AGENT_STATS['pattern_analyses_performed'] += 1
        AGENT_STATS['total_processing_time'] += processing_time
        
        analysis_results['processing_metrics'] = {
            'processing_time_seconds': processing_time,
            'accounts_analyzed': len(target_accounts),
            'transactions_processed': len(target_transactions),
            'pattern_types_completed': len([pt for pt in pattern_types if pt in analysis_results.get('ml_pattern_analysis', {}) or pt == 'typologies'])
        }
        
        logger.info(f"Pattern analysis completed in {processing_time:.2f}s")
        logger.info(f"Risk score: {risk_assessment['risk_score']:.3f} ({risk_assessment['risk_level']})")
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'investigation_id': investigation_id,
            'error': str(e),
            'analysis_timestamp': datetime.now().isoformat(),
            'target_accounts': target_accounts
        }
        
def detect_network_anomalies(
    target_accounts: List[str],
    investigation_id: str,
    sensitivity: float = 0.1
) -> Dict[str, Any]:
    """
    Detect network-level anomalies in transaction patterns.
    
    Analyzes transaction networks to identify unusual connectivity patterns,
    suspicious flows, and potential money mule indicators.
    
    Args:
        target_accounts: List of account IDs to analyze
        investigation_id: Unique identifier for this investigation
        sensitivity: Anomaly detection sensitivity (0.01-0.5, lower = more sensitive)
        
    Returns:
        Dict containing network anomaly analysis results
    """
    global TRANSACTION_DATA, NETWORK_GRAPH, AGENT_STATS
    
    start_time = datetime.now()
    
    try:
        # Ensure data is loaded
        if TRANSACTION_DATA is None or TRANSACTION_DATA.empty:
            load_transaction_data()
        
        logger.info(f"Detecting network anomalies for {len(target_accounts)} accounts")
        logger.info(f"Investigation ID: {investigation_id}")
        logger.info(f"Sensitivity: {sensitivity}")
        
        # Build network if not exists
        if NETWORK_GRAPH is None:
            NETWORK_GRAPH = build_network_from_transactions(TRANSACTION_DATA)
            AGENT_STATS['networks_analyzed'] += 1
        
        network_results = {
            'investigation_id': investigation_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'target_accounts': target_accounts,
            'sensitivity': sensitivity,
            'network_statistics': {},
            'anomaly_detection': {},
            'suspicious_patterns': {},
            'processing_metrics': {}
        }
        
        # Network statistics
        network_results['network_statistics'] = {
            'total_nodes': NETWORK_GRAPH.number_of_nodes(),
            'total_edges': NETWORK_GRAPH.number_of_edges(),
            'network_density': nx.density(NETWORK_GRAPH),
            'target_nodes_in_network': len([acc for acc in target_accounts if acc in NETWORK_GRAPH])
        }
        
        # Centrality anomaly detection
        centrality_anomalies = detect_centrality_anomalies(NETWORK_GRAPH, target_accounts, sensitivity)
        network_results['anomaly_detection']['centrality_analysis'] = centrality_anomalies
        
        # Suspicious flow detection
        suspicious_flows = detect_suspicious_flows(NETWORK_GRAPH, TRANSACTION_DATA, target_accounts)
        network_results['suspicious_patterns']['flow_analysis'] = suspicious_flows
        
        # Money mule detection
        money_mule_indicators = detect_money_mules(NETWORK_GRAPH, TRANSACTION_DATA, target_accounts)
        network_results['suspicious_patterns']['money_mule_indicators'] = money_mule_indicators
        
        # Account relationship analysis
        relationship_analysis = analyze_account_relationships(target_accounts, investigation_id)
        network_results['suspicious_patterns']['relationship_analysis'] = relationship_analysis
        
        # Processing metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        network_results['processing_metrics'] = {
            'processing_time_seconds': processing_time,
            'nodes_analyzed': len(target_accounts),
            'network_size': NETWORK_GRAPH.number_of_nodes(),
            'anomalies_detected': len(centrality_anomalies.get('hub_accounts', []))
        }
        
        logger.info(f"Network anomaly detection completed in {processing_time:.2f}s")
        return network_results
        
    except Exception as e:
        logger.error(f"Network anomaly detection failed: {e}")
        return {
            'investigation_id': investigation_id,
            'error': str(e),
            'analysis_timestamp': datetime.now().isoformat(),
            'target_accounts': target_accounts
        }

def identify_ml_typologies(
    target_accounts: List[str],
    investigation_id: str,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Identify known money laundering typologies in transaction patterns.
    
    Detects specific ML patterns including structuring, layering, round-tripping,
    and smurfing using rule-based and pattern recognition techniques.
    
    Args:
        target_accounts: List of account IDs to analyze
        investigation_id: Unique identifier for this investigation
        confidence_threshold: Minimum confidence for positive detection
        
    Returns:
        Dict containing ML typology detection results
    """
    global TRANSACTION_DATA, NETWORK_GRAPH, AGENT_STATS
    
    start_time = datetime.now()
    
    try:
        # Ensure data is loaded
        if TRANSACTION_DATA is None or TRANSACTION_DATA.empty:
            load_transaction_data()
        
        logger.info(f"Identifying ML typologies for {len(target_accounts)} accounts")
        logger.info(f"Investigation ID: {investigation_id}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        
        # Build network if needed
        if NETWORK_GRAPH is None:
            NETWORK_GRAPH = build_network_from_transactions(TRANSACTION_DATA)
        
        typology_results = {
            'investigation_id': investigation_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'target_accounts': target_accounts,
            'confidence_threshold': confidence_threshold,
            'typology_findings': {},
            'summary': {},
            'processing_metrics': {}
        }
        
        # Detect each typology
        typology_findings = {}
        
        # Structuring detection
        logger.info("Detecting structuring patterns...")
        structuring_results = detect_structuring_patterns(
            TRANSACTION_DATA, target_accounts, ML_TYPOLOGIES['STRUCTURING']
        )
        typology_findings['structuring'] = structuring_results
        
        # Layering detection
        logger.info("Detecting layering patterns...")
        layering_results = detect_layering_patterns(
            TRANSACTION_DATA, target_accounts, NETWORK_GRAPH, ML_TYPOLOGIES['LAYERING']
        )
        typology_findings['layering'] = layering_results
        
        # Round-tripping detection
        logger.info("Detecting round-tripping patterns...")
        round_trip_results = detect_round_tripping_patterns(
            TRANSACTION_DATA, target_accounts, NETWORK_GRAPH, ML_TYPOLOGIES['ROUND_TRIPPING']
        )
        typology_findings['round_tripping'] = round_trip_results
        
        # Smurfing detection
        logger.info("Detecting smurfing patterns...")
        smurfing_results = detect_smurfing_patterns(
            TRANSACTION_DATA, target_accounts, ML_TYPOLOGIES['SMURFING']
        )
        typology_findings['smurfing'] = smurfing_results
        
        typology_results['typology_findings'] = typology_findings
        
        # Summary statistics
        total_findings = sum_typology_findings(typology_findings)
        high_confidence_typologies = len([
            name for name, results in typology_findings.items()
            if results.get('overall_confidence', 0) >= confidence_threshold
        ])
        
        typology_results['summary'] = {
            'total_typologies_tested': len(typology_findings),
            'total_findings': total_findings,
            'high_confidence_typologies': high_confidence_typologies,
            'overall_suspicion_level': calculate_overall_suspicion_level(typology_findings)
        }
        
        # Update agent statistics
        AGENT_STATS['typologies_identified'] += total_findings
        
        # Processing metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        typology_results['processing_metrics'] = {
            'processing_time_seconds': processing_time,
            'accounts_analyzed': len(target_accounts),
            'typologies_detected': high_confidence_typologies
        }
        
        logger.info(f"Typology detection completed in {processing_time:.2f}s")
        logger.info(f"Found {total_findings} total findings, {high_confidence_typologies} high-confidence")
        
        return typology_results
        
    except Exception as e:
        logger.error(f"Typology detection failed: {e}")
        return {
            'investigation_id': investigation_id,
            'error': str(e),
            'analysis_timestamp': datetime.now().isoformat(),
            'target_accounts': target_accounts
        }

def generate_pattern_insights(
    investigation_id: str,
    target_accounts: List[str],
    analysis_period_days: int = 90,
    include_recommendations: bool = True
) -> Dict[str, Any]:
    """
    Generate comprehensive insights and recommendations from pattern analysis.
    
    This tool runs a complete pattern analysis and then generates insights.
    
    Args:
        investigation_id: Unique identifier for this investigation
        target_accounts: List of account IDs to analyze
        analysis_period_days: Number of days to analyze
        include_recommendations: Whether to include actionable recommendations
        
    Returns:
        Dict containing comprehensive insights and recommendations
    """
    
    start_time = datetime.now()
    
    try:
        logger.info(f"Generating pattern insights for investigation {investigation_id}")
        
        # Step 1: Run comprehensive pattern analysis
        analysis_results = analyze_transaction_patterns(
            target_accounts=target_accounts,
            investigation_id=investigation_id,
            analysis_period_days=analysis_period_days,
            include_network_analysis=True,
            pattern_types=['amount', 'behavioral', 'velocity', 'network', 'typologies']
        )
        
        # Step 2: Generate insights using the utility functions
        from .utils import (
            generate_pattern_insights as utils_generate_insights,
            get_primary_concern,
            get_investigation_urgency,
            sum_total_anomalies,
            sum_typology_findings,
            calculate_overall_suspicion_level,
            assess_data_quality,
            assess_analysis_completeness,
            check_ctr_requirement,
            generate_compliance_actions
        )
        
        # Get basic insights from utils
        pattern_insights = utils_generate_insights(analysis_results)
        
        # Build comprehensive insights result
        risk_assessment = analysis_results.get('risk_assessment', {})
        ml_analysis = analysis_results.get('ml_pattern_analysis', {})
        typology_detection = analysis_results.get('typology_detection', {})
        
        insights_results = {
            'investigation_id': investigation_id,
            'generation_timestamp': datetime.now().isoformat(),
            'target_accounts': target_accounts,
            'analysis_period_days': analysis_period_days,
            'executive_summary': {
                'overall_risk_score': risk_assessment.get('risk_score', 0),
                'risk_level': risk_assessment.get('risk_level', 'UNKNOWN'),
                'primary_concern': get_primary_concern(analysis_results),
                'investigation_urgency': get_investigation_urgency(risk_assessment.get('risk_score', 0)),
                'total_anomalies': sum_total_anomalies(ml_analysis),
                'total_typology_findings': sum_typology_findings(typology_detection)
            },
            'key_findings': [
                {
                    'category': 'PATTERN_ANALYSIS',
                    'finding': insight,
                    'severity': 'HIGH' if 'high' in insight.lower() else 'MEDIUM'
                } for insight in pattern_insights
            ],
            'confidence_assessment': {
                'overall_confidence': assess_data_quality(analysis_results) * assess_analysis_completeness(analysis_results),
                'confidence_level': 'HIGH',
                'data_quality_score': assess_data_quality(analysis_results),
                'analysis_completeness': assess_analysis_completeness(analysis_results)
            },
            'recommendations': [],
            'regulatory_implications': {
                'sar_filing_recommended': risk_assessment.get('risk_score', 0) > 0.7,
                'ctr_requirement_check': check_ctr_requirement(analysis_results),
                'compliance_actions': generate_compliance_actions(analysis_results)
            }
        }
        
        # Add recommendations if requested
        if include_recommendations:
            risk_score = risk_assessment.get('risk_score', 0)
            if risk_score > 0.8:
                insights_results['recommendations'].extend([
                    "Immediate escalation to senior compliance officer recommended",
                    "Consider filing SAR within 30 days",
                    "Implement enhanced monitoring for all related accounts"
                ])
            elif risk_score > 0.6:
                insights_results['recommendations'].extend([
                    "Increase monitoring frequency for target accounts",
                    "Review account relationships and transaction patterns"
                ])
            
            # Add typology-specific recommendations
            for typology_name, results in typology_detection.items():
                if results.get('findings_count', 0) > 0 and results.get('overall_confidence', 0) > 0.7:
                    insights_results['recommendations'].append(
                        f"Investigate {typology_name.replace('_', ' ').title()} patterns identified with high confidence"
                    )
        
        # Processing metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        insights_results['processing_metrics'] = {
            'total_processing_time_seconds': processing_time,
            'insight_generation_time_seconds': processing_time,
            'findings_generated': len(insights_results['key_findings']),
            'recommendations_generated': len(insights_results.get('recommendations', []))
        }
        
        logger.info(f"Pattern insights generated in {processing_time:.2f}s")
        logger.info(f"Generated {len(insights_results['key_findings'])} findings and {len(insights_results.get('recommendations', []))} recommendations")
        
        return insights_results
        
    except Exception as e:
        logger.error(f"Pattern insight generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'investigation_id': investigation_id,
            'error': str(e),
            'generation_timestamp': datetime.now().isoformat()
        }

def get_pattern_agent_status() -> Dict[str, Any]:
    """
    Get comprehensive status of the Pattern Analysis Agent.
    
    Returns detailed information about agent health, model status,
    processing statistics, and operational capabilities.
    
    Returns:
        Dict containing complete agent status information
    """
    
    try:
        # Calculate performance metrics
        avg_processing_time = (
            AGENT_STATS['total_processing_time'] / max(AGENT_STATS['pattern_analyses_performed'], 1)
        )
        
        return {
            'agent_info': {
                'name': os.getenv('AGENT_NAME', 'PatternAnalysisAgent'),
                'version': os.getenv('AGENT_VERSION', '1.0.0'),
                'description': 'Advanced pattern analysis using ML models and graph analytics',
                'status': 'active'
            },
            'model_status': {
                'models_loaded': AGENT_STATS['models_loaded'],
                'model_categories': list(TRAINED_MODELS.keys()),
                'total_models': sum(len(models) for models in TRAINED_MODELS.values()),
                'scalers_loaded': len(SCALERS),
                'model_metadata': MODEL_METADATA
            },
            'data_status': {
                'transaction_data_loaded': TRANSACTION_DATA is not None,
                'transaction_count': len(TRANSACTION_DATA) if TRANSACTION_DATA is not None else 0,
                'network_graph_built': NETWORK_GRAPH is not None,
                'network_size': NETWORK_GRAPH.number_of_nodes() if NETWORK_GRAPH else 0
            },
            'processing_statistics': {
                'pattern_analyses_performed': AGENT_STATS['pattern_analyses_performed'],
                'networks_analyzed': AGENT_STATS['networks_analyzed'],
                'total_anomalies_detected': AGENT_STATS['anomalies_detected'],
                'total_typologies_identified': AGENT_STATS['typologies_identified'],
                'average_processing_time_seconds': avg_processing_time,
                'total_processing_time_seconds': AGENT_STATS['total_processing_time'],
                'consensus_decisions': AGENT_STATS['consensus_decisions']
            },
            'capabilities': [
                'ml_anomaly_detection',
                'behavioral_clustering',
                'velocity_analysis',
                'network_analysis',
                'typology_detection',
                'pattern_insights',
                'risk_scoring',
                'regulatory_assessment'
            ],
            'performance_metrics': {
                'system_health': 'healthy' if AGENT_STATS['models_loaded'] else 'degraded',
                'processing_efficiency': 'excellent' if avg_processing_time < 30 else 'good' if avg_processing_time < 60 else 'needs_optimization',
                'anomaly_detection_rate': AGENT_STATS['anomalies_detected'] / max(AGENT_STATS['pattern_analyses_performed'], 1),
                'typology_identification_rate': AGENT_STATS['typologies_identified'] / max(AGENT_STATS['pattern_analyses_performed'], 1)
            },
            'configuration': {
                'model_path': os.getenv('PATTERN_MODEL_PATH', 'models/pattern_analysis'),
                'data_path': os.getenv('TRANSACTION_DATA_PATH', '../../data/processed/features/feature_enhanced_data.parquet'),
                'anomaly_sensitivity': float(os.getenv('ANOMALY_SENSITIVITY', '0.1')),
                'typology_confidence_threshold': float(os.getenv('TYPOLOGY_CONFIDENCE_THRESHOLD', '0.7')),
                'max_processing_time': int(os.getenv('MAX_PROCESSING_TIME', '180')),
                'enable_network_analysis': os.getenv('ENABLE_NETWORK_ANALYSIS', 'true').lower() == 'true',
                'enable_caching': os.getenv('ENABLE_CACHING', 'true').lower() == 'true'
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'agent_info': {
                'name': 'PatternAnalysisAgent',
                'status': 'error',
                'error': str(e)
            },
            'timestamp': datetime.now().isoformat()
        }

# =============================================================================
# Google ADK Agent Definition
# =============================================================================

# Initialize data and models on startup
logger.info("Initializing Pattern Analysis Agent...")
load_transaction_data()
load_trained_models()

# Create the Google ADK Pattern Analysis Agent
pattern_analysis_agent = Agent(
    name=os.getenv('AGENT_NAME', 'PatternAnalysisAgent'),
    model=os.getenv('MODEL_NAME', 'gemini-2.0-flash-exp'),
    description="""
    I am an expert AML Pattern Analysis Agent that uses advanced machine learning models and graph analytics 
    to detect sophisticated money laundering patterns and transaction anomalies.
    
    My core capabilities include:
    
    🤖 ADVANCED ML ANOMALY DETECTION: I use ensemble machine learning models including Isolation Forest, 
    Local Outlier Factor, One-Class SVM, and clustering algorithms to detect amount, behavioral, velocity, 
    and network anomalies with high precision and confidence scoring.
    
    🕸️ NETWORK ANALYSIS: I build and analyze complex transaction networks to identify suspicious flows, 
    money mule indicators, centrality anomalies, and hidden relationships between accounts using graph 
    analytics and network science techniques.
    
    🕵️ ML TYPOLOGY DETECTION: I identify known money laundering patterns including structuring (smurfing), 
    layering, round-tripping, and coordinated transactions using sophisticated pattern recognition and 
    rule-based detection algorithms.
    
    📊 BEHAVIORAL ANALYTICS: I analyze account behavior patterns, transaction velocity changes, timing 
    anomalies, and cross-border activity to detect deviations from normal financial behavior.
    
    🎯 CONSENSUS ANALYSIS: I combine predictions from multiple ML models to provide high-confidence 
    anomaly detection with detailed reasoning and evidence for each finding.
    
    📈 COMPREHENSIVE INSIGHTS: I generate detailed investigative insights, risk assessments, regulatory 
    recommendations, and actionable intelligence for compliance teams and investigators.
    
    I work with pre-trained models for fast inference and can process complex transaction networks to 
    uncover the most sophisticated money laundering schemes while providing explainable AI results 
    for regulatory compliance.
    """,
    instruction="""
    You are an expert AML Pattern Analysis Agent specializing in advanced ML-based pattern detection. When users request pattern analysis:
    
    1. Use analyze_transaction_patterns for comprehensive multi-modal analysis including ML anomaly detection and typology identification
    2. Use detect_network_anomalies for focused network-level analysis of suspicious connectivity and flows
    3. Use identify_ml_typologies for specific money laundering pattern detection (structuring, layering, etc.)
    4. Use generate_pattern_insights for synthesizing findings into actionable intelligence and recommendations
    5. Use get_pattern_agent_status for system health and capability information
    
    Always provide detailed explanations including:
    - ML model consensus and confidence levels
    - Specific anomalies detected and their significance
    - Network analysis findings and relationship patterns
    - Typology detection results with evidence
    - Risk assessments and regulatory implications
    - Clear recommendations for investigators
    
    Focus on providing explainable AI results that help investigators understand why patterns are suspicious and what actions to take.
    """,
    tools=[
        analyze_transaction_patterns,
        detect_network_anomalies, 
        identify_ml_typologies,
        generate_pattern_insights,
        get_pattern_agent_status
    ]
)