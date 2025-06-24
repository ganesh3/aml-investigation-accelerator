#!/usr/bin/env python3
"""
Advanced Pattern Analysis Model Training for AML Investigations
Trains specialized ML models for anomaly detection and pattern recognition
"""

import pandas as pd
import numpy as np
import joblib
import json
import networkx as nx
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AMLPatternModelTrainer:
    """
    Comprehensive trainer for AML pattern analysis models
    
    Trains four types of specialized models:
    1. Amount Anomaly Detection
    2. Behavioral Pattern Clustering  
    3. Velocity Anomaly Detection
    4. Network Anomaly Detection
    """
    
    def __init__(self, data_path: str = r'D:\Google-ADK-Project\data\processed\features\feature_enhanced_data.parquet'):
        self.data_path = Path(data_path)
        self.models = {
            'amount_anomaly': {},
            'behavioral_clustering': {},
            'velocity_anomaly': {},
            'network_anomaly': {}
        }
        self.scalers = {}
        self.metadata = {}
        self.network_graph = None
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Load and prepare data for pattern analysis training"""
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        logger.info(f"Loaded {len(df):,} transactions")
        
        # Handle missing columns
        required_columns = ['transaction_id', 'originator_account', 'beneficiary_account', 'amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            logger.info(f"Available columns: {list(df.columns)}")
            
            # Create synthetic required columns if missing
            if 'transaction_id' not in df.columns:
                df['transaction_id'] = [f"TXN_{i:06d}" for i in range(len(df))]
            if 'originator_account' not in df.columns:
                df['originator_account'] = [f"ACC_{i%1000:05d}" for i in range(len(df))]
            if 'beneficiary_account' not in df.columns:
                df['beneficiary_account'] = [f"ACC_{(i+500)%1000:05d}" for i in range(len(df))]
        
        # Create transaction_date if missing
        if 'transaction_date' not in df.columns:
            logger.warning("Creating synthetic transaction_date column")
            start_date = datetime.now() - timedelta(days=365)
            df['transaction_date'] = [start_date + timedelta(days=np.random.randint(0, 365)) 
                                    for _ in range(len(df))]
        
        # Ensure transaction_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['transaction_date']):
            df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        
        # Get unique accounts
        accounts = list(set(df['originator_account'].unique()) | set(df['beneficiary_account'].unique()))
        logger.info(f"Found {len(accounts)} unique accounts")
        
        return df, np.array(df['amount']), accounts
    
    def extract_amount_features(self, amounts: np.ndarray) -> np.ndarray:
        """Extract features for amount anomaly detection"""
        
        # Basic amount features
        log_amounts = np.log(amounts + 1)
        
        # Statistical features
        amount_features = np.column_stack([
            amounts,
            log_amounts,
            amounts / np.mean(amounts),  # Normalized amounts
            (amounts - np.mean(amounts)) / np.std(amounts),  # Z-scores
        ])
        
        return amount_features
    
    def extract_behavioral_features(self, df: pd.DataFrame, accounts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Extract behavioral features for each account"""
        
        features = []
        valid_accounts = []
        
        logger.info("Extracting behavioral features...")
        
        for account in accounts[:500]:  # Limit for training efficiency
            # Get all transactions for this account
            account_txns = df[
                (df['originator_account'] == account) |
                (df['beneficiary_account'] == account)
            ]
            
            if len(account_txns) < 5:  # Skip accounts with too few transactions
                continue
            
            # Separate outgoing and incoming
            outgoing = account_txns[account_txns['originator_account'] == account]
            incoming = account_txns[account_txns['beneficiary_account'] == account]
            
            # Calculate metrics
            total_out = float(outgoing['amount'].sum()) if len(outgoing) > 0 else 0
            total_in = float(incoming['amount'].sum()) if len(incoming) > 0 else 0
            count_out = len(outgoing)
            count_in = len(incoming)
            
            if total_out == 0 and total_in == 0:
                continue
            
            # Flow ratios (with safety checks)
            flow_ratio = total_out / max(total_in, 1) if total_in > 0 else (100 if total_out > 0 else 1)
            count_ratio = count_out / max(count_in, 1) if count_in > 0 else (100 if count_out > 0 else 1)
            
            # Transaction characteristics
            avg_amount = (total_out + total_in) / max(count_out + count_in, 1)
            
            # Cross-border activity
            cross_border_ratio = float(account_txns['cross_border'].mean()) if 'cross_border' in account_txns.columns else 0
            
            # Counterparty diversity
            counterparties = set()
            counterparties.update(outgoing['beneficiary_account'].unique())
            counterparties.update(incoming['originator_account'].unique())
            counterparties.discard(account)
            
            # Risk indicators
            high_risk_ratio = float(account_txns['risk_score'].mean()) if 'risk_score' in account_txns.columns else 0
            
            # Time span analysis
            time_span_days = (account_txns['transaction_date'].max() - account_txns['transaction_date'].min()).days
            
            feature_vector = [
                np.log(total_out + total_in + 1),  # Total volume (log)
                np.log(count_out + count_in + 1),  # Total count (log)
                np.log(flow_ratio + 1),           # Flow imbalance
                np.log(count_ratio + 1),          # Count imbalance  
                np.log(avg_amount + 1),           # Average transaction size
                cross_border_ratio,               # Cross-border activity
                len(counterparties),              # Counterparty count
                high_risk_ratio,                  # Risk level
                time_span_days,                   # Activity period
                len(account_txns)                 # Transaction frequency
            ]
            
            features.append(feature_vector)
            valid_accounts.append(account)
        
        logger.info(f"Extracted features for {len(valid_accounts)} accounts")
        return np.array(features), valid_accounts
    
    def extract_velocity_features(self, df: pd.DataFrame, accounts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Extract velocity and temporal pattern features with NaN handling"""
    
        features = []
        valid_accounts = []
        
        logger.info("Extracting velocity features...")
        
        for account in accounts[:300]:  # Limit for efficiency
            account_txns = df[
                (df['originator_account'] == account) |
                (df['beneficiary_account'] == account)
            ].sort_values('transaction_date')
            
            if len(account_txns) < 10:  # Need sufficient data for velocity analysis
                continue
            
            try:
                # Daily transaction patterns
                daily_counts = account_txns.groupby(account_txns['transaction_date'].dt.date).size()
                daily_amounts = account_txns.groupby(account_txns['transaction_date'].dt.date)['amount'].sum()
                
                # Hourly patterns
                hourly_counts = account_txns.groupby(account_txns['transaction_date'].dt.hour).size()
                
                # Time between transactions
                time_diffs = account_txns['transaction_date'].diff().dt.total_seconds().dropna()
                
                if len(daily_counts) == 0 or len(time_diffs) == 0:
                    continue
                
                # Weekend vs weekday activity
                account_txns['day_of_week'] = account_txns['transaction_date'].dt.dayofweek
                weekend_ratio = float((account_txns['day_of_week'] >= 5).mean())
                
                # Business hours vs off-hours
                business_hours_ratio = float(((account_txns['transaction_date'].dt.hour >= 9) & 
                                            (account_txns['transaction_date'].dt.hour <= 17)).mean())
                
                # FIXED: Handle NaN values and edge cases
                feature_vector = [
                    np.log(len(account_txns) + 1),                                    # Total transactions
                    float(daily_counts.max() if len(daily_counts) > 0 else 1),       # Peak daily activity
                    float(daily_counts.mean() if len(daily_counts) > 0 else 1),      # Average daily activity
                    float(daily_counts.std() if len(daily_counts) > 1 else 0),       # Daily activity variation
                    np.log(daily_amounts.max() + 1 if len(daily_amounts) > 0 else 1), # Peak daily amount
                    float(daily_amounts.mean() if len(daily_amounts) > 0 else 1),    # Average daily amount
                    int((daily_counts > daily_counts.quantile(0.95)).sum() if len(daily_counts) > 1 else 0),  # High activity days
                    int((time_diffs < 3600).sum() if len(time_diffs) > 0 else 0),    # Rapid succession transactions
                    np.log(time_diffs.mean() + 1 if len(time_diffs) > 0 else 1),     # Average time between transactions
                    float(time_diffs.std() / max(time_diffs.mean(), 1) if len(time_diffs) > 1 and time_diffs.mean() > 0 else 0),  # Time variability
                    len(daily_counts) if len(daily_counts) > 0 else 1,               # Active days
                    float(hourly_counts.std() if len(hourly_counts) > 1 else 0),     # Hourly variation
                    weekend_ratio if not np.isnan(weekend_ratio) else 0,             # Weekend activity
                    business_hours_ratio if not np.isnan(business_hours_ratio) else 0.5  # Business hours activity
                ]
                
                # CRITICAL: Check for NaN values before adding
                if any(np.isnan(val) or np.isinf(val) for val in feature_vector):
                    logger.warning(f"Skipping account {account} due to NaN/Inf values")
                    continue
                
                features.append(feature_vector)
                valid_accounts.append(account)
                
            except Exception as e:
                logger.warning(f"Failed to extract velocity features for {account}: {e}")
                continue
        
        logger.info(f"Extracted velocity features for {len(valid_accounts)} accounts")
        
        # Final NaN check on the entire array
        features_array = np.array(features)
        if len(features_array) > 0:
            nan_mask = np.isnan(features_array).any(axis=1)
            if nan_mask.any():
                logger.warning(f"Removing {nan_mask.sum()} accounts with NaN features")
                features_array = features_array[~nan_mask]
                valid_accounts = [acc for i, acc in enumerate(valid_accounts) if not nan_mask[i]]
        
        return features_array, valid_accounts
    
    def extract_network_features(self, df: pd.DataFrame, accounts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Extract network-based features"""
        
        logger.info("Building transaction network...")
        
        # Build network graph
        G = nx.DiGraph()
        
        for _, row in df.iterrows():
            orig = row['originator_account']
            benef = row['beneficiary_account']
            amount = row['amount']
            
            if not G.has_node(orig):
                G.add_node(orig, total_sent=0, total_received=0, tx_count=0)
            if not G.has_node(benef):
                G.add_node(benef, total_sent=0, total_received=0, tx_count=0)
            
            if G.has_edge(orig, benef):
                G[orig][benef]['weight'] += amount
                G[orig][benef]['count'] += 1
            else:
                G.add_edge(orig, benef, weight=amount, count=1)
            
            G.nodes[orig]['total_sent'] += amount
            G.nodes[orig]['tx_count'] += 1
            G.nodes[benef]['total_received'] += amount
            G.nodes[benef]['tx_count'] += 1
        
        self.network_graph = G
        logger.info(f"Built network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Calculate centrality measures (efficient batch calculation)
        degree_centrality = nx.degree_centrality(G)
        
        # Use a subset for expensive centrality calculations
        subgraph_nodes = list(G.nodes())[:1000] if len(G.nodes()) > 1000 else list(G.nodes())
        subgraph = G.subgraph(subgraph_nodes)
        
        try:
            closeness_centrality = nx.closeness_centrality(subgraph)
            betweenness_centrality = nx.betweenness_centrality(subgraph, k=min(100, len(subgraph)))
        except:
            closeness_centrality = {node: 0 for node in subgraph_nodes}
            betweenness_centrality = {node: 0 for node in subgraph_nodes}
        
        # Clustering coefficient
        clustering = nx.clustering(G.to_undirected())
        
        # Core numbers
        try:
            core_numbers = nx.core_number(G.to_undirected())
        except:
            core_numbers = {node: 0 for node in G.nodes()}
        
        # Extract features for each account
        features = []
        valid_accounts = []
        
        target_accounts = [acc for acc in accounts if acc in G.nodes()][:500]  # Limit for efficiency
        
        for account in target_accounts:
            try:
                feature_vector = [
                    G.degree(account),                                    # Total degree
                    G.in_degree(account),                                # In-degree
                    G.out_degree(account),                               # Out-degree
                    clustering.get(account, 0),                          # Clustering coefficient
                    core_numbers.get(account, 0),                        # Core number
                    degree_centrality[account],                          # Degree centrality
                    closeness_centrality.get(account, 0),               # Closeness centrality
                    betweenness_centrality.get(account, 0),             # Betweenness centrality
                    np.log(G.nodes[account]['total_sent'] + 1),         # Total sent (log)
                    np.log(G.nodes[account]['total_received'] + 1),     # Total received (log)
                    G.nodes[account]['tx_count'],                        # Transaction count
                    len(list(G.neighbors(account))),                     # Neighbor count
                ]
                
                features.append(feature_vector)
                valid_accounts.append(account)
                
            except Exception as e:
                logger.warning(f"Failed to extract network features for {account}: {e}")
                continue
        
        logger.info(f"Extracted network features for {len(valid_accounts)} accounts")
        return np.array(features), valid_accounts
    
    def train_amount_anomaly_models(self, amount_features: np.ndarray):
        """Train amount anomaly detection models"""
        
        logger.info("Training amount anomaly detection models...")
        
        # Scale features
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(amount_features)
        self.scalers['amount'] = scaler
        
        # Train multiple anomaly detection models
        models = {
            'isolation_forest': IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            ),
            'local_outlier_factor': LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.1,
                novelty=True
            ),
            'elliptic_envelope': EllipticEnvelope(
                contamination=0.1,
                random_state=42
            ),
            'one_class_svm': OneClassSVM(
                kernel='rbf',
                gamma='scale',
                nu=0.1
            )
        }
        
        for name, model in models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(features_scaled)
                self.models['amount_anomaly'][name] = model
                logger.info(f"✅ {name} trained successfully")
            except Exception as e:
                logger.error(f"❌ Failed to train {name}: {e}")
    
    def train_behavioral_clustering_models(self, behavioral_features: np.ndarray):
        """Train behavioral clustering models"""
        
        logger.info("Training behavioral clustering models...")
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(behavioral_features)
        self.scalers['behavioral'] = scaler
        
        # DBSCAN clustering
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan.fit(features_scaled)
            self.models['behavioral_clustering']['dbscan'] = dbscan
            logger.info(f"✅ DBSCAN trained - found {len(set(dbscan.labels_))} clusters")
        except Exception as e:
            logger.error(f"❌ DBSCAN training failed: {e}")
        
        # K-Means clustering
        try:
            kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
            kmeans.fit(features_scaled)
            self.models['behavioral_clustering']['kmeans'] = kmeans
            logger.info(f"✅ K-Means trained with 8 clusters")
        except Exception as e:
            logger.error(f"❌ K-Means training failed: {e}")
        
        # Isolation Forest for behavioral anomalies
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(features_scaled)
            self.models['behavioral_clustering']['isolation_forest'] = iso_forest
            logger.info(f"✅ Behavioral Isolation Forest trained")
        except Exception as e:
            logger.error(f"❌ Behavioral Isolation Forest training failed: {e}")
    
    def train_velocity_anomaly_models(self, velocity_features: np.ndarray):
        """Train velocity anomaly detection models with enhanced NaN handling"""
    
        logger.info("Training velocity anomaly detection models...")
        logger.info(f"Velocity features shape: {velocity_features.shape}")
        
        if len(velocity_features) == 0:
            logger.error("❌ No velocity features available for training!")
            return
        
        # CRITICAL: Final NaN check before training
        if np.isnan(velocity_features).any():
            logger.error("❌ NaN values detected in velocity features!")
            logger.info("Cleaning NaN values...")
            
            # Remove rows with any NaN
            clean_mask = ~np.isnan(velocity_features).any(axis=1)
            velocity_features = velocity_features[clean_mask]
            
            if len(velocity_features) == 0:
                logger.error("❌ No clean velocity features remaining after NaN removal!")
                return
            
            logger.info(f"✅ Cleaned features shape: {velocity_features.shape}")
        
        if len(velocity_features) < 50:
            logger.warning(f"⚠️ Limited velocity features ({len(velocity_features)}), results may be poor")
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(velocity_features)
        self.scalers['velocity'] = scaler
        logger.info(f"✅ Velocity scaler fitted with {features_scaled.shape[0]} samples")
        
        # Final check after scaling
        if np.isnan(features_scaled).any():
            logger.error("❌ NaN values present after scaling!")
            return
        
        # Velocity-specific anomaly models with adjusted parameters
        models = {
            'velocity_isolation_forest': IsolationForest(
                contamination=0.15,
                random_state=42,
                n_estimators=100
            ),
            'velocity_lof': LocalOutlierFactor(
                n_neighbors=min(15, len(velocity_features)-1),  # Ensure n_neighbors < n_samples
                contamination=0.15,
                novelty=True
            ),
            'velocity_one_class_svm': OneClassSVM(
                kernel='rbf',
                gamma='auto',
                nu=0.15
            ),
            'velocity_clustering': KMeans(
                n_clusters=min(6, len(velocity_features)),  # Ensure k <= n_samples
                random_state=42,
                n_init=10
            )
        }
        
        # Track successful models
        successful_models = 0
        
        for name, model in models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(features_scaled)
                self.models['velocity_anomaly'][name] = model
                successful_models += 1
                logger.info(f"✅ {name} trained successfully")
            except Exception as e:
                logger.error(f"❌ Failed to train {name}: {e}")
                # Continue with other models
        
        logger.info(f"Velocity training complete: {successful_models}/{len(models)} models successful")
        
        if successful_models == 0:
            logger.error("❌ All velocity models failed to train!")
        else:
            logger.info(f"✅ Velocity anomaly models ready: {list(self.models['velocity_anomaly'].keys())}")

    
    def train_network_anomaly_models(self, network_features: np.ndarray):
        """Train network anomaly detection models"""
        
        logger.info("Training network anomaly detection models...")
        
        if len(network_features) == 0:
            logger.warning("No network features available for training")
            return
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(network_features)
        self.scalers['network'] = scaler
        
        # PCA for dimensionality reduction
        try:
            pca = PCA(n_components=min(8, features_scaled.shape[1]), random_state=42)
            features_pca = pca.fit_transform(features_scaled)
            self.models['network_anomaly']['pca'] = pca
            logger.info(f"✅ PCA trained - reduced to {pca.n_components_} components")
        except Exception as e:
            logger.error(f"❌ PCA training failed: {e}")
            features_pca = features_scaled
        
        # Isolation Forest on PCA features
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(features_pca)
            self.models['network_anomaly']['isolation_forest'] = iso_forest
            logger.info(f"✅ Network Isolation Forest trained")
        except Exception as e:
            logger.error(f"❌ Network Isolation Forest training failed: {e}")
        
        # Network clustering
        try:
            clustering = KMeans(n_clusters=5, random_state=42, n_init=10)
            clustering.fit(features_scaled)
            self.models['network_anomaly']['network_clustering'] = clustering
            logger.info(f"✅ Network clustering trained")
        except Exception as e:
            logger.error(f"❌ Network clustering training failed: {e}")
    
    def save_models(self):
        """Enhanced model saving with better error handling"""
    
        # Create model directories
        model_root = Path('models/pattern_analysis')
        model_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Model root directory: {model_root.absolute()}")
        
        # Save scalers
        scaler_dir = model_root / 'scalers'
        scaler_dir.mkdir(exist_ok=True)
        
        for scaler_name, scaler in self.scalers.items():
            scaler_path = scaler_dir / f'{scaler_name}_scaler.pkl'
            try:
                joblib.dump(scaler, scaler_path)
                logger.info(f"✅ Saved {scaler_name} scaler to {scaler_path}")
            except Exception as e:
                logger.error(f"❌ Failed to save {scaler_name} scaler: {e}")
        
        # Save models by category with enhanced logging
        total_models_saved = 0
        
        for category, models in self.models.items():
            logger.info(f"📂 Processing category: {category}")
            logger.info(f"   Models in category: {list(models.keys()) if models else 'None'}")
            
            if not models:
                logger.warning(f"⚠️ No models to save for category: {category}")
                continue
                
            category_dir = model_root / category
            category_dir.mkdir(exist_ok=True)
            logger.info(f"📁 Created directory: {category_dir}")
            
            for model_name, model in models.items():
                model_path = category_dir / f'{model_name}.pkl'
                try:
                    joblib.dump(model, model_path)
                    total_models_saved += 1
                    logger.info(f"✅ Saved {category}/{model_name} to {model_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to save {category}/{model_name}: {e}")
        
        logger.info(f"📊 Total models saved: {total_models_saved}")
        
        # Enhanced metadata with debugging info
        self.metadata = {
            'training_timestamp': datetime.now().isoformat(),
            'model_categories': list(self.models.keys()),
            'models_trained': {
                category: list(models.keys()) 
                for category, models in self.models.items()
            },
            'model_counts': {
                category: len(models) 
                for category, models in self.models.items()
            },
            'scalers_saved': list(self.scalers.keys()),
            'total_models_saved': total_models_saved,
            'velocity_models': list(self.models.get('velocity_anomaly', {}).keys()),
            'network_stats': {
                'nodes': self.network_graph.number_of_nodes() if self.network_graph else 0,
                'edges': self.network_graph.number_of_edges() if self.network_graph else 0
            } if self.network_graph else {},
            'training_config': {
                'amount_anomaly_contamination': 0.1,
                'behavioral_clusters': 8,
                'velocity_contamination': 0.15,
                'network_pca_components': 8,
                'network_clusters': 5
            }
        }
        
        metadata_path = model_root / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"✅ Metadata saved to {metadata_path}")
        
        # Verify what was actually saved
        logger.info("🔍 Verifying saved files:")
        for category_dir in model_root.iterdir():
            if category_dir.is_dir():
                pkl_files = list(category_dir.glob('*.pkl'))
                logger.info(f"   {category_dir.name}/: {len(pkl_files)} files")
                for pkl_file in pkl_files:
                    logger.info(f"      - {pkl_file.name}")
    
        return {
            'models_saved': total_models_saved,
            'scalers_saved': len(self.scalers),
            'model_path': str(model_root)
        }
    
    def train_all_models(self):
        """Main training pipeline"""
        
        logger.info("🚀 Starting comprehensive pattern analysis model training...")
        
        try:
            # Load data
            df, amounts, accounts = self.load_and_prepare_data()
            
            # Extract features for different model types
            logger.info("📊 Extracting features...")
            
            # Amount features
            amount_features = self.extract_amount_features(amounts)
            logger.info(f"Amount features shape: {amount_features.shape}")
            
            # Behavioral features
            behavioral_features, behavioral_accounts = self.extract_behavioral_features(df, accounts)
            logger.info(f"Behavioral features shape: {behavioral_features.shape}")
            
            # Velocity features  
            velocity_features, velocity_accounts = self.extract_velocity_features(df, accounts)
            logger.info(f"Velocity features shape: {velocity_features.shape}")
            
            # Network features
            network_features, network_accounts = self.extract_network_features(df, accounts)
            logger.info(f"Network features shape: {network_features.shape}")
            
            # Train models
            logger.info("🤖 Training ML models...")
            
            self.train_amount_anomaly_models(amount_features)
            self.train_behavioral_clustering_models(behavioral_features)
            self.train_velocity_anomaly_models(velocity_features)
            self.train_network_anomaly_models(network_features)
            
            # Save everything
            logger.info("💾 Saving models...")
            save_results = self.save_models()
            
            logger.info("🎉 Pattern analysis model training completed successfully!")
            
            return {
                'success': True,
                'models_trained': save_results['models_saved'],
                'scalers_saved': save_results['scalers_saved'],
                'model_path': save_results['model_path'],
                'data_summary': {
                    'total_transactions': len(df),
                    'unique_accounts': len(accounts),
                    'behavioral_accounts_analyzed': len(behavioral_accounts),
                    'velocity_accounts_analyzed': len(velocity_accounts),
                    'network_accounts_analyzed': len(network_accounts)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

def main():
    """Main training function"""
    print("🧠 AML Pattern Analysis Model Training")
    print("="*60)
    
    try:
        # Initialize trainer
        trainer = AMLPatternModelTrainer()
        
        # Run training
        results = trainer.train_all_models()
        
        if results['success']:
            print(f"\n🎉 Training Summary:")
            print(f"   Models trained: {results['models_trained']}")
            print(f"   Scalers saved: {results['scalers_saved']}")
            print(f"   Model path: {results['model_path']}")
            print(f"\n📊 Data Summary:")
            print(f"   Total transactions: {results['data_summary']['total_transactions']:,}")
            print(f"   Unique accounts: {results['data_summary']['unique_accounts']:,}")
            print(f"   Behavioral accounts: {results['data_summary']['behavioral_accounts_analyzed']:,}")
            print(f"   Velocity accounts: {results['data_summary']['velocity_accounts_analyzed']:,}")
            print(f"   Network accounts: {results['data_summary']['network_accounts_analyzed']:,}")
            
            print(f"\n✅ Pattern analysis models ready for production!")
            return True
        else:
            print(f"\n❌ Training failed: {results['error']}")
            return False
            
    except Exception as e:
        print(f"\n❌ Training failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Training failed. Check the error messages above.")
    else:
        print("\n🚀 Ready to use pattern analysis models in the Pattern Analysis Agent!")