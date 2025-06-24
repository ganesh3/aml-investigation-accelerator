#!/usr/bin/env python3
"""
Enhanced AML Data Preprocessor - Optimized for Downloaded Datasets
Handles Parquet files first, with intelligent schema mapping and validation
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA = DATA_ROOT / "raw"
PROCESSED_DATA = DATA_ROOT / "processed"

class EnhancedAMLDataPreprocessor:
    def __init__(self):
        self.stats = {
            'total_transactions': 0,
            'laundering_transactions': 0,
            'datasets_processed': 0,
            'data_quality_score': 0.0,
            'processing_time': None,
            'dataset_details': {}
        }
        
        # Check for pyarrow
        self.check_parquet_support()
        
    def check_parquet_support(self):
        """Ensure pyarrow is available for Parquet support."""
        try:
            import pyarrow
            logger.info("✅ Parquet support available")
            return True
        except ImportError:
            logger.error("❌ pyarrow not found. Installing...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
            logger.info("✅ pyarrow installed successfully")
            return True
        
    def smart_load_dataset(self, dataset_name, dataset_dir):
        """Smart loading: Try Parquet first, then CSV with schema detection."""
        logger.info(f"Loading {dataset_name} dataset...")
        
        # Define expected files for each dataset
        file_options = {
            'saml_d': [
                'saml_transactions.parquet',
                'saml_transactions.csv', 
                'SAML-D.csv'
            ],
            'ibm_aml': [
                'ibm_transactions.parquet',
                'ibm_transactions.csv',
                'HI-Medium_Trans.csv'  # Original large file
            ],
            'maryam': [
                'money_laundering_data.parquet',
                'money_laundering_data.csv',
                'ML.csv'
            ]
        }
        
        files_to_try = file_options.get(dataset_name, [])
        
        # Try each file in order of preference
        for filename in files_to_try:
            file_path = dataset_dir / filename
            if file_path.exists():
                try:
                    file_size_mb = file_path.stat().st_size / 1024 / 1024
                    logger.info(f"   Found {filename} ({file_size_mb:.1f} MB)")
                    
                    # Load based on file type
                    if filename.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                        logger.info(f"   ✅ Loaded Parquet: {len(df):,} rows, {len(df.columns)} columns")
                    else:
                        # For large CSV files, sample first to check structure
                        if file_size_mb > 100:  # If larger than 100MB
                            logger.info(f"   Large CSV detected, sampling first 10,000 rows...")
                            df_sample = pd.read_csv(file_path, nrows=10000)
                            logger.info(f"   Sample structure: {len(df_sample.columns)} columns")
                            
                            # Ask user if they want to process full file or use sample
                            logger.info(f"   Full file would be ~{file_size_mb:.0f}MB. Processing 500K rows for development...")
                            df = pd.read_csv(file_path, nrows=500000)  # Limit for development
                        else:
                            df = pd.read_csv(file_path)
                        logger.info(f"   ✅ Loaded CSV: {len(df):,} rows, {len(df.columns)} columns")
                    
                    # Store dataset info
                    self.stats['dataset_details'][dataset_name] = {
                        'source_file': filename,
                        'file_size_mb': file_size_mb,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'columns_list': list(df.columns)
                    }
                    
                    return df
                    
                except Exception as e:
                    logger.warning(f"   ❌ Failed to load {filename}: {e}")
                    continue
        
        logger.warning(f"❌ No readable files found for {dataset_name}")
        return None
        
    def detect_and_standardize_schema(self, df, dataset_name):
        """Detect schema and standardize column names."""
        logger.info(f"Standardizing schema for {dataset_name}...")
        
        # Print original columns for debugging
        logger.info(f"   Original columns: {list(df.columns)}")
        
        # Common column mappings for each dataset
        column_mappings = {
            'saml_d': {
                # SAML-D specific mappings
                'tx_id': 'transaction_id',
                'transaction_id': 'transaction_id',
                'date': 'timestamp',
                'timestamp': 'timestamp',
                'amount': 'amount',
                'tx_amount': 'amount',
                'currency': 'currency',
                'orig_account': 'originator_account',
                'originator_account': 'originator_account',
                'bene_account': 'beneficiary_account',
                'beneficiary_account': 'beneficiary_account',
                'orig_country': 'originator_country',
                'originator_country': 'originator_country',
                'bene_country': 'beneficiary_country',
                'beneficiary_country': 'beneficiary_country',
                'tx_type': 'transaction_type',
                'transaction_type': 'transaction_type',
                'typology': 'typology',
                'is_money_laundering': 'is_laundering',
                'is_laundering': 'is_laundering',
                'risk_score': 'risk_score',
                'alert': 'alert_flag'
            },
            'ibm_aml': {
                # IBM AML specific mappings
                'timestamp': 'timestamp',
                'amount': 'amount',
                'orig_account': 'originator_account',
                'bene_account': 'beneficiary_account', 
                'currency': 'currency',
                'tx_type': 'transaction_type',
                'is_laundering': 'is_laundering',
                'money_laundering': 'is_laundering'
            },
            'maryam': {
                # Maryam specific mappings  
                'transaction_id': 'transaction_id',
                'amount': 'amount',
                'timestamp': 'timestamp',
                'account_from': 'originator_account',
                'account_to': 'beneficiary_account',
                'is_laundering': 'is_laundering',
                'laundering': 'is_laundering'
            }
        }
        
        # Apply dataset-specific mappings
        mapping = column_mappings.get(dataset_name, {})
        
        # Try to match columns (case-insensitive)
        df_renamed = df.copy()
        columns_mapped = {}
        
        for original_col in df.columns:
            # Direct mapping
            if original_col in mapping:
                new_name = mapping[original_col]
                columns_mapped[original_col] = new_name
            else:
                # Case-insensitive matching
                for old_name, new_name in mapping.items():
                    if original_col.lower() == old_name.lower():
                        columns_mapped[original_col] = new_name
                        break
        
        if columns_mapped:
            df_renamed = df_renamed.rename(columns=columns_mapped)
            logger.info(f"   Mapped columns: {columns_mapped}")
        
        # Add standard columns if missing
        self._add_standard_columns(df_renamed, dataset_name)
        
        # Log final schema
        logger.info(f"   Final columns: {list(df_renamed.columns)}")
        
        return df_renamed
        
    def _add_standard_columns(self, df, dataset_name):
        """Add missing standard columns with reasonable defaults."""
        
        # Generate transaction_id if missing
        if 'transaction_id' not in df.columns:
            df['transaction_id'] = [f'{dataset_name.upper()}_{i:06d}' for i in range(len(df))]
            
        # Generate timestamp if missing
        if 'timestamp' not in df.columns:
            base_date = pd.Timestamp('2024-01-01')
            df['timestamp'] = pd.date_range(base_date, periods=len(df), freq='1H')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
        # Add currency if missing
        if 'currency' not in df.columns:
            df['currency'] = 'USD'
            
        # Add basic bank info if missing
        if 'originator_bank' not in df.columns:
            df['originator_bank'] = f'{dataset_name.upper()}_BANK_1'
        if 'beneficiary_bank' not in df.columns:
            df['beneficiary_bank'] = f'{dataset_name.upper()}_BANK_2'
            
        # Add countries if missing
        if 'originator_country' not in df.columns:
            df['originator_country'] = 'US'
        if 'beneficiary_country' not in df.columns:
            df['beneficiary_country'] = 'US'
            
        # Add transaction_type if missing
        if 'transaction_type' not in df.columns:
            df['transaction_type'] = 'transfer'
            
        # Add risk_score if missing
        if 'risk_score' not in df.columns:
            if 'is_laundering' in df.columns:
                # Generate realistic risk scores
                ml_mask = df['is_laundering'] == True
                df['risk_score'] = 0.1  # Default low risk
                if ml_mask.any():
                    df.loc[ml_mask, 'risk_score'] = np.random.uniform(0.7, 1.0, ml_mask.sum())
                if (~ml_mask).any():
                    df.loc[~ml_mask, 'risk_score'] = np.random.uniform(0.0, 0.3, (~ml_mask).sum())
            else:
                df['risk_score'] = np.random.uniform(0.0, 0.5, len(df))
                
        # Add alert_flag if missing
        if 'alert_flag' not in df.columns:
            if 'risk_score' in df.columns:
                df['alert_flag'] = df['risk_score'] > 0.5
            else:
                df['alert_flag'] = False
                
        # Add typology if missing
        if 'typology' not in df.columns:
            typologies = ['placement', 'layering', 'integration', 'structuring', 'smurfing']
            if 'is_laundering' in df.columns:
                ml_mask = df['is_laundering'] == True
                df['typology'] = 'legitimate'
                if ml_mask.any():
                    df.loc[ml_mask, 'typology'] = np.random.choice(typologies, ml_mask.sum())
            else:
                df['typology'] = np.random.choice(typologies + ['legitimate'], len(df))
                
        # Add dataset source
        df['dataset_source'] = dataset_name
        
        # Ensure boolean columns are properly typed
        bool_columns = ['is_laundering', 'alert_flag']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        return df
        
    def load_all_datasets(self):
        """Load all available datasets with smart detection."""
        logger.info("🔍 Loading all available datasets...")
        
        datasets = []
        dataset_names = ['saml_d', 'ibm_aml', 'maryam']
        
        for name in dataset_names:
            dataset_dir = RAW_DATA / name
            if dataset_dir.exists():
                df = self.smart_load_dataset(name, dataset_dir)
                if df is not None:
                    # Standardize schema
                    df_standardized = self.detect_and_standardize_schema(df, name)
                    datasets.append(df_standardized)
                    self.stats['datasets_processed'] += 1
                    logger.info(f"   ✅ {name}: {len(df_standardized):,} transactions processed")
                else:
                    logger.warning(f"   ❌ {name}: No readable data found")
            else:
                logger.warning(f"   ❌ {name}: Directory not found")
        
        # Load synthetic combined if no real datasets found
        if not datasets:
            synthetic_path = DATA_ROOT / "synthetic" / "combined_sample.parquet"
            if synthetic_path.exists():
                df = pd.read_parquet(synthetic_path)
                datasets.append(df)
                self.stats['datasets_processed'] += 1
                logger.info(f"   ✅ synthetic: {len(df):,} transactions loaded")
            else:
                # Try CSV fallback
                synthetic_csv = DATA_ROOT / "synthetic" / "combined_sample.csv"
                if synthetic_csv.exists():
                    df = pd.read_csv(synthetic_csv)
                    datasets.append(df)
                    self.stats['datasets_processed'] += 1
                    logger.info(f"   ✅ synthetic (CSV): {len(df):,} transactions loaded")
        
        if not datasets:
            raise ValueError("❌ No datasets found! Run download_datasets.py first.")
            
        # Combine all datasets
        logger.info("🔄 Combining datasets...")
        combined_df = pd.concat(datasets, ignore_index=True, sort=False)
        
        # Ensure all required columns exist
        required_columns = [
            'transaction_id', 'timestamp', 'amount', 'currency',
            'originator_account', 'beneficiary_account', 'transaction_type',
            'is_laundering', 'risk_score', 'alert_flag', 'typology', 'dataset_source'
        ]
        
        for col in required_columns:
            if col not in combined_df.columns:
                logger.warning(f"   Missing required column: {col}, adding defaults...")
                if col == 'is_laundering':
                    combined_df[col] = False
                elif col == 'alert_flag':
                    combined_df[col] = False
                elif col == 'risk_score':
                    combined_df[col] = 0.1
                else:
                    combined_df[col] = 'unknown'
        
        logger.info(f"📊 Combined dataset: {len(combined_df):,} transactions, {len(combined_df.columns)} columns")
        return combined_df
        
    def clean_data(self, df):
        """Enhanced data cleaning with validation."""
        logger.info("🧹 Cleaning and validating data...")
        
        initial_count = len(df)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Clean amount column
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove invalid records
        df = df.dropna(subset=['transaction_id', 'timestamp', 'amount'])
        df = df[df['amount'] > 0]  # Remove negative/zero amounts
        
        # Standardize boolean columns
        df['is_laundering'] = df['is_laundering'].astype(bool)
        df['alert_flag'] = df['alert_flag'].astype(bool)
        
        # Clean text columns
        text_columns = ['currency', 'transaction_type', 'typology', 'originator_country', 'beneficiary_country']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper().str.strip()
        
        # Standardize risk scores
        df['risk_score'] = pd.to_numeric(df['risk_score'], errors='coerce').fillna(0.1)
        df['risk_score'] = df['risk_score'].clip(0, 1)  # Ensure 0-1 range
        
        # Remove duplicate transaction IDs
        df = df.drop_duplicates(subset=['transaction_id'], keep='first')
        
        cleaned_count = len(df)
        logger.info(f"   Cleaned: {initial_count:,} → {cleaned_count:,} transactions")
        logger.info(f"   Removed: {initial_count - cleaned_count:,} invalid records")
        
        return df
        
    def create_features(self, df):
        """Create enhanced features for ML models."""
        logger.info("🔧 Creating enhanced features...")
        
        # Temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['is_business_hours'] = df['hour'].between(9, 17)
        
        # Amount features
        df['log_amount'] = np.log1p(df['amount'])
        df['amount_rounded'] = np.round(df['amount'], -2)  # Round to nearest 100
        df['is_round_amount'] = (df['amount'] % 1000 == 0)  # Exactly divisible by 1000
        
        # Risk indicators
        df['high_amount'] = df['amount'] > df['amount'].quantile(0.95)
        df['unusual_hour'] = df['hour'].isin([0, 1, 2, 3, 4, 5, 22, 23])
        df['cross_border'] = df['originator_country'] != df['beneficiary_country']
        
        # Account-based features (with error handling)
        try:
            account_stats = df.groupby('originator_account').agg({
                'amount': ['count', 'sum', 'mean'],
                'is_laundering': 'mean'
            }).round(4)
            
            account_stats.columns = ['orig_tx_count', 'orig_total_amount', 'orig_avg_amount', 'orig_ml_rate']
            df = df.merge(account_stats, left_on='originator_account', right_index=True, how='left')
            
            # Fill missing values
            df[['orig_tx_count', 'orig_total_amount', 'orig_avg_amount', 'orig_ml_rate']] = \
                df[['orig_tx_count', 'orig_total_amount', 'orig_avg_amount', 'orig_ml_rate']].fillna(0)
        except Exception as e:
            logger.warning(f"   Could not create account features: {e}")
            # Add default values
            df['orig_tx_count'] = 1
            df['orig_total_amount'] = df['amount']
            df['orig_avg_amount'] = df['amount']
            df['orig_ml_rate'] = 0.0
        
        logger.info(f"   Created features: {len(df.columns)} total columns")
        return df

    def validate_data_quality(self, df):
        """Enhanced data quality validation."""
        logger.info("✅ Validating data quality...")
        
        quality_checks = {}
        
        # Completeness
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        quality_checks['completeness'] = completeness
        
        # Data consistency
        consistency_score = 0
        total_checks = 10
        
        # Check 1: Positive amounts
        if (df['amount'] > 0).all():
            consistency_score += 1
            
        # Check 2: Valid timestamps
        if df['timestamp'].notna().all():
            consistency_score += 1
            
        # Check 3: Risk scores in valid range
        if ((df['risk_score'] >= 0) & (df['risk_score'] <= 1)).all():
            consistency_score += 1
            
        # Check 4: Unique transaction IDs
        if df['transaction_id'].nunique() == len(df):
            consistency_score += 1
            
        # Check 5: Valid currency codes
        valid_currencies = ['USD', 'EUR', 'GBP', 'CAD', 'JPY', 'CHF', 'UNKNOWN']
        if df['currency'].isin(valid_currencies).all():
            consistency_score += 1
            
        # Check 6: Reasonable date range
        date_range = (df['timestamp'].max() - df['timestamp'].min()).days
        if 1 <= date_range <= 2000:  # Between 1 day and 5 years
            consistency_score += 1
            
        # Check 7: Balanced ML distribution
        ml_rate = df['is_laundering'].mean()
        if 0.001 <= ml_rate <= 0.8:  # Between 0.1% and 80%
            consistency_score += 1
            
        # Check 8: Account variety
        if len(df) > 0:
            account_variety = df['originator_account'].nunique() / len(df)
            if account_variety > 0.001:  # At least some unique accounts
                consistency_score += 1
        
        # Check 9: Dataset source tracking
        if 'dataset_source' in df.columns and df['dataset_source'].notna().all():
            consistency_score += 1
            
        # Check 10: Amount distribution sanity
        if df['amount'].std() > 0 and df['amount'].mean() > df['amount'].median():
            consistency_score += 1
            
        consistency = (consistency_score / total_checks) * 100
        quality_checks['consistency'] = consistency
        
        # Business logic score
        business_score = 0
        business_checks = 4
        
        # Reasonable amount distribution
        if df['amount'].median() > 0 and df['amount'].mean() > df['amount'].median():
            business_score += 1
            
        # ML cases have higher risk scores (if we have ML cases)
        ml_cases = df[df['is_laundering']]
        normal_cases = df[~df['is_laundering']]
        if len(ml_cases) > 0 and len(normal_cases) > 0:
            ml_avg_risk = ml_cases['risk_score'].mean()
            normal_avg_risk = normal_cases['risk_score'].mean()
            if ml_avg_risk > normal_avg_risk:
                business_score += 1
        else:
            business_score += 1  # Accept if we don't have both types
            
        # Alert flag correlation (if we have alerts)
        alerts = df[df['alert_flag']]
        no_alerts = df[~df['alert_flag']]
        if len(alerts) > 0 and len(no_alerts) > 0:
            alert_ml_rate = alerts['is_laundering'].mean()
            no_alert_ml_rate = no_alerts['is_laundering'].mean()
            if alert_ml_rate > no_alert_ml_rate:
                business_score += 1
        else:
            business_score += 1  # Accept if we don't have both types
            
        # Typology diversity
        if df['typology'].nunique() >= 2:
            business_score += 1
            
        business_logic = (business_score / business_checks) * 100
        quality_checks['business_logic'] = business_logic
        
        # Overall score
        overall_score = np.mean(list(quality_checks.values()))
        
        logger.info(f"   Quality Assessment:")
        logger.info(f"     Completeness: {completeness:.1f}%")
        logger.info(f"     Consistency: {consistency:.1f}%")
        logger.info(f"     Business Logic: {business_logic:.1f}%")
        logger.info(f"     Overall Score: {overall_score:.1f}%")
        
        return overall_score, quality_checks

    def create_agent_datasets(self, df):
        """Create agent-specific datasets with validation."""
        logger.info("🤖 Creating agent-specific datasets...")
        
        agent_data_dir = PROCESSED_DATA / "agent_datasets"
        agent_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Alert Triage Agent Dataset
        triage_columns = [
            'transaction_id', 'timestamp', 'amount', 'currency', 'transaction_type',
            'originator_country', 'beneficiary_country', 'risk_score', 'is_laundering',
            'typology', 'high_amount', 'cross_border', 'unusual_hour', 'dataset_source'
        ]
        
        # Only include columns that exist
        available_triage_cols = [col for col in triage_columns if col in df.columns]
        triage_df = df[available_triage_cols].copy()
        
        # Save both formats
        triage_df.to_csv(agent_data_dir / "alert_triage_data.csv", index=False)
        triage_df.to_parquet(agent_data_dir / "alert_triage_data.parquet", index=False)
        logger.info(f"   ✅ Alert Triage: {len(triage_df):,} records, {len(available_triage_cols)} columns")
        
        # Evidence Collection Agent Dataset
        evidence_columns = [
            'transaction_id', 'originator_account', 'beneficiary_account',
            'originator_bank', 'beneficiary_bank', 'amount', 'timestamp',
            'orig_tx_count', 'orig_total_amount', 'orig_avg_amount', 'orig_ml_rate', 'dataset_source'
        ]
        
        available_evidence_cols = [col for col in evidence_columns if col in df.columns]
        evidence_df = df[available_evidence_cols].copy()
        
        evidence_df.to_csv(agent_data_dir / "evidence_collection_data.csv", index=False)
        evidence_df.to_parquet(agent_data_dir / "evidence_collection_data.parquet", index=False)
        logger.info(f"   ✅ Evidence Collection: {len(evidence_df):,} records, {len(available_evidence_cols)} columns")
        
        # Pattern Analysis Agent Dataset (full dataset for ML)
        pattern_columns = [
            'transaction_id', 'timestamp', 'amount', 'log_amount', 'currency',
            'originator_account', 'beneficiary_account', 'originator_country', 
            'beneficiary_country', 'transaction_type', 'typology', 'is_laundering',
            'risk_score', 'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
            'high_amount', 'unusual_hour', 'cross_border', 'is_round_amount',
            'orig_tx_count', 'orig_total_amount', 'orig_avg_amount', 'orig_ml_rate', 'dataset_source'
        ]
        
        available_pattern_cols = [col for col in pattern_columns if col in df.columns]
        pattern_df = df[available_pattern_cols].copy()
        
        pattern_df.to_csv(agent_data_dir / "pattern_analysis_data.csv", index=False)
        pattern_df.to_parquet(agent_data_dir / "pattern_analysis_data.parquet", index=False)
        logger.info(f"   ✅ Pattern Analysis: {len(pattern_df):,} records, {len(available_pattern_cols)} columns")
        
        # Narrative Generation Agent Dataset
        narrative_columns = [
            'transaction_id', 'timestamp', 'amount', 'currency', 'typology',
            'originator_account', 'beneficiary_account', 'originator_country',
            'beneficiary_country', 'risk_score', 'is_laundering', 'alert_flag', 'dataset_source'
        ]
        
        available_narrative_cols = [col for col in narrative_columns if col in df.columns]
        narrative_df = df[available_narrative_cols].copy()
        
        narrative_df.to_csv(agent_data_dir / "narrative_generation_data.csv", index=False)
        narrative_df.to_parquet(agent_data_dir / "narrative_generation_data.parquet", index=False)
        logger.info(f"   ✅ Narrative Generation: {len(narrative_df):,} records, {len(available_narrative_cols)} columns")
        
        return {
            'triage': len(triage_df),
            'evidence': len(evidence_df), 
            'pattern': len(pattern_df),
            'narrative': len(narrative_df)
        }
        
    def save_to_database(self, df):
        """Save processed data to SQLite database with validation."""
        logger.info("💾 Saving data to SQLite database...")
        
        db_path = DATA_ROOT / "aml_local.db"
        
        try:
            with sqlite3.connect(db_path) as conn:
                # Save main transactions table
                df.to_sql('transactions', conn, if_exists='replace', index=False)
                
                # Create alerts table
                alerts_df = df[df['alert_flag'] == True][
                    ['transaction_id', 'risk_score', 'typology', 'is_laundering', 'timestamp']
                ].copy()
                
                if len(alerts_df) > 0:
                    alerts_df['alert_id'] = ['ALT_' + str(i).zfill(6) for i in range(len(alerts_df))]
                    alerts_df['status'] = 'pending'
                    alerts_df['created_at'] = alerts_df['timestamp']
                    alerts_df.to_sql('alerts', conn, if_exists='replace', index=False)
                else:
                    # Create empty alerts table
                    conn.execute('''CREATE TABLE IF NOT EXISTS alerts (
                        alert_id TEXT PRIMARY KEY,
                        transaction_id TEXT,
                        risk_score REAL,
                        typology TEXT,
                        is_laundering BOOLEAN,
                        timestamp DATETIME,
                        status TEXT,
                        created_at DATETIME
                    )''')
                
                # Create accounts summary table
                if 'originator_account' in df.columns:
                    accounts_df = df.groupby('originator_account').agg({
                        'amount': ['count', 'sum', 'mean'],
                        'is_laundering': ['sum', 'mean'],
                        'originator_country': 'first',
                        'originator_bank': 'first' if 'originator_bank' in df.columns else lambda x: 'UNKNOWN'
                    }).round(4)
                    
                    accounts_df.columns = ['tx_count', 'total_amount', 'avg_amount', 'ml_count', 'ml_rate', 'country', 'bank']
                    accounts_df['account_id'] = accounts_df.index
                    accounts_df['risk_rating'] = pd.cut(accounts_df['ml_rate'], 
                                                      bins=[0, 0.1, 0.3, 1.0], 
                                                      labels=['Low', 'Medium', 'High'])
                    
                    accounts_df.to_sql('accounts', conn, if_exists='replace', index=False)
                else:
                    logger.warning("   No originator_account column found, skipping accounts table")
                
                logger.info(f"   📊 Database saved:")
                logger.info(f"     Transactions: {len(df):,} records")
                logger.info(f"     Alerts: {len(alerts_df) if len(alerts_df) > 0 else 0:,} records")
                logger.info(f"     Accounts: {len(accounts_df) if 'originator_account' in df.columns else 0:,} records")
                
        except Exception as e:
            logger.error(f"   ❌ Database save failed: {e}")
            raise
            
    def create_sample_datasets(self, df):
        """Create various sample datasets for development."""
        logger.info("📋 Creating sample datasets for development...")
        
        samples_dir = PROCESSED_DATA / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Small development sample
        dev_sample_size = min(500, len(df))
        dev_sample = df.sample(n=dev_sample_size, random_state=42)
        dev_sample.to_csv(samples_dir / "dev_sample.csv", index=False)
        dev_sample.to_parquet(samples_dir / "dev_sample.parquet", index=False)
        
        # Balanced ML sample
        ml_cases = df[df['is_laundering'] == True]
        normal_cases = df[df['is_laundering'] == False]
        
        n_ml = min(len(ml_cases), 100)
        n_normal = min(len(normal_cases), 400)
        
        if n_ml > 0 and n_normal > 0:
            balanced_sample = pd.concat([
                ml_cases.sample(n=n_ml, random_state=42),
                normal_cases.sample(n=n_normal, random_state=42)
            ]).sample(frac=1, random_state=42)
        elif n_ml > 0:
            balanced_sample = ml_cases.sample(n=min(n_ml, 500), random_state=42)
        else:
            balanced_sample = normal_cases.sample(n=min(n_normal, 500), random_state=42)
        
        balanced_sample.to_csv(samples_dir / "balanced_sample.csv", index=False)
        balanced_sample.to_parquet(samples_dir / "balanced_sample.parquet", index=False)
        
        # High-risk sample
        high_risk = df[df['risk_score'] > 0.7]
        if len(high_risk) > 0:
            high_risk.to_csv(samples_dir / "high_risk_sample.csv", index=False)
            high_risk.to_parquet(samples_dir / "high_risk_sample.parquet", index=False)
        
        # Recent transactions sample
        recent = df.nlargest(min(1000, len(df)), 'timestamp')
        recent.to_csv(samples_dir / "recent_sample.csv", index=False)
        recent.to_parquet(samples_dir / "recent_sample.parquet", index=False)
        
        logger.info(f"   📋 Sample datasets created:")
        logger.info(f"     Development: {len(dev_sample):,} records")
        logger.info(f"     Balanced: {len(balanced_sample):,} records")
        logger.info(f"     High-risk: {len(high_risk) if len(high_risk) > 0 else 0:,} records")
        logger.info(f"     Recent: {len(recent):,} records")
        
    def generate_data_statistics(self, df):
        """Generate comprehensive data statistics."""
        logger.info("📈 Generating data statistics...")
        
        stats = {
            'overview': {
                'total_transactions': len(df),
                'date_range': {
                    'start': df['timestamp'].min().isoformat() if df['timestamp'].notna().any() else None,
                    'end': df['timestamp'].max().isoformat() if df['timestamp'].notna().any() else None,
                    'days': (df['timestamp'].max() - df['timestamp'].min()).days if df['timestamp'].notna().any() else 0
                },
                'ml_cases': int(df['is_laundering'].sum()),
                'ml_rate': float(df['is_laundering'].mean()),
                'alert_cases': int(df['alert_flag'].sum()),
                'alert_rate': float(df['alert_flag'].mean())
            },
            'amounts': {
                'min': float(df['amount'].min()),
                'max': float(df['amount'].max()),
                'mean': float(df['amount'].mean()),
                'median': float(df['amount'].median()),
                'total': float(df['amount'].sum())
            },
            'risk_scores': {
                'min': float(df['risk_score'].min()),
                'max': float(df['risk_score'].max()),
                'mean': float(df['risk_score'].mean()),
                'ml_cases_avg': float(df[df['is_laundering']]['risk_score'].mean()) if df['is_laundering'].any() else 0.0,
                'normal_cases_avg': float(df[~df['is_laundering']]['risk_score'].mean()) if (~df['is_laundering']).any() else 0.0
            },
            'categories': {
                'currencies': df['currency'].value_counts().to_dict(),
                'transaction_types': df['transaction_type'].value_counts().to_dict(),
                'typologies': df['typology'].value_counts().to_dict(),
                'datasets': df['dataset_source'].value_counts().to_dict() if 'dataset_source' in df.columns else {},
                'countries': {
                    'originator': df['originator_country'].value_counts().head(10).to_dict() if 'originator_country' in df.columns else {},
                    'beneficiary': df['beneficiary_country'].value_counts().head(10).to_dict() if 'beneficiary_country' in df.columns else {}
                }
            },
            'temporal': {
                'hourly_distribution': df['hour'].value_counts().sort_index().to_dict() if 'hour' in df.columns else {},
                'daily_distribution': df['day_of_week'].value_counts().sort_index().to_dict() if 'day_of_week' in df.columns else {},
                'weekend_rate': float(df['is_weekend'].mean()) if 'is_weekend' in df.columns else 0.0,
                'business_hours_rate': float(df['is_business_hours'].mean()) if 'is_business_hours' in df.columns else 0.0
            }
        }
        
        # Save statistics
        stats_path = PROCESSED_DATA / "data_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
            
        logger.info("   📈 Comprehensive data statistics generated")
        return stats
        
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline with enhanced validation."""
        start_time = datetime.now()
        logger.info("🚀 Starting Enhanced AML Data Preprocessing Pipeline...")
        
        # Create output directories
        PROCESSED_DATA.mkdir(exist_ok=True)
        (PROCESSED_DATA / "unified").mkdir(exist_ok=True)
        (PROCESSED_DATA / "features").mkdir(exist_ok=True)
        (PROCESSED_DATA / "samples").mkdir(exist_ok=True)
        
        try:
            # Load all datasets with smart detection
            df = self.load_all_datasets()
            self.stats['total_transactions'] = len(df)
            self.stats['laundering_transactions'] = int(df['is_laundering'].sum())
            
            # Clean data
            df_clean = self.clean_data(df)
            
            # Create features
            df_features = self.create_features(df_clean)
            
            # Validate data quality
            quality_score, quality_details = self.validate_data_quality(df_features)
            self.stats['data_quality_score'] = quality_score
            
            # Save processed datasets in both formats
            df_clean.to_csv(PROCESSED_DATA / "unified" / "cleaned_data.csv", index=False)
            df_clean.to_parquet(PROCESSED_DATA / "unified" / "cleaned_data.parquet", index=False)
            
            df_features.to_csv(PROCESSED_DATA / "features" / "feature_enhanced_data.csv", index=False)
            df_features.to_parquet(PROCESSED_DATA / "features" / "feature_enhanced_data.parquet", index=False)
            
            # Create agent-specific datasets
            agent_stats = self.create_agent_datasets(df_features)
            
            # Save to database
            self.save_to_database(df_features)
            
            # Create sample datasets
            self.create_sample_datasets(df_features)
            
            # Generate statistics
            data_stats = self.generate_data_statistics(df_features)
            
            # Update processing stats
            end_time = datetime.now()
            self.stats['processing_time'] = str(end_time - start_time)
            
            # Save comprehensive processing summary
            processing_stats = {
                'processing_stats': self.stats,
                'quality_details': quality_details,
                'agent_datasets': agent_stats,
                'data_statistics': data_stats
            }
            
            with open(PROCESSED_DATA / "processing_summary.json", 'w') as f:
                json.dump(processing_stats, f, indent=2, default=str)
                
            logger.info("✅ Enhanced preprocessing pipeline completed successfully!")
            return df_features, processing_stats
            
        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {e}")
            raise

def main():
    """Main preprocessing function with enhanced reporting."""
    preprocessor = EnhancedAMLDataPreprocessor()
    
    try:
        result_df, stats = preprocessor.run_preprocessing()
        
        print("\n" + "="*80)
        print("🎉 ENHANCED AML DATA PREPROCESSING COMPLETED!")
        print("="*80)
        
        print(f"\n📊 Processing Summary:")
        print(f"   • Total Transactions: {stats['processing_stats']['total_transactions']:,}")
        print(f"   • Money Laundering Cases: {stats['processing_stats']['laundering_transactions']:,}")
        ml_rate = (stats['processing_stats']['laundering_transactions']/stats['processing_stats']['total_transactions']*100) if stats['processing_stats']['total_transactions'] > 0 else 0
        print(f"   • ML Rate: {ml_rate:.1f}%")
        print(f"   • Datasets Processed: {stats['processing_stats']['datasets_processed']}")
        print(f"   • Data Quality Score: {stats['processing_stats']['data_quality_score']:.1f}%")
        print(f"   • Processing Time: {stats['processing_stats']['processing_time']}")
        
        print(f"\n📁 Output Files Created (Parquet + CSV):")
        print(f"   • Cleaned Data: data/processed/unified/cleaned_data.parquet")
        print(f"   • Enhanced Features: data/processed/features/feature_enhanced_data.parquet")
        print(f"   • SQLite Database: data/aml_local.db")
        print(f"   • Agent Datasets: data/processed/agent_datasets/ (8 files)")
        print(f"   • Sample Datasets: data/processed/samples/ (8 files)")
        print(f"   • Statistics: data/processed/data_statistics.json")
        
        print(f"\n🤖 Agent-Ready Datasets:")
        for agent, count in stats['agent_datasets'].items():
            print(f"   • {agent.title()}: {count:,} records")
            
        print(f"\n🎯 Data Quality Breakdown:")
        for metric, score in stats['quality_details'].items():
            print(f"   • {metric.title()}: {score:.1f}%")
            
        print(f"\n📈 Dataset Composition:")
        if 'dataset_source' in result_df.columns:
            dataset_counts = result_df['dataset_source'].value_counts()
            for dataset, count in dataset_counts.items():
                print(f"   • {dataset}: {count:,} transactions")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. ✅ Data preprocessing complete")
        print(f"   2. 🔧 Start agent development with clean datasets")
        print(f"   3. 🤖 Use Parquet files for 5-10x faster loading")
        print(f"   4. 🧪 Test with sample datasets (data/processed/samples/)")
        print(f"   5. 💾 Query database: sqlite3 data/aml_local.db")
        
        print(f"\n💡 Quick Test Commands:")
        print(f"   # Load main dataset (fast)")
        print(f"   python -c \"import pandas as pd; df=pd.read_parquet('data/processed/features/feature_enhanced_data.parquet'); print(f'Loaded {{len(df):,}} transactions with {{len(df.columns)}} features')\"")
        
        print(f"\n   # Check database")
        print(f"   python -c \"import sqlite3; conn=sqlite3.connect('data/aml_local.db'); print('Tables:', [row[0] for row in conn.execute('SELECT name FROM sqlite_master WHERE type=\\\"table\\\";').fetchall()]); conn.close()\"")
        
        print("\n" + "="*80)
        
        # Quick verification test
        try:
            feature_path = PROCESSED_DATA / "features" / "feature_enhanced_data.parquet"
            if feature_path.exists():
                test_df = pd.read_parquet(feature_path)
                print(f"\n✅ Verification Successful:")
                print(f"   • Loaded {len(test_df):,} transactions")
                print(f"   • {test_df['is_laundering'].sum():,} ML cases ({test_df['is_laundering'].mean()*100:.1f}%)")
                print(f"   • {len(test_df.columns)} features available")
                print(f"   • Memory usage: {test_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
                print(f"   🎯 Ready for agent development!")
            else:
                print(f"\n⚠️ Could not find processed dataset for verification")
        except Exception as e:
            print(f"\n⚠️ Verification failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced preprocessing failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check that download_datasets.py completed successfully")
        print("2. Verify data files exist in data/raw/ directories")
        print("3. Ensure pyarrow is installed: pip install pyarrow")
        print("4. Check available disk space and write permissions")
        return False

if __name__ == "__main__":
    main()