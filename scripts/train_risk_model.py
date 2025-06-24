#!/usr/bin/env python3
"""
Advanced Ensemble Model Training for AML Risk Scoring
Combines Random Forest, XGBoost, and CatBoost for maximum accuracy
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, List, Tuple

# ML Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Advanced ML Models
import xgboost as xgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not installed. Install with: pip install catboost")
    
# Ensemble
from sklearn.ensemble import VotingClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AMLEnsembleTrainer:
    """
    Advanced ensemble trainer for AML risk scoring models
    
    Features:
    - Random Forest (handles feature interactions well)
    - XGBoost (gradient boosting, excellent performance)
    - CatBoost (handles categorical features natively)
    - Voting ensemble for final predictions
    - Feature importance analysis
    - Model performance comparison
    """
    
    def __init__(self, data_path: str = r'D:\Google-ADK-Project\data\processed\features\feature_enhanced_data.parquet'):
        self.data_path = Path(data_path)
        self.models = {}
        self.scaler = None  # Fixed: was 'scalar' 
        self.ensemble_model = None
        self.feature_columns = []
        self.model_performance = {}  # Fixed: was list, should be dict
        self.categorical_features = []
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare training data"""
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        logger.info(f"Loaded {len(df):,} transactions")
        logger.info(f"ML cases: {df['is_laundering'].sum():,} ({df['is_laundering'].mean():.1%})")
        
        # Feature selection for AML risk model
        feature_columns = [
            # Amount features
            'amount', 'log_amount', 'is_round_amount', 'high_amount',
            
            # Risk Indicators
            'risk_score', 'cross_border', 'unusual_hour',
            
            # Temporal Features
            'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
            
            # Account Features - Fixed column names
            'orig_tx_count', 'orig_total_amount', 'orig_avg_amount', 'orig_ml_rate',
            
            # Transaction Type (if available)
            'transaction_type'
        ]

        available_features = [col for col in feature_columns if col in df.columns]
        
        df_processed = df[available_features + ['is_laundering']].copy()
        
        # Encode categorical columns
        categorical_features = []
        
        for col in available_features:
            if df_processed[col].dtype == 'object':
                df_processed[col] = pd.Categorical(df_processed[col]).codes
                categorical_features.append(col)
            elif df_processed[col].dtype == 'bool':
                df_processed[col] = df_processed[col].astype(int)
                
        df_processed = df_processed.fillna(0)
        
        # Prepare features and target
        X = df_processed[available_features]
        y = df_processed['is_laundering'].astype(int)
        
        self.feature_columns = available_features
        self.categorical_features = categorical_features
        
        logger.info(f"Final features: {len(self.feature_columns)}")
        logger.info(f"Final categorical features: {len(self.categorical_features)}")
        
        return X, y
    
    def train_individual_models(self, X_train_scaled, X_test_scaled, X_train_raw, X_test_raw, y_train, y_test):
        """Train individual models"""

        # Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200, 
            random_state=42,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1)  # Use all cores
        
        rf_model.fit(X_train_scaled, y_train)
        rf_score = roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])
        
        self.models['random_forest'] = rf_model
        self.model_performance['random_forest'] = rf_score
        logger.info(f"Random Forest AUC: {rf_score:.4f}")
        
        # XGBoost
        logger.info("Training XGBoost...") 
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]), # Handle imbalance
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
            n_jobs=-1  # Use all cores
        )
        
        xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
        
        xgb_score = roc_auc_score(y_test, xgb_model.predict_proba(X_test_scaled)[:, 1])
        self.models['xgboost'] = xgb_model
        self.model_performance['xgboost'] = xgb_score
        logger.info(f"XGBoost AUC: {xgb_score:.4f}")
        
        # CatBoost
        if CATBOOST_AVAILABLE:
            logger.info("Training CatBoost...")
            try:
                """# Find categorical features ONLY in the raw (unscaled) data
                cat_features_idx = []
                for i, col in enumerate(self.feature_columns):
                    if col in self.categorical_features:
                        # Check if this column actually has categorical data in raw form
                        unique_vals = len(np.unique(X_train_raw.iloc[:, i]))
                        if unique_vals < len(X_train_raw) * 0.5:  # Less than 50% unique values
                            cat_features_idx.append(i)"""
                
                cb_model = cb.CatBoostClassifier(
                    iterations=200,
                    depth=8,
                    learning_rate=0.1,
                    # Remove cat_features parameter entirely - treat all as numeric on scaled data
                    random_seed=42,
                    verbose=0,
                    class_weights=[1, len(y_train[y_train==0]) / len(y_train[y_train==1])]  # Handle class imbalance
                )
        
                # Use scaled data for training and evaluation (same as other models)
                cb_model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test), verbose=False)
                cb_score = roc_auc_score(y_test, cb_model.predict_proba(X_test_scaled)[:, 1])
                
                self.models['catboost'] = cb_model
                self.model_performance['catboost'] = cb_score
                logger.info(f"CatBoost (scaled data) AUC: {cb_score:.4f}")
                        
            except Exception as e:
                logger.error(f"CatBoost training failed: {e}")
                logger.warning("Continuing with Random Forest and XGBoost only")
        else:
            logger.warning("CatBoost not available. Using only Random Forest & XGBoost")
            
    def create_ensemble(self, X_train_scaled, y_train):
        """Create ensemble model"""

        logger.info("Creating ensemble model...")

        # Ensemble model
        estimators = [
            ('rf', self.models['random_forest']),
            ('xgb', self.models['xgboost'])
        ]
        
        if 'catboost' in self.models:
            estimators.append(('cb', self.models['catboost']))

        logger.info("Training using Stacking Classifier")
         # Meta-learner (learns how to combine base models)
        meta_learner = LogisticRegression(
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Create StackingClassifier
        self.ensemble_model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,  # 5-fold cross-validation for training meta-learner
            stack_method='predict_proba',  # Use probabilities, not hard predictions
            n_jobs=1,  # Windows compatibility
            passthrough=False  # Only use base model predictions, not original features
        )   
    
        self.ensemble_model.fit(X_train_scaled, y_train)
        logger.info("Ensemble model created successfully")
        
    def evaluate_ensemble(self, X_test_scaled, X_test_raw, y_test):
        """Evaluate ensemble model"""

        logger.info("Evaluating ensemble model...")

        predictions = {}
        
        predictions['rf'] = self.models['random_forest'].predict_proba(X_test_scaled)[:, 1]
        predictions['xgb'] = self.models['xgboost'].predict_proba(X_test_scaled)[:, 1]
        
        if 'catboost' in self.models:
            predictions['cb'] = self.models['catboost'].predict_proba(X_test_raw)[:, 1]
            
        # Simple ensemble
        if 'catboost' in self.models:
            ensemble_proba = (predictions['rf'] + predictions['xgb'] + predictions['cb']) / 3
        else:
            ensemble_proba = (predictions['rf'] + predictions['xgb']) / 2
        
        # Weighted Ensemble
        weights = []
        proba_list = []
        
        for model_name in ['random_forest', 'xgboost', 'catboost']:
            if model_name in self.model_performance:
                weights.append(self.model_performance[model_name])
                if model_name == 'random_forest':
                    proba_list.append(predictions['rf'])
                elif model_name == 'xgboost':
                    proba_list.append(predictions['xgb'])
                elif model_name == 'catboost':
                    proba_list.append(predictions['cb'])
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        weighted_ensemble_proba = np.average(proba_list, axis=0, weights=weights)
        
        simple_auc = roc_auc_score(y_test, ensemble_proba)
        weighted_auc = roc_auc_score(y_test, weighted_ensemble_proba)
        
        logger.info(f"Simple Ensemble AUC: {simple_auc:.4f}")
        logger.info(f"Weighted Ensemble AUC: {weighted_auc:.4f}")
        
        if weighted_auc > simple_auc:
            logger.info("Weighted ensemble outperforms simple ensemble")
            self.final_ensemble_proba = weighted_ensemble_proba
            self.ensemble_weights = weights
            self.ensemble_method = 'weighted'
            final_auc = weighted_auc
        else:
            self.final_ensemble_proba = ensemble_proba
            self.ensemble_weights = None
            self.ensemble_method = 'simple'
            final_auc = simple_auc
            
        self.model_performance['ensemble'] = final_auc
        
        self._generate_detailed_evaluation(y_test, self.final_ensemble_proba)
        
        return final_auc
    
    def _generate_detailed_evaluation(self, y_test, ensemble_proba):
        """Generate detailed evaluation metrics and plots"""
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        # Classification report
        print("\n" + "="*60)
        print("ENSEMBLE MODEL EVALUATION")
        print("="*60)
        print("\nClassification Report:")
        print(classification_report(y_test, ensemble_pred))
        
        cm = confusion_matrix(y_test, ensemble_pred)
        print(f"\nConfusion Matrix:")
        print(f"True Negatives: {cm[0,0]:,}")
        print(f"False Positives: {cm[0,1]:,}")
        print(f"False Negatives: {cm[1,0]:,}")
        print(f"True Positives: {cm[1,1]:,}")
        
        # Model comparison
        print(f"\nModel Performance Comparison (AUC):")
        for model_name, auc_score in sorted(self.model_performance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model_name.replace('_', ' ').title()}: {auc_score:.4f}")
        
        # Feature importance (from Random Forest)
        if 'random_forest' in self.models:
            feature_importance = dict(zip(self.feature_columns, 
                                        self.models['random_forest'].feature_importances_))
            
            print(f"\nTop 10 Most Important Features:")
            for feature, importance in sorted(feature_importance.items(), 
                                            key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {feature}: {importance:.4f}")
                
    def save_ensemble_model(self):
        """Save the complete ensemble model"""
        
        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)

        # Save individual models
        for model_name, model in self.models.items():
            joblib.dump(model, model_dir / f'{model_name}_model.pkl')

        # Save scaler and ensemble
        joblib.dump(self.scaler, model_dir / 'ensemble_scaler.pkl')
        joblib.dump(self.ensemble_model, model_dir / 'ensemble_model.pkl')
        
        # Save ensemble metadata
        ensemble_metadata = {
            'feature_columns': self.feature_columns,
            'categorical_features': self.categorical_features,
            'model_performance': self.model_performance,
            'ensemble_method': self.ensemble_method,
            'ensemble_weights': self.ensemble_weights.tolist() if self.ensemble_weights is not None else None,
            'models_included': list(self.models.keys()),
            'training_date': datetime.now().isoformat(),
            'catboost_available': CATBOOST_AVAILABLE
        }
        
        with open(model_dir / 'ensemble_metadata.json', 'w') as f:
            json.dump(ensemble_metadata, f, indent=2)
        
        logger.info(f"Ensemble model saved to {model_dir}")
        
        # Save model summary
        summary = {
            'total_models': len(self.models),
            'best_individual_model': max(self.model_performance.items(), key=lambda x: x[1]),
            'ensemble_performance': self.model_performance['ensemble'],
            'features_used': len(self.feature_columns),
            'ensemble_method': self.ensemble_method
        }
        
        return summary
    
    def train_complete_ensemble(self):
        """Main training pipeline"""
        
        logger.info("Starting ensemble model training...")
        
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for RF and XGBoost
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Keep raw data for CatBoost
        X_train_raw = X_train.copy()
        X_test_raw = X_test.copy()
        
        # Train individual models
        self.train_individual_models(X_train_scaled, X_test_scaled, X_train_raw, X_test_raw, y_train, y_test)
        
        # Create and evaluate ensemble
        self.create_ensemble(X_train_scaled, y_train)
        ensemble_auc = self.evaluate_ensemble(X_test_scaled, X_test_raw, y_test)
        
        logger.info(f"Ensemble AUC: {ensemble_auc:.4f}")
        
        # Save models
        summary = self.save_ensemble_model()
        
        logger.info("Ensemble training completed successfully!")
        
        return summary

def main():
    """Main training function"""
    print("🤖 AML Ensemble Model Training")
    print("="*50)
    
    # Check if required libraries are installed
    required_packages = ['xgboost']
    missing_packages = []
    
    try:
        import xgboost
    except ImportError:
        missing_packages.append('xgboost')
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    
    try:
        # Initialize trainer
        trainer = AMLEnsembleTrainer()
        
        # Train ensemble
        summary = trainer.train_complete_ensemble()
        
        print(f"\n🎉 Training Summary:")
        print(f"   Models trained: {summary['total_models']}")
        print(f"   Best individual: {summary['best_individual_model'][0]} ({summary['best_individual_model'][1]:.4f})")
        print(f"   Ensemble performance: {summary['ensemble_performance']:.4f}")
        print(f"   Features used: {summary['features_used']}")
        print(f"   Ensemble method: {summary['ensemble_method']}")
        
        print(f"\n✅ Ensemble model ready for production!")
        print(f"📁 Models saved in: models/")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Training failed. Check the error messages above.")
    else:
        print("\n🚀 Ready to use the ensemble model in Alert Triage Agent!")