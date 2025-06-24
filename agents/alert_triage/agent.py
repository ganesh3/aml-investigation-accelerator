import time
import json
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from agents.base.base_agent import BaseAMLAgent, AMLAgentBase
from agents.base.message_system import AgentMessage, AgentResponse, MessageType
import pandas as pd

class AlertTriageAgent(AMLAgentBase):
    """
    Alert triage agent - Uses pre-build machine learning models for risk assessment

    Capabilities:
    - Loads pre-trained Random Forest, XGBoost, and Catboost models
    - Performs risk scoring using ensemble approach
    - Priortizes alerts based on risk thresholds
    - Provides reasoning for all decisions
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__('alert_triage', config)
        
        self.thresholds = config.get('thresholds', {
            'high_risk_score': 0.8,
            'medium_risk_score': 0.5,
            'auto_escalate_score': 0.9,
            'auto_dismiss_score': 0.1
        })
        
        # Get model path from config
        raw_model_path = config.get('model_path', 'models')
        path_obj = Path(raw_model_path)
        # If the path is relative, resolve it based on project root
        if not path_obj.is_absolute():
            # Assumes agent.py is 2 levels down (agents/alert_triage/agent.py)
            project_root = Path(__file__).resolve().parents[2]
            path_obj = (project_root / path_obj).resolve()

        self.model_path = path_obj
        self.logger.info(f"Resolved model path: {self.model_path}")
        
        self.models = {}
        self.scalar = None
        self.feature_columns = []
        self.ensemble_weights = None
        
        self._load_pretrained_models()
        
        self.alerts_processed = 0
        self.high_risk_alerts = 0
        self.auto_escalated = 0
        self.auto_dismissed = 0
        
        self.logger.info(f"Alert triage agent initialized with {len(self.models)} models")
        
    def _load_pretrained_models(self):
        """Load pre-trained models and metadata"""
        try:
            metadata_path = self.model_path / 'ensemble_metadata.json'
            self.logger.info(f"Loading ensemble model metadata from {metadata_path}")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                    self.feature_columns = metadata['feature_columns']
                    self.model_performance = metadata['model_performance']
                    self.ensemble_weights = metadata.get('ensemble_weights')
                    
                    self.logger.info(f"Loaded metadata: {len(self.feature_columns)} features")
            
            scalar_path = self.model_path / 'ensemble_scalar.pkl'
            
            if scalar_path.exists():
                self.scalar = joblib.load(scalar_path)
                self.logger.info("Loaded scalar successfully")
                
            model_files = {
                'random_forest': 'random_forest_model.pkl',
                'xgboost': 'xgboost_model.pkl',
                'catboost': 'catboost_model.pkl'
            }
            
            for model_name,filename in model_files.items():
                model_path = self.model_path / filename
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    self.logger.info(f"Loaded {model_name} model successfully")
           

            self.logger.info(f"Successfully loaded {len(self.models)} models")
        except Exception as e:
            self.logger.error(f"Failed to load pre-trained model: {e}")
            raise
            
    def prepare_features(self, transaction_data: Dict[str, Any]) -> np.ndarray:
        """Prepare features for model prediction"""
        
        try:
            features = []
            
            for feature_name in self.feature_columns:
                value = transaction_data.get(feature_name, 0)
                
                if isinstance(value, bool):
                    value = int(value)
                elif isinstance(value, str):
                    value = 0
                elif pd.isna(value):
                    value = 0
                
                features.append(float(value))
                
            self.logger.info("Prepared features {features}")
            features_array = np.array(features).reshape(1, -1)
            self.logger.info(f"Features array shape: {features_array.shape}")
            
            if self.scalar is not None:
                features_array = self.scalar.transform(features_array)
                
            return features_array
        
        except Exception as e:
            self.logger.error(f"Error in preparing features: {e}")
            raise
        
    def calculate_ensemble_risk_score(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk score using ensemble approach"""
        try:
            features = self.prepare_features(transaction_data)
            
            predictions={}
            for model_name, model in self.models.items():
                pred_proba = model.predict_proba(features)[0]
                predictions[model_name] = pred_proba[1]
                
            if self.ensemble_weights is not None and len(self.ensemble_weights) == len(predictions):
                risk_score = sum(weight * pred for weight, pred in zip(self.ensemble_weights, predictions.values()))
                ensemble_method = 'weighted'
            else:
                risk_score = np.mean(list(predictions.values()))
                ensemble_method = 'average'
            
            pred_values = list(predictions.values())
            confidence = 1.0 - np.std(pred_values)
                
            return {
                'risk_score': risk_score,
                'confidence': float(max(0.0, min(1.0, confidence))),
                'individual_predictions': predictions, 
                'ensemble_method': ensemble_method,
                'reasoning': self._generate_reasoning(risk_score, transaction_data, predictions)
            }
        
        except Exception as e:
            self.logger.error(f"Risk calculation failed: {e}")
            return {
                'risk_score': 0.5,
                'confidence': 0.5,
                'individual_predictions': {},
                'ensemble_method': 'fallback',
                'reasoning': ['Risk calculation failed, using default score']
            }
            
    def _generate_reasoning(self, risk_score: float, transaction_data: Dict[str, Any], 
                          predictions: Dict[str, float]) -> List[str]:
        """Generate human-readable reasoning for the risk score"""
        reasoning = []
        
        # Overall risk level
        if risk_score >= 0.9:
            reasoning.append(f"Very high ML probability: {risk_score:.1%}")
        elif risk_score >= 0.7:
            reasoning.append(f"High ML probability: {risk_score:.1%}")
        elif risk_score >= 0.5:
            reasoning.append(f"Moderate ML probability: {risk_score:.1%}")
        else:
            reasoning.append(f"Low ML probability: {risk_score:.1%}")
        
        # Model consensus
        if max(predictions.values()) - min(predictions.values()) > 0.3:
            reasoning.append("Models show mixed predictions")
        else:
            reasoning.append("Models show good consensus")
        
        # Feature-based reasoning
        amount = float(transaction_data.get('amount', 0))
        if amount > 50000:
            reasoning.append(f"High transaction amount: ${amount:,.2f}")
        
        if transaction_data.get('cross_border', False):
            reasoning.append("Cross-border transaction")
        
        if transaction_data.get('unusual_hour', False):
            reasoning.append("Transaction at unusual hour")
        
        orig_ml_rate = float(transaction_data.get('orig_ml_rate', 0))
        if orig_ml_rate > 0.1:
            reasoning.append(f"Account has {orig_ml_rate:.1%} ML history")
        
        base_risk = float(transaction_data.get('risk_score', 0))
        if base_risk > 0.7:
            reasoning.append(f"High base risk score: {base_risk:.1%}")
        
        return reasoning
    
    def prioritize_alert(self, risk_assessment: Dict[str, Any], 
                        transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine alert priority and recommended actions"""
        
        risk_score = risk_assessment['risk_score']
        
        priority_info = {
            'risk_score': risk_score,
            'confidence': risk_assessment['confidence'],
            'priority_level': 'MEDIUM',
            'recommended_action': 'INVESTIGATE',
            'auto_action': None,
            'reasoning': risk_assessment['reasoning'],
            'individual_predictions': risk_assessment['individual_predictions'],
            'ensemble_method': risk_assessment['ensemble_method']
        }
        
        # Determine priority and actions based on thresholds
        if risk_score >= self.thresholds['auto_escalate_score']:
            priority_info.update({
                'priority_level': 'CRITICAL',
                'auto_action': 'ESCALATE',
                'recommended_action': 'IMMEDIATE_INVESTIGATION'
            })
            
        elif risk_score >= self.thresholds['high_risk_score']:
            priority_info.update({
                'priority_level': 'HIGH',
                'recommended_action': 'INVESTIGATE'
            })
            
        elif risk_score <= self.thresholds['auto_dismiss_score']:
            priority_info.update({
                'priority_level': 'LOW',
                'auto_action': 'DISMISS',
                'recommended_action': 'DISMISS'
            })
            
        elif risk_score >= self.thresholds['medium_risk_score']:
            priority_info.update({
                'priority_level': 'MEDIUM',
                'recommended_action': 'INVESTIGATE'
            })
            
        else:
            priority_info.update({
                'priority_level': 'LOW',
                'recommended_action': 'MONITOR'
            })
        
        return priority_info
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities"""
        return [
            "ensemble_risk_scoring",
            "alert_prioritization", 
            "ml_based_classification",
            "automated_triage",
            "reasoning_generation",
            f"model_ensemble_{len(self.models)}_models"
        ]
        
    async def process_request(self, message: AgentMessage):
        """Process request and return response"""
        start_time = time.time()
        
        try:
            if message.message_type == MessageType.RISK_ASSESSMENT_REQUEST.value:
                return await self._handle_risk_assessment(message)
            
            elif message.message_type == MessageType.ALERT_RECEIVED.value:
                return await self._handle_new_alert(message)
            
            elif message.message_type == MessageType.AGENT_STATUS_REQUEST.value:
                return await self._handle_status_request(message)
            
            else:
                return AgentResponse(
                    agent_id = self.agent_id,
                    success = False,
                    error_message = str(e),
                    processing_time=time.time() - start_time
                )
        except Exception as e:
            self._update_performance_metrics(time.time() - start_time, False)
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    async def _handle_risk_assessment(self, message: AgentMessage):
        """Handle risk assessment requests"""
        
        start_time = time.time()
        
        transaction_data = message.payload.get('transaction_data', {})
        transaction_id = transaction_data.get('transaction_id')
        
        if not transaction_id:
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                error_message="Missing transaction_id in request",
                processing_time=time.time() - start_time
            )
        
        # Get additional data from database if available
        db_transaction = self.get_transaction(transaction_id)
        if db_transaction:
            transaction_data.update(db_transaction)
            
        # Perform risk assessment
        risk_assessment = self.calculate_ensemble_risk_score(transaction_data)
        priority_info = self.prioritize_alert(risk_assessment, transaction_data)
        
        self.alerts_processed += 1
        
        if priority_info['priority_level'] in ['HIGH', 'CRITICAL']:
            self.high_risk_alerts += 1
        if priority_info.get('auto_action') == 'ESCALATE': 
            self.auto_escalated += 1
        elif priority_info.get('auto_action') == 'DISMISS':
            self.auto_dismissed += 1
            
        self._update_performance_metrics(time.time() - start_time, True)
        
        return AgentResponse(
            agent_id=self.agent_id,
            success=True,
            data={
                'transaction_id': transaction_id, 
                'assessment': priority_info
                },
            processing_time=time.time() - start_time,
            confidence_score=risk_assessment['confidence']
        )
    
    async def _handle_new_alert(self, message: AgentMessage):
        """Handle new alert notifications"""
        
        alert_data = message.payload.get('alert_data', {})
        
        risk_message = AgentMessage(
            sender=message.sender,
            receiver=self.agent_id, 
            message_type = MessageType.RISK_ASSESSMENT_REQUEST.value,
            payload={'transaction_data': alert_data},
            timestamp = datetime.now(),
            correlation_id=message.correlation_id
        )

        return await self._handle_risk_assessment(risk_message)
    
    async def _handle_status_request(self, message: AgentMessage):
        """Handle status requests"""

        start_time = time.time()
        
        status_data = self.get_status()
        status_data.update({
        'models_loaded': list(self.models.keys()),
        'ensemble_performance': self.model_performance,
        'feature_count': len(self.feature_columns),               
        'statistics': {
            'alerts_processed': self.alerts_processed,
            'high_risk_alerts': self.high_risk_alerts,
            'auto_escalated': self.auto_escalated,
            'auto_dismissed': self.auto_dismissed,
            'high_risk_rate': (self.high_risk_alerts / max(self.alerts_processed, 1)) * 100
                }
            }
        )

        return AgentResponse(
            agent_id=self.agent_id,
            success=True,
            data=status_data,
            processing_time=time.time() - start_time,
            confidence_score=1.0
        )