from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import uuid

class MessageType(Enum):
    """Standard message types for agent communication"""
    ALERT_RECEIVED = "alert_received"
    RISK_ASSESSMENT_REQUEST = "alert_received"
    RISK_ASSESSMEST_RESPONSE = "risk_assessment_response"
    EVIDENCE_REQUEST = "evidence_request"
    EVIDENCE_RESPONSE = "evidence_response"
    PATTERN_ANALYSIS_REQUEST = "pattern_analysis_request"
    PATTERN_ANALYSIS_RESPONSE = "pattern_analysis_response"
    NARRATIVE_REQUEST = "narrative_request"
    NARRATIVE_RESPONSE = "narrative_response"
    AGENT_STATUS_REQUEST = "agent_status_request"
    ESCALATION_REQUEST = "escalation_request"
    INVESTIGATION_START = "investigation_start"
    INVESTIGATION_COMPLETE = "investigation_complete"
    ERROR = "error"
    
@dataclass
class AgentMessage:
    """Standard message format between agents"""
    sender: str
    receiver: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: str
    priority: int = 5 # 1-10, higher=more urgent
    
    def __post_init__(self):
        """Generate a unique correlation ID if not provided"""
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())
            
@dataclass
class AgentResponse:
    """Standard response format for agents"""
    agent_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    confidence_score: Optional[float] = None
    recommendations: Optional[List[str]] = None