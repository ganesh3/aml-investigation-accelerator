import asyncio
import time
import pandas as pd
import sqlite3
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from agents.base.message_system import AgentMessage, MessageType, AgentResponse

class BaseAMLAgent(ABC):
    """Abstract base class that defines what every agent must implement"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.logger = logging.getLogger(f"agent.{agent_id}")
        self.is_active = False
        self.start_time = None
        
        #Performance Tracking
        self.processed_count = 0
        self.error_count = 0
        self.total_processing_time = 0
        self.average_processing_time = 0
        
    @abstractmethod
    async def process_request(self, message: AgentMessage) -> AgentResponse:
        """Every agent must implement this - handles incoming requests"""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Every agent must define what it can do"""
        pass
    
    def start(self):
        """Start the agent"""
        self.is_active = True
        self.start_time = datetime.now()
        self.logger.info(f"Agent {self.agent_id} started at {self.start_time}")
        
    def stop(self):
        """Stop the agent"""
        self.is_active = False
        self.logger.info(f"Agent {self.agent_id} stopped at {datetime.now()}")
        self.logger.info(f"Processed {self.processed_count} messages with an average processing time of {self.average_processing_time:.2f} seconds")
        self.logger.info(f"Encountered {self.error_count} errors during processing")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            "agent_id": self.agent_id,
            "is_active": self.is_active,
            "uptime_seconds": uptime,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "average_processing_time": self.average_processing_time,
            "success_rate": ((self.processed_count - self.error_count)/max(self.processed_count, 1)) * 100,
            "capabilities": self.get_capabilities()
        }
        
    def _update_performance_metrics(self, processing_time: float, success: bool):
        """Track the performance metrics of the agent"""
        self.processed_count += 1
        self.total_processing_time += processing_time
        self.average_processing_time = self.total_processing_time / self.processed_count

        if not success:
            self.error_count += 1
            
class AMLAgentBase(BaseAMLAgent):
    """Concrete base class with AML-specific functionality"""
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        
        # Data access setup
        self.data_root = Path(config.get("data_root", "data"))
        self.db_path = self.data_root / 'aml_local.db'
        
        # Load agent data
        self.agent_data = self._load_agent_data()
        self.logger = logging.getLogger(f"agent.{agent_id}")
        
    def _load_agent_data(self) -> pd.DataFrame:
        """Load agent data from the database"""
        try:
            # Try agent specific parquet first
            parquet_path = self.data_root / 'processed' / 'agent_datasets' /f'{self.agent_id}_data.parquet'
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                self.logger.info(f"Using parquet dataset for agent data from {parquet_path}")
                return df
            
            # Try agent specific csv
            csv_path = self.data_root / 'processed' / 'agent_datasets' /f'{self.agent_id}_data.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                self.logger.info(f"Using CSV dataset for agent data from {csv_path}")
                return df
            
            # Fallback to main dataset
            main_data_path = self.data_root / 'processed' / 'features' / 'feature_enhanced_data.parquet'
            if main_data_path.exists():
                df = pd.read_parquet(main_data_path)
                self.logger.info(f"Using main dataset for agent data from {main_data_path}")
                return df
            
            self.logger.warning(f"No dataset found for agent {self.agent_id}")
            return pd.DataFrame()
        
        except Exception as e:
            self.logger.error(f"Error loading agent data: {e}")
            return pd.DataFrame()
    
    def query_database(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute SQL queries against the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            return []
        
    def get_transaction(self, transaction_id: str) -> Optional[Dict]:
        """get specific trasnaction by ID"""
        results = self.query_database("SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,))
        
        return results[0] if results else None
    
    def get_account_history(self, account_id: str, limit: int = 100) -> List[Dict]:
        """get account history"""
        results = self.query_database("SELECT * FROM transactions WHERE originator_account = ? or beneficiary_account = ? ORDER BY transaction_date DESC LIMIT ?", (account_id, account_id, limit))

        return results
    