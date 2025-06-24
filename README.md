# 🏗️ Complete AML Multi-Agent Architecture

## System Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                    AML Investigation Accelerator                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📊 Data Layer                                                  │
│  ├── SQLite Database (6,000 transactions)                      │
│  ├── Parquet Files (feature-enhanced datasets)                 │
│  └── Sanctions Data (OFAC lists)                               │
│                                                                 │
│  🤖 Agent Layer (5 Specialized Agents)                         │
│  ├── 🚨 Alert Triage Agent                                     │
│  │   ├── Risk Scoring (ML + Rules)                             │
│  │   ├── Priority Assignment                                   │
│  │   └── Auto Escalation/Dismissal                             │
│  │                                                             │
│  ├── 🔍 Evidence Collection Agent                              │
│  │   ├── Transaction History Gathering                         │
│  │   ├── Account Profile Building                              │
│  │   └── External Data Lookup                                  │
│  │                                                             │
│  ├── 🕸️ Pattern Analysis Agent                                │
│  │   ├── Network Analysis                                      │
│  │   ├── Temporal Pattern Detection                            │
│  │   └── ML Typology Classification                            │
│  │                                                             │
│  ├── 📝 Narrative Generation Agent                             │
│  │   ├── SAR Report Generation                                 │
│  │   ├── Investigation Summaries                               │
│  │   └── Compliance Documentation                              │
│  │                                                             │
│  └── 🎯 Coordination Agent                                     │
│      ├── Workflow Orchestration                                │
│      ├── Agent Communication                                   │
│      └── Escalation Management                                 │
│                                                                 │
│  🌐 API Layer                                                  │
│  ├── FastAPI REST Endpoints                                    │
│  ├── Agent Communication Hub                                   │
│  └── Status Monitoring                                         │
│                                                                 │
│  💻 User Interface                                             │
│  ├── Streamlit Dashboard                                       │
│  ├── Investigation Workspace                                   │
│  └── Real-time Monitoring                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Interaction Flow

### Investigation Workflow
```
1. 🚨 New Alert Arrives
   │
   ▼
2. 🤖 Alert Triage Agent
   ├── Calculates Risk Score (0.0 - 1.0)
   ├── Assigns Priority (LOW/MEDIUM/HIGH/CRITICAL)
   └── Decides Next Action
   │
   ▼
3. 🔍 Evidence Collection Agent (if not auto-dismissed)
   ├── Gathers Transaction History
   ├── Builds Account Profiles
   ├── Checks Sanctions Lists
   └── Compiles Evidence Package
   │
   ▼
4. 🕸️ Pattern Analysis Agent (for complex cases)
   ├── Network Analysis (who's connected?)
   ├── Temporal Analysis (when do transactions occur?)
   ├── Amount Analysis (structuring patterns?)
   └── Typology Classification (what type of ML?)
   │
   ▼
5. 📝 Narrative Generation Agent
   ├── Synthesizes All Findings
   ├── Generates SAR Narrative
   ├── Creates Investigation Summary
   └── Formats for Compliance
   │
   ▼
6. 🎯 Final Decision & Actions
   ├── File SAR (if required)
   ├── Close Investigation
   ├── Escalate to Human
   └── Archive Case
```

## Message Flow Between Agents

### Example: High-Risk Alert Processing
```python
# 1. Coordinator receives new alert
alert_message = {
    "type": "ALERT_RECEIVED",
    "transaction_id": "TXN_001", 
    "amount": 95000,
    "sender": "investigation_system",
    "priority": 5
}

# 2. Coordinator sends to Triage Agent
triage_request = {
    "type": "RISK_ASSESSMENT_REQUEST",
    "payload": alert_message,
    "sender": "coordinator",
    "receiver": "alert_triage"
}

# 3. Triage Agent responds
triage_response = {
    "type": "RISK_ASSESSMENT_RESPONSE", 
    "risk_score": 0.92,
    "priority": "CRITICAL",
    "action": "ESCALATE",
    "reasoning": ["High amount", "Cross-border", "Unusual timing"]
}

# 4. If high risk, Evidence Collection starts
evidence_request = {
    "type": "EVIDENCE_REQUEST",
    "transaction_id": "TXN_001",
    "scope": "FULL_INVESTIGATION"
}

# 5. Pattern Analysis runs in parallel
pattern_request = {
    "type": "PATTERN_ANALYSIS_REQUEST", 
    "focus": ["NETWORK", "TEMPORAL", "AMOUNT"]
}

# 6. Finally, Narrative Generation
narrative_request = {
    "type": "NARRATIVE_REQUEST",
    "evidence": evidence_data,
    "patterns": pattern_data,
    "format": "SAR_REPORT"
}
```

## Technology Stack

### Local Development Stack
```
📁 Project Structure
├── 🐍 Python 3.8+ (Core language)
├── 🐼 Pandas (Data processing)
├── 🤖 Scikit-learn (Machine learning)
├── ⚡ FastAPI (Web framework)
├── 🗄️ SQLite (Local database)
├── 📊 Streamlit (User interface)
└── 🔧 Google ADK (Agent framework)
```

### Google Cloud Stack (Deployment)
```
☁️ Google Cloud Platform
├── 🧠 Vertex AI (Agent hosting)
├── 📊 BigQuery (Data warehouse)
├── 🏃 Cloud Run (API hosting)
├── 🔒 Secret Manager (Credentials)
├── 📱 Document AI (Document processing)
└── 🌐 API Gateway (Traffic management)
```