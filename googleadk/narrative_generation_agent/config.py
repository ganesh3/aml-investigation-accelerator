#!/usr/bin/env python3
"""
AML Narrative Generation Agent - Configuration
Configuration settings, templates, and constants for narrative generation
"""

import os
from typing import Dict, Any, List
from datetime import datetime

# =============================================================================
# Agent Configuration
# =============================================================================

AGENT_CONFIG = {
    'name': 'NarrativeGenerationAgent',
    'version': '1.0.0',
    'description': 'Regulatory-compliant narrative and report generation for AML investigations',
    'model_name': os.getenv('MODEL_NAME', 'gemini-2.0-flash-exp'),
    'author': 'AML Investigation System',
    'created_date': '2025-06-22',
    'agent_type': 'narrative_generation',
    'execution_mode': 'production'
}

# =============================================================================
# SAR Narrative Requirements (FinCEN Compliance)
# =============================================================================

SAR_NARRATIVE_REQUIREMENTS = {
    'max_length': 50000,  # Maximum characters
    'min_length': 500,    # Minimum characters
    'target_length': 2500, # Target characters for optimal narrative
    'max_pages': 20,      # Maximum pages when printed
    'required_sections': [
        'executive_summary',
        'subject_information',
        'transaction_analysis',
        'activity_description',
        'investigation_findings',
        'regulatory_determination',
        'recommended_actions'
    ],
    'essential_elements': [
        'who',    # Who was involved
        'what',   # What happened
        'when',   # When did it occur
        'where',  # Where did it happen
        'why',    # Why is it suspicious
        'how'     # How was it detected
    ],
    'fincen_keywords': [
        'suspicious activity',
        'money laundering',
        'structuring',
        'layering',
        'placement',
        'integration',
        'unusual transaction',
        'risk factors',
        'compliance',
        'investigation',
        'financial crime',
        'typology',
        'red flags',
        'due diligence',
        'know your customer',
        'customer identification',
        'beneficial ownership',
        'politically exposed person',
        'sanctions screening',
        'wire transfer',
        'cash transaction',
        'correspondent banking',
        'shell company',
        'trade-based money laundering',
        'cyber crime',
        'fraud',
        'terrorist financing',
        'bulk cash smuggling',
        'currency exchange',
        'virtual currency',
        'cryptocurrency',
        'digital assets',
        'remittance',
        'money service business',
        'prepaid cards',
        'correspondent account',
        'concentration account',
        'payable through account',
        'foreign correspondent',
        'offshore banking',
        'tax haven',
        'nominee account',
        'straw man',
        'front company'
    ],
    'regulatory_citations': [
        '31 CFR 1020.320',
        '31 CFR 1010.311',
        '31 CFR 1010.314',
        'FinCEN SAR Instructions',
        'BSA Compliance Program Requirements',
        'USA PATRIOT Act',
        'Bank Secrecy Act',
        'Anti-Money Laundering Act of 2020',
        'FFIEC BSA/AML Examination Manual'
    ],
    'compliance_standards': {
        'fincen_form': 'FinCEN SAR (TD F 90-22.47)',
        'filing_deadline_days': 30,
        'continuing_activity_deadline_days': 120,
        'record_retention_years': 5,
        'confidentiality_required': True,
        'law_enforcement_sharing': 'permitted',
        'safe_harbor_protection': True
    }
}

# =============================================================================
# Narrative Templates
# =============================================================================

NARRATIVE_TEMPLATES = {
    'sar': {
        'executive_summary': """
EXECUTIVE SUMMARY

This Suspicious Activity Report documents {risk_level} risk suspicious activity involving {description}. 
The investigation conducted by our comprehensive AML multi-agent system identified several indicators 
warranting regulatory notification under Bank Secrecy Act requirements.

Key findings include:
- Risk Level: {risk_level}
- Transaction Amount: {amount}
- Primary Concern: {primary_concern}
- Investigation Priority: {investigation_priority}

{pattern_analysis_summary}

This report has been prepared in accordance with FinCEN requirements and BSA regulations, 
utilizing advanced machine learning models, behavioral analysis, network examination, 
and typology detection to assess the suspicious nature of this activity.
        """.strip(),
        
        'subject_information': """
SUBJECT INFORMATION

Primary Subject: {primary_subject}
- Role: {primary_role}
- Account Type: {primary_account_type}
- Transaction History: {primary_transaction_history}
- Risk Profile: {primary_risk_profile}
- Customer Since: {primary_customer_since}

{secondary_subject_info}

All subjects are being monitored for continuing suspicious activity. 
Additional subjects may be identified during ongoing investigation.
        """.strip(),
        
        'activity_description': """
SUSPICIOUS ACTIVITY DESCRIPTION

Nature of Suspicious Activity:
{activity_nature}

Risk Assessment Classification: {risk_classification}

Detailed Activity Analysis:
{detailed_analysis}

{typology_findings}

{ml_findings}

Supporting Evidence:
{supporting_evidence}

Conclusion:
Based on comprehensive analysis using advanced ML models, behavioral clustering, 
network analysis, and typology detection, this activity exhibits patterns consistent 
with potential money laundering and warrants regulatory notification under BSA requirements.
        """.strip(),
        
        'regulatory_determination': """
REGULATORY DETERMINATION

BSA Reporting Requirement Analysis:
This activity meets the criteria for Suspicious Activity Report filing under 
the Bank Secrecy Act and FinCEN requirements.

Specific Regulatory Triggers:
{regulatory_triggers}

Filing Requirements:
- SAR Filing Required: YES
- Filing Deadline: {filing_deadline}
- Continuing Activity: {continuing_activity_status}
- Law Enforcement Notification: {law_enforcement_required}

Regulatory Citations:
{regulatory_citations}

{enhanced_reporting_requirements}

Documentation Compliance:
This report includes all required elements per FinCEN SAR Instructions and 
supports our institution's commitment to preventing money laundering and terrorist financing.
        """.strip(),
        
        'investigation_conclusion': """
Based on the investigation findings, this activity warrants regulatory notification and 
demonstrates our institution's commitment to BSA compliance and financial crime prevention.

The multi-agent investigation system has provided comprehensive analysis ensuring 
thorough examination of all relevant factors and regulatory requirements.
        """.strip()
    },
    
    'ctr': {
        'transaction_summary': """
CTR FILING SUMMARY

Transaction Details:
- Date: {transaction_date}
- Amount: {transaction_amount}
- Type: {transaction_type}
- Account: {account_number}

This Currency Transaction Report is filed for a cash transaction exceeding $10,000 
in accordance with BSA reporting requirements under 31 CFR 1010.311.
        """.strip(),
        
        'customer_information': """
CUSTOMER INFORMATION

Customer Details:
- Name: {customer_name}
- Identification: {customer_id}
- Address: {customer_address}
- Occupation: {customer_occupation}

Transaction conducted by: {conductor_name}
Relationship to account: {relationship}
        """.strip()
    },
    
    'investigation_report': {
        'executive_summary': """
EXECUTIVE SUMMARY

This comprehensive AML investigation report documents the analysis of suspicious 
financial activity using our advanced multi-agent investigation system.

Investigation Highlights:
{investigation_highlights}

Methodology:
{investigation_methodology}

Key Findings:
{key_findings}

Recommendations:
{recommendations}
        """.strip(),
        
        'methodology': """
INVESTIGATION METHODOLOGY

Our investigation utilized a sophisticated 4-agent system:

1. Alert Triage Agent: ML-based risk assessment using ensemble models
2. Evidence Collection Agent: Comprehensive data gathering and sanctions screening
3. Pattern Analysis Agent: Advanced anomaly detection and typology identification
4. Narrative Generation Agent: Regulatory-compliant documentation

Each agent employed specialized analytical capabilities to ensure comprehensive 
examination of all relevant factors and regulatory compliance requirements.
        """.strip(),
        
        'conclusions': """
CONCLUSIONS AND RECOMMENDATIONS

Based on comprehensive multi-agent analysis, the following conclusions are drawn:

{conclusions}

Recommended Actions:
{recommended_actions}

Regulatory Compliance:
{regulatory_compliance}

This investigation demonstrates the effectiveness of our advanced AML detection 
capabilities and commitment to regulatory compliance.
        """.strip()
    }
}

# =============================================================================
# Quality Thresholds and Scoring
# =============================================================================

QUALITY_THRESHOLDS = {
    'narrative_quality': {
        'excellent': 0.95,
        'good': 0.85,
        'acceptable': 0.75,
        'needs_improvement': 0.65,
        'unacceptable': 0.50
    },
    'regulatory_compliance': {
        'fully_compliant': 0.95,
        'substantially_compliant': 0.85,
        'mostly_compliant': 0.75,
        'partially_compliant': 0.65,
        'non_compliant': 0.50
    },
    'content_completeness': {
        'complete': 1.0,
        'mostly_complete': 0.9,
        'adequate': 0.8,
        'incomplete': 0.7,
        'insufficient': 0.6
    },
    'validation_scores': {
        'length_adequacy': 0.8,
        'keyword_coverage': 0.6,
        'section_completeness': 0.9,
        'regulatory_language': 0.8,
        'professional_tone': 0.7
    }
}

# =============================================================================
# Filing Type Configurations
# =============================================================================

FILING_CONFIGURATIONS = {
    'SAR': {
        'form_type': 'FinCEN SAR',
        'form_number': 'TD F 90-22.47',
        'deadline_days': 30,
        'priority_deadline_days': 15,
        'required_sections': [
            'subject_information',
            'suspicious_activity_description', 
            'transaction_details',
            'supporting_documentation'
        ],
        'submission_methods': ['FinCEN SAR Portal', 'BSA E-Filing'],
        'follow_up_requirements': {
            'continuing_activity': True,
            'supplemental_sars': True,
            'law_enforcement_sharing': 'permitted'
        }
    },
    'CTR': {
        'form_type': 'FinCEN CTR',
        'form_number': 'FinCEN Form 112',
        'deadline_days': 15,
        'amount_threshold': 10000,
        'required_sections': [
            'transaction_information',
            'person_information',
            'conductor_information'
        ],
        'submission_methods': ['FinCEN CTR Portal', 'BSA E-Filing'],
        'aggregation_rules': {
            'multiple_transactions': True,
            'single_day': True,
            'related_transactions': True
        }
    },
    'SAR_CONTINUING': {
        'form_type': 'FinCEN SAR',
        'form_number': 'TD F 90-22.47',
        'deadline_days': 120,
        'reference_required': True,
        'original_sar_reference': True
    },
    'SAR_CORRECTED': {
        'form_type': 'FinCEN SAR',
        'form_number': 'TD F 90-22.47',
        'deadline_days': 30,
        'correction_type': True,
        'original_sar_reference': True,
        'correction_explanation': True
    }
}

# =============================================================================
# Jurisdiction-Specific Requirements
# =============================================================================

JURISDICTION_REQUIREMENTS = {
    'US': {
        'primary_regulator': 'FinCEN',
        'secondary_regulators': ['OCC', 'Federal Reserve', 'FDIC', 'NCUA'],
        'filing_system': 'BSA E-Filing',
        'currency': 'USD',
        'language': 'English',
        'timezone': 'Eastern',
        'business_days': [1, 2, 3, 4, 5],  # Monday-Friday
        'reporting_thresholds': {
            'SAR': 5000,  # USD
            'CTR': 10000,  # USD
            'FBAR': 10000,  # USD aggregate
            'Form_8300': 10000  # USD cash
        },
        'special_requirements': {
            'patriot_act_compliance': True,
            'ofac_screening': True,
            'beneficial_ownership': True,
            'correspondent_banking': True
        }
    },
    'UK': {
        'primary_regulator': 'FCA',
        'secondary_regulators': ['PRA', 'HMRC'],
        'filing_system': 'SAR Online',
        'currency': 'GBP',
        'language': 'English',
        'timezone': 'GMT',
        'business_days': [1, 2, 3, 4, 5],
        'reporting_thresholds': {
            'SAR': 0,  # No minimum threshold
            'CTR_equivalent': 10000  # GBP
        },
        'special_requirements': {
            'proceeds_of_crime_act': True,
            'terrorism_act_compliance': True,
            'money_laundering_regulations': True
        }
    },
    'EU': {
        'primary_regulator': 'EBA',
        'filing_system': 'National FIUs',
        'currency': 'EUR',
        'language': 'Multiple',
        'timezone': 'CET',
        'business_days': [1, 2, 3, 4, 5],
        'reporting_thresholds': {
            'SAR': 0,  # No minimum threshold
            'CTR_equivalent': 10000  # EUR
        },
        'special_requirements': {
            'amld_compliance': True,
            'gdpr_compliance': True,
            'cross_border_reporting': True
        }
    },
    'CANADA': {
        'primary_regulator': 'FINTRAC',
        'filing_system': 'F2R Online',
        'currency': 'CAD',
        'language': 'English/French',
        'timezone': 'Eastern',
        'business_days': [1, 2, 3, 4, 5],
        'reporting_thresholds': {
            'STR': 0,  # No minimum threshold
            'CTR_equivalent': 10000  # CAD
        },
        'special_requirements': {
            'pcmltfa_compliance': True,
            'client_identification': True,
            'beneficial_ownership': True
        }
    }
}

# =============================================================================
# Typology Descriptions and Templates
# =============================================================================

TYPOLOGY_DESCRIPTIONS = {
    'STRUCTURING': {
        'name': 'Structuring/Smurfing',
        'description': 'Breaking large amounts into smaller transactions to avoid reporting thresholds',
        'indicators': [
            'Multiple transactions just below reporting thresholds',
            'Coordinated timing across multiple accounts',
            'Similar transaction amounts',
            'Use of multiple locations or individuals'
        ],
        'narrative_template': """
Structuring Pattern Detected:
The investigation identified {findings_count} instances of potential structuring activity involving 
transactions totaling ${total_amount:,.2f} within a {time_window} period. The transactions 
exhibit characteristics consistent with deliberate structuring to avoid BSA reporting requirements.

Specific Evidence:
- {transaction_count} transactions averaging ${average_amount:,.2f}
- {proximity_percentage}% of transactions within {proximity_threshold}% of reporting thresholds
- Coordinated timing suggesting intentional structuring

This pattern significantly elevates the suspicion level and supports the determination 
that this activity is designed to evade regulatory detection.
        """.strip()
    },
    
    'LAYERING': {
        'name': 'Layering',
        'description': 'Complex series of transactions through multiple intermediaries to obscure money trail',
        'indicators': [
            'Multiple transaction layers',
            'Rapid movement between accounts',
            'Complex routing patterns',
            'Minimal legitimate business purpose'
        ],
        'narrative_template': """
Layering Pattern Detected:
Advanced pattern analysis identified {findings_count} instances of layering activity involving 
transaction chains of {average_chain_length} intermediaries. The complex routing appears 
designed to obscure the ultimate source and destination of funds.

Chain Analysis:
- Maximum chain length: {max_chain_length} intermediaries
- Average amount consistency: {amount_consistency:.1%}
- Total funds layered: ${total_amount:,.2f}
- Time span: {time_span} hours

The systematic nature of these transaction chains, combined with the lack of apparent 
legitimate business purpose, strongly indicates money laundering activity.
        """.strip()
    },
    
    'ROUND_TRIPPING': {
        'name': 'Round Tripping',
        'description': 'Circular transaction flows that return funds to the originating account',
        'indicators': [
            'Circular transaction patterns',
            'Funds returning to originator',
            'Minimal net change in balances',
            'Complex intermediary routing'
        ],
        'narrative_template': """
Round Tripping Pattern Detected:
Network analysis identified {findings_count} instances of circular transaction flows where 
funds ultimately return to the originating account. This pattern suggests deliberate 
obfuscation of fund sources through complex routing.

Circular Flow Analysis:
- Average cycle length: {average_cycle_length} accounts
- Net change ratio: {net_change_ratio:.1%}
- Total flow amount: ${total_amount:,.2f}
- Completion rate: {completion_rate:.1%}

The minimal net change combined with complex routing strongly indicates these transactions 
serve no legitimate business purpose and are designed to create a false audit trail.
        """.strip()
    },
    
    'SMURFING': {
        'name': 'Smurfing',
        'description': 'Using multiple accounts or individuals to conduct coordinated transactions',
        'indicators': [
            'Coordinated timing across accounts',
            'Similar transaction amounts',
            'Multiple participants',
            'Synchronized activity patterns'
        ],
        'narrative_template': """
Smurfing Pattern Detected:
The investigation identified {findings_count} instances of coordinated activity across 
{account_count} accounts within {time_window} minute windows. The synchronized nature 
of these transactions indicates organized coordination to circumvent detection.

Coordination Evidence:
- Average coordination window: {coordination_window} minutes
- Amount similarity: {amount_similarity:.1%}
- Participating accounts: {participating_accounts}
- Total coordinated amount: ${total_amount:,.2f}

The high degree of coordination and timing precision strongly suggests these transactions 
are part of an organized scheme to evade regulatory detection through distributed activity.
        """.strip()
    }
}

# =============================================================================
# Risk Level Configurations
# =============================================================================

RISK_LEVEL_CONFIG = {
    'CRITICAL': {
        'score_range': (0.9, 1.0),
        'color': '#DC2626',  # Red
        'priority': 1,
        'response_time_hours': 4,
        'escalation_required': True,
        'management_notification': 'IMMEDIATE',
        'description': 'Extremely high probability of money laundering activity requiring immediate action'
    },
    'HIGH': {
        'score_range': (0.7, 0.89),
        'color': '#EA580C',  # Orange-Red
        'priority': 2,
        'response_time_hours': 24,
        'escalation_required': True,
        'management_notification': 'SAME_DAY',
        'description': 'High probability of suspicious activity requiring prompt investigation'
    },
    'MEDIUM': {
        'score_range': (0.5, 0.69),
        'color': '#D97706',  # Orange
        'priority': 3,
        'response_time_hours': 72,
        'escalation_required': False,
        'management_notification': 'NEXT_BUSINESS_DAY',
        'description': 'Moderate risk level requiring standard investigation procedures'
    },
    'LOW': {
        'score_range': (0.3, 0.49),
        'color': '#16A34A',  # Green
        'priority': 4,
        'response_time_hours': 168,
        'escalation_required': False,
        'management_notification': 'WEEKLY_SUMMARY',
        'description': 'Low risk level suitable for routine monitoring'
    },
    'MINIMAL': {
        'score_range': (0.0, 0.29),
        'color': '#059669',  # Dark Green
        'priority': 5,
        'response_time_hours': 720,
        'escalation_required': False,
        'management_notification': 'MONTHLY_SUMMARY',
        'description': 'Minimal risk level requiring basic documentation only'
    }
}

# =============================================================================
# Processing Configurations
# =============================================================================

PROCESSING_CONFIG = {
    'max_processing_time_seconds': 300,  # 5 minutes
    'max_narrative_generation_time': 120,  # 2 minutes
    'max_validation_time': 60,  # 1 minute
    'max_filing_preparation_time': 180,  # 3 minutes
    'concurrent_sections': True,
    'enable_caching': True,
    'cache_ttl_hours': 24,
    'batch_processing': {
        'enabled': True,
        'max_batch_size': 50,
        'batch_timeout_minutes': 30
    },
    'retry_configuration': {
        'max_retries': 3,
        'retry_delay_seconds': 5,
        'exponential_backoff': True
    },
    'memory_limits': {
        'max_memory_mb': 512,
        'cleanup_frequency': 10,
        'garbage_collection': True
    }
}

# =============================================================================
# Output Format Configurations
# =============================================================================

OUTPUT_FORMATS = {
    'json': {
        'extension': '.json',
        'mime_type': 'application/json',
        'encoding': 'utf-8',
        'pretty_print': True,
        'include_metadata': True
    },
    'txt': {
        'extension': '.txt',
        'mime_type': 'text/plain',
        'encoding': 'utf-8',
        'line_endings': 'CRLF',
        'include_headers': True
    },
    'docx': {
        'extension': '.docx',
        'mime_type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'template_available': False,  # Would require python-docx
        'formatting': {
            'font_family': 'Times New Roman',
            'font_size': 12,
            'line_spacing': 1.15,
            'margins': '1 inch'
        }
    },
    'pdf': {
        'extension': '.pdf',
        'mime_type': 'application/pdf',
        'template_available': False,  # Would require reportlab
        'formatting': {
            'page_size': 'Letter',
            'margins': '1 inch',
            'font_family': 'Times-Roman',
            'font_size': 12
        }
    }
}

# =============================================================================
# Agent Statistics Tracking
# =============================================================================

AGENT_STATS = {
    'narratives_generated': 0,
    'sar_reports_created': 0,
    'ctr_reports_created': 0,
    'investigation_reports_created': 0,
    'total_processing_time': 0.0,
    'average_narrative_length': 0,
    'regulatory_compliance_score': 0.0,
    'validation_errors_total': 0,
    'successful_filings': 0,
    'failed_filings': 0,
    'template_usage': {},
    'jurisdiction_distribution': {},
    'risk_level_distribution': {},
    'performance_metrics': {
        'average_generation_time': 0.0,
        'average_validation_time': 0.0,
        'cache_hit_rate': 0.0,
        'error_rate': 0.0
    }
}

# =============================================================================
# Error Messages and Codes
# =============================================================================

ERROR_MESSAGES = {
    'NR001': 'Investigation data missing or incomplete',
    'NR002': 'Invalid narrative type specified',
    'NR003': 'Template not found for requested narrative type',
    'NR004': 'Narrative validation failed - does not meet minimum requirements',
    'NR005': 'Regulatory compliance check failed',
    'NR006': 'Filing preparation failed - missing required components',
    'NR007': 'Export format not supported',
    'NR008': 'Processing timeout exceeded',
    'NR009': 'Memory limit exceeded during generation',
    'NR010': 'Template rendering failed',
    'NR011': 'Validation timeout exceeded',
    'NR012': 'Jurisdiction requirements not met',
    'NR013': 'Filing deadline calculation failed',
    'NR014': 'Submission package validation failed',
    'NR015': 'Audit trail creation failed'
}

WARNING_MESSAGES = {
    'NW001': 'Narrative length approaching maximum limit',
    'NW002': 'Some pattern analysis results missing',
    'NW003': 'Template placeholders not fully populated',
    'NW004': 'Validation score below optimal threshold',
    'NW005': 'Processing time exceeding target',
    'NW006': 'Cache miss - using real-time generation',
    'NW007': 'Some regulatory keywords missing',
    'NW008': 'Cross-border requirements may apply',
    'NW009': 'Enhanced due diligence recommended',
    'NW010': 'Multiple typologies detected - review carefully'
}

# =============================================================================
# Feature Flags and Toggles
# =============================================================================

FEATURE_FLAGS = {
    'enable_advanced_templates': True,
    'enable_multi_language': False,  # Future feature
    'enable_docx_export': False,     # Requires python-docx
    'enable_pdf_export': False,      # Requires reportlab
    'enable_template_caching': True,
    'enable_parallel_processing': True,
    'enable_validation_caching': True,
    'enable_audit_logging': True,
    'enable_performance_monitoring': True,
    'enable_automated_filing': False,  # Requires external API
    'enable_regulatory_updates': False,  # Future feature
    'enable_machine_translation': False,  # Future feature
    'enable_voice_generation': False,     # Future feature
    'enable_blockchain_timestamping': False  # Future feature
}

# =============================================================================
# Integration Settings
# =============================================================================

INTEGRATION_SETTINGS = {
    'alert_triage_agent': {
        'enabled': True,
        'risk_score_weight': 0.4,
        'confidence_threshold': 0.7,
        'include_model_details': True
    },
    'evidence_collection_agent': {
        'enabled': True,
        'evidence_weight': 0.3,
        'include_transaction_details': True,
        'include_sanctions_results': True
    },
    'pattern_analysis_agent': {
        'enabled': True,
        'pattern_weight': 0.3,
        'include_ml_findings': True,
        'include_typology_results': True,
        'include_network_analysis': True
    },
    'coordinator_agent': {
        'enabled': True,
        'orchestration_support': True,
        'workflow_integration': True,
        'status_reporting': True
    },
    'external_systems': {
        'fincen_filing_system': {
            'enabled': False,
            'api_endpoint': None,
            'authentication': None,
            'timeout_seconds': 60
        },
        'regulatory_database': {
            'enabled': False,
            'connection_string': None,
            'read_timeout': 30
        },
        'document_management': {
            'enabled': False,
            'storage_path': None,
            'retention_policy': 'regulatory_required'
        }
    }
}

# =============================================================================
# Security and Privacy Settings
# =============================================================================

SECURITY_SETTINGS = {
    'data_encryption': {
        'encrypt_pii': True,
        'encrypt_narratives': False,
        'encryption_algorithm': 'AES-256',
        'key_rotation_days': 90
    },
    'access_control': {
        'require_authentication': True,
        'role_based_access': True,
        'audit_access': True,
        'session_timeout_minutes': 60
    },
    'data_privacy': {
        'anonymize_accounts': False,  # Set True for demo/test
        'mask_sensitive_data': False,
        'gdpr_compliance': True,
        'data_retention_days': 1825  # 5 years
    },
    'audit_requirements': {
        'log_all_access': True,
        'log_generation_activities': True,
        'log_validation_results': True,
        'log_filing_activities': True,
        'audit_trail_encryption': True
    }
}

# =============================================================================
# Environment-Specific Overrides
# =============================================================================

ENVIRONMENT_OVERRIDES = {
    'development': {
        'processing_config': {
            'max_processing_time_seconds': 600,  # Extended for debugging
            'enable_caching': False,  # Disable for testing
            'retry_configuration': {
                'max_retries': 1  # Faster failure for development
            }
        },
        'security_settings': {
            'data_privacy': {
                'anonymize_accounts': True,  # Protect dev data
                'mask_sensitive_data': True
            }
        },
        'feature_flags': {
            'enable_performance_monitoring': False,  # Reduce overhead
            'enable_audit_logging': False
        }
    },
    'testing': {
        'processing_config': {
            'max_processing_time_seconds': 30,  # Fast for testing
            'enable_caching': False,
            'concurrent_sections': False  # Sequential for testing
        },
        'sar_narrative_requirements': {
            'min_length': 100,  # Shorter for testing
            'max_length': 5000
        }
    },
    'production': {
        'processing_config': {
            'max_processing_time_seconds': 300,
            'enable_caching': True,
            'concurrent_sections': True
        },
        'security_settings': {
            'audit_requirements': {
                'log_all_access': True,
                'audit_trail_encryption': True
            }
        },
        'feature_flags': {
            'enable_performance_monitoring': True,
            'enable_audit_logging': True
        }
    }
}

# =============================================================================
# Validation and Initialization
# =============================================================================

def validate_config() -> Dict[str, Any]:
    """Validate configuration completeness and consistency"""
    
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'missing_components': []
    }
    
    # Check required sections
    required_components = [
        'AGENT_CONFIG',
        'SAR_NARRATIVE_REQUIREMENTS', 
        'NARRATIVE_TEMPLATES',
        'QUALITY_THRESHOLDS',
        'FILING_CONFIGURATIONS',
        'PROCESSING_CONFIG'
    ]
    
    current_module = globals()
    for component in required_components:
        if component not in current_module:
            validation_results['missing_components'].append(component)
            validation_results['valid'] = False
    
    # Validate template completeness
    if 'NARRATIVE_TEMPLATES' in current_module:
        templates = current_module['NARRATIVE_TEMPLATES']
        required_sar_sections = SAR_NARRATIVE_REQUIREMENTS['required_sections']
        
        sar_templates = templates.get('sar', {})
        for section in required_sar_sections:
            if section not in sar_templates:
                validation_results['warnings'].append(f"Missing SAR template for section: {section}")
    
    # Validate risk levels
    if 'RISK_LEVEL_CONFIG' in current_module:
        risk_config = current_module['RISK_LEVEL_CONFIG']
        score_ranges = [config['score_range'] for config in risk_config.values()]
        
        # Check for gaps or overlaps in score ranges
        sorted_ranges = sorted(score_ranges, key=lambda x: x[0])
        for i in range(len(sorted_ranges) - 1):
            if sorted_ranges[i][1] < sorted_ranges[i+1][0]:
                validation_results['warnings'].append("Gap detected in risk level score ranges")
            elif sorted_ranges[i][1] > sorted_ranges[i+1][0]:
                validation_results['errors'].append("Overlap detected in risk level score ranges")
                validation_results['valid'] = False
    
    return validation_results

def initialize_agent_config() -> Dict[str, Any]:
    """Initialize agent configuration with environment-specific settings"""
    
    import os
    
    # Get environment
    environment = os.getenv('ENVIRONMENT', 'development').lower()
    
    # Apply environment-specific overrides
    if environment in ENVIRONMENT_OVERRIDES:
        overrides = ENVIRONMENT_OVERRIDES[environment]
        
        # Apply overrides to global configurations
        for config_name, override_values in overrides.items():
            if config_name in globals():
                current_config = globals()[config_name]
                if isinstance(current_config, dict) and isinstance(override_values, dict):
                    current_config.update(override_values)
    
    # Set environment variables
    config_status = {
        'environment': environment,
        'agent_name': AGENT_CONFIG['name'],
        'version': AGENT_CONFIG['version'],
        'initialization_time': datetime.now().isoformat(),
        'config_validation': validate_config(),
        'features_enabled': sum(1 for flag in FEATURE_FLAGS.values() if flag),
        'total_templates': len(NARRATIVE_TEMPLATES.get('sar', {})) + len(NARRATIVE_TEMPLATES.get('ctr', {})),
        'supported_jurisdictions': list(JURISDICTION_REQUIREMENTS.keys()),
        'supported_filings': list(FILING_CONFIGURATIONS.keys())
    }
    
    return config_status

# Initialize configuration on module load
CONFIG_STATUS = initialize_agent_config()

# Export configuration validation status
def get_config_status() -> Dict[str, Any]:
    """Get current configuration status"""
    return CONFIG_STATUS

# Configuration completeness check
def check_config_completeness() -> float:
    """Check configuration completeness percentage"""
    
    validation = validate_config()
    if validation['valid'] and not validation['missing_components']:
        return 1.0
    else:
        total_components = 10  # Expected number of major components
        missing_count = len(validation['missing_components'])
        return max(0.0, (total_components - missing_count) / total_components)

# Export final status
CONFIGURATION_COMPLETE = check_config_completeness() >= 0.95

if CONFIGURATION_COMPLETE:
    print("✅ Narrative Generation Agent configuration is complete")
else:
    print(f"⚠️ Configuration completeness: {check_config_completeness():.1%}")

# Export key configurations for external access
__all__ = [
    'AGENT_CONFIG',
    'SAR_NARRATIVE_REQUIREMENTS',
    'NARRATIVE_TEMPLATES', 
    'QUALITY_THRESHOLDS',
    'FILING_CONFIGURATIONS',
    'JURISDICTION_REQUIREMENTS',
    'TYPOLOGY_DESCRIPTIONS',
    'RISK_LEVEL_CONFIG',
    'PROCESSING_CONFIG',
    'AGENT_STATS',
    'get_config_status',
    'validate_config',
    'CONFIGURATION_COMPLETE'
]