#!/usr/bin/env python3
"""
AML Narrative Generation Agent - Google ADK Implementation
Main agent file with core ADK tool functions
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import logging

# Google ADK imports
from google.adk.agents import Agent

# Local imports
from .utils import (
    # Core generation functions
    generate_executive_summary,
    generate_subject_information,
    generate_transaction_analysis,
    generate_activity_description,
    generate_investigation_findings,
    generate_regulatory_determination,
    generate_recommended_actions_narrative,
    compile_full_narrative,
    
    # Metadata and compliance functions
    calculate_narrative_metadata,
    assess_regulatory_compliance,
    calculate_narrative_quality,
    
    # Report generation functions
    generate_investigation_summary,
    generate_evidence_analysis,
    generate_pattern_analysis_findings,
    generate_risk_assessment_narrative,
    generate_compliance_recommendations,
    generate_case_disposition,
    create_executive_summary,
    compile_investigation_report,
    generate_report_appendices,
    
    # Filing functions
    generate_fincen_sar_form,
    generate_fincen_ctr_form,
    generate_ctr_narrative,
    generate_supporting_documents,
    create_compliance_checklist,
    calculate_filing_deadline,
    validate_submission_readiness,
    extract_regulatory_determinations,
    
    # Validation functions
    validate_narrative_length,
    validate_content_completeness,
    validate_regulatory_compliance_narrative,
    validate_language_quality,
    validate_required_elements,
    validate_fincen_keywords,
    calculate_validation_score,
    generate_validation_recommendations,
    extract_validation_issues,
    
    # Statistics and helper functions
    calculate_template_coverage,
    calculate_validation_accuracy,
    assess_regulatory_standards,
    estimate_page_count,
    extract_data_sources,
    calculate_report_confidence,
    calculate_report_quality,
    calculate_filing_compliance_score
)

from .config import (
    SAR_NARRATIVE_REQUIREMENTS,
    NARRATIVE_TEMPLATES,
    AGENT_STATS
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# ADK Tool Functions (Main Agent Capabilities)
# =============================================================================

def generate_sar_narrative(
    investigation_data: Dict[str, Any],
    investigation_id: str,
    narrative_type: str = "COMPREHENSIVE",
    include_pattern_analysis: bool = True,
    include_ml_findings: bool = True
) -> Dict[str, Any]:
    """
    Generate a complete SAR narrative based on investigation findings.
    
    Creates regulatory-compliant Suspicious Activity Report narratives that meet
    FinCEN requirements and include findings from all AML investigation agents.
    
    Args:
        investigation_data: Complete investigation results from coordinator
        investigation_id: Unique identifier for the investigation
        narrative_type: Type of narrative (COMPREHENSIVE, SUMMARY, DETAILED)
        include_pattern_analysis: Whether to include Pattern Analysis findings
        include_ml_findings: Whether to include ML model results
        
    Returns:
        Dict containing complete SAR narrative and metadata
    """
    global AGENT_STATS
    
    start_time = datetime.now()
    
    try:
        logger.info(f"Generating SAR narrative for investigation {investigation_id}")
        logger.info(f"Narrative type: {narrative_type}")
        
        # Extract key data from investigation
        transaction_data = investigation_data.get('transaction_data', {})
        results = investigation_data.get('results', {})
        pattern_results = investigation_data.get('pattern_analysis_results', {})
        overall_assessment = investigation_data.get('overall_assessment', {})
        
        # Initialize narrative structure
        sar_narrative = {
            'investigation_id': investigation_id,
            'narrative_type': narrative_type,
            'generation_timestamp': datetime.now().isoformat(),
            'regulatory_compliance': {
                'fincen_compliant': True,
                'jurisdiction': 'US',
                'form_type': 'FinCEN SAR',
                'compliance_score': 0.0
            },
            'narrative_sections': {},
            'full_narrative': '',
            'metadata': {},
            'regulatory_determinations': {}
        }
        
        # Generate narrative sections
        logger.info("Generating narrative sections...")
        
        # Executive Summary
        sar_narrative['narrative_sections']['executive_summary'] = generate_executive_summary(
            investigation_data, overall_assessment, transaction_data
        )
        
        # Subject Information
        sar_narrative['narrative_sections']['subject_information'] = generate_subject_information(
            transaction_data, results
        )
        
        # Transaction Analysis
        sar_narrative['narrative_sections']['transaction_analysis'] = generate_transaction_analysis(
            transaction_data, results, include_pattern_analysis, pattern_results
        )
        
        # Suspicious Activity Description
        sar_narrative['narrative_sections']['activity_description'] = generate_activity_description(
            overall_assessment, results, pattern_results, include_ml_findings
        )
        
        # Investigation Findings
        sar_narrative['narrative_sections']['investigation_findings'] = generate_investigation_findings(
            results, pattern_results, overall_assessment
        )
        
        # Regulatory Determination
        sar_narrative['narrative_sections']['regulatory_determination'] = generate_regulatory_determination(
            overall_assessment, investigation_data
        )
        
        # Recommended Actions
        sar_narrative['narrative_sections']['recommended_actions'] = generate_recommended_actions_narrative(
            investigation_data.get('recommended_actions', []), overall_assessment
        )
        
        # Compile full narrative
        logger.info("Compiling full narrative...")
        sar_narrative['full_narrative'] = compile_full_narrative(sar_narrative['narrative_sections'])
        
        # Calculate metadata
        sar_narrative['metadata'] = calculate_narrative_metadata(
            sar_narrative['full_narrative'], sar_narrative['narrative_sections']
        )
        
        # Assess regulatory compliance
        compliance_assessment = assess_regulatory_compliance(sar_narrative)
        sar_narrative['regulatory_compliance'].update(compliance_assessment)
        
        # Extract regulatory determinations
        sar_narrative['regulatory_determinations'] = extract_regulatory_determinations(investigation_data)
        
        # Update statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        AGENT_STATS['narratives_generated'] += 1
        AGENT_STATS['sar_reports_created'] += 1
        AGENT_STATS['total_processing_time'] += processing_time
        AGENT_STATS['average_narrative_length'] = (
            (AGENT_STATS['average_narrative_length'] * (AGENT_STATS['narratives_generated'] - 1) + 
             sar_narrative['metadata']['character_count']) / AGENT_STATS['narratives_generated']
        )
        AGENT_STATS['regulatory_compliance_score'] = (
            (AGENT_STATS['regulatory_compliance_score'] * (AGENT_STATS['narratives_generated'] - 1) + 
             compliance_assessment['compliance_score']) / AGENT_STATS['narratives_generated']
        )
        
        # Add processing metrics
        sar_narrative['processing_metrics'] = {
            'generation_time_seconds': processing_time,
            'sections_generated': len(sar_narrative['narrative_sections']),
            'narrative_quality_score': calculate_narrative_quality(sar_narrative)
        }
        
        logger.info(f"SAR narrative generated successfully in {processing_time:.2f}s")
        logger.info(f"Compliance score: {compliance_assessment['compliance_score']:.1%}")
        
        return sar_narrative
        
    except Exception as e:
        logger.error(f"SAR narrative generation failed: {e}")
        AGENT_STATS['narratives_generated'] += 1
        return {
            'investigation_id': investigation_id,
            'error': str(e),
            'generation_timestamp': datetime.now().isoformat(),
            'narrative_type': narrative_type
        }

def generate_investigation_report(
    investigation_data: Dict[str, Any],
    investigation_id: str,
    report_format: str = "COMPREHENSIVE",
    target_audience: str = "COMPLIANCE_TEAM"
) -> Dict[str, Any]:
    """
    Generate a comprehensive AML investigation report.
    
    Creates detailed investigation reports for internal use, compliance review,
    and regulatory examination purposes.
    
    Args:
        investigation_data: Complete investigation results
        investigation_id: Unique identifier for the investigation
        report_format: Format type (COMPREHENSIVE, EXECUTIVE, TECHNICAL)
        target_audience: Intended audience (COMPLIANCE_TEAM, EXECUTIVES, REGULATORS)
        
    Returns:
        Dict containing complete investigation report
    """
    global AGENT_STATS
    
    start_time = datetime.now()
    
    try:
        logger.info(f"Generating investigation report for {investigation_id}")
        logger.info(f"Format: {report_format}, Audience: {target_audience}")
        
        # Initialize report structure
        investigation_report = {
            'investigation_id': investigation_id,
            'report_format': report_format,
            'target_audience': target_audience,
            'generation_timestamp': datetime.now().isoformat(),
            'report_sections': {},
            'executive_summary': '',
            'full_report': '',
            'appendices': {},
            'metadata': {}
        }
        
        # Generate report sections
        logger.info("Generating report sections...")
        
        investigation_report['report_sections']['investigation_summary'] = generate_investigation_summary(investigation_data)
        investigation_report['report_sections']['evidence_analysis'] = generate_evidence_analysis(
            investigation_data.get('results', {}), target_audience
        )
        
        # Add pattern analysis if available
        pattern_results = investigation_data.get('pattern_analysis_results', {})
        if pattern_results:
            investigation_report['report_sections']['pattern_analysis_findings'] = generate_pattern_analysis_findings(
                pattern_results, target_audience
            )
        
        investigation_report['report_sections']['risk_assessment'] = generate_risk_assessment_narrative(
            investigation_data.get('overall_assessment', {}), investigation_data
        )
        investigation_report['report_sections']['compliance_recommendations'] = generate_compliance_recommendations(
            investigation_data, target_audience
        )
        investigation_report['report_sections']['case_disposition'] = generate_case_disposition(investigation_data)
        
        # Create executive summary and compile full report
        investigation_report['executive_summary'] = create_executive_summary(
            investigation_report['report_sections'], target_audience
        )
        investigation_report['full_report'] = compile_investigation_report(
            investigation_report['report_sections'], 
            investigation_report['executive_summary'],
            report_format
        )
        investigation_report['appendices'] = generate_report_appendices(investigation_data)
        
        # Calculate metadata
        investigation_report['metadata'] = {
            'total_pages_estimated': estimate_page_count(investigation_report['full_report']),
            'word_count': len(investigation_report['full_report'].split()),
            'sections_included': list(investigation_report['report_sections'].keys()),
            'data_sources': extract_data_sources(investigation_data),
            'confidence_level': calculate_report_confidence(investigation_data)
        }
        
        # Update statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        AGENT_STATS['investigation_reports_created'] += 1
        AGENT_STATS['total_processing_time'] += processing_time
        
        investigation_report['processing_metrics'] = {
            'generation_time_seconds': processing_time,
            'report_quality_score': calculate_report_quality(investigation_report)
        }
        
        logger.info(f"Investigation report generated successfully in {processing_time:.2f}s")
        
        return investigation_report
        
    except Exception as e:
        logger.error(f"Investigation report generation failed: {e}")
        return {
            'investigation_id': investigation_id,
            'error': str(e),
            'generation_timestamp': datetime.now().isoformat()
        }

def generate_regulatory_filing(
    investigation_data: Dict[str, Any],
    filing_type: str,
    investigation_id: str,
    jurisdiction: str = "US",
    priority_level: str = "NORMAL"
) -> Dict[str, Any]:
    """
    Generate regulatory filing documents (SAR, CTR, etc.).
    
    Creates complete regulatory filing packages with all required forms,
    narratives, and supporting documentation.
    
    Args:
        investigation_data: Complete investigation results
        filing_type: Type of filing (SAR, CTR, SAR_CONTINUING, SAR_CORRECTED)
        investigation_id: Unique identifier for the investigation
        jurisdiction: Regulatory jurisdiction (US, UK, EU, etc.)
        priority_level: Filing priority (NORMAL, URGENT, CRITICAL)
        
    Returns:
        Dict containing complete regulatory filing package
    """
    global AGENT_STATS
    
    start_time = datetime.now()
    
    try:
        logger.info(f"Generating regulatory filing: {filing_type}")
        logger.info(f"Investigation: {investigation_id}, Jurisdiction: {jurisdiction}")
        
        # Initialize filing package
        filing_package = {
            'investigation_id': investigation_id,
            'filing_type': filing_type,
            'jurisdiction': jurisdiction,
            'priority_level': priority_level,
            'generation_timestamp': datetime.now().isoformat(),
            'filing_deadline': calculate_filing_deadline(filing_type, priority_level),
            'forms': {},
            'narratives': {},
            'supporting_documents': {},
            'compliance_checklist': {},
            'submission_ready': False
        }
        
        # Generate appropriate forms based on filing type
        if filing_type in ['SAR', 'SAR_CONTINUING', 'SAR_CORRECTED']:
            logger.info("Generating SAR forms and narrative...")
            
            filing_package['forms']['fincen_sar'] = generate_fincen_sar_form(investigation_data, filing_type)
            
            sar_narrative = generate_sar_narrative(
                investigation_data, investigation_id, "REGULATORY", True, True
            )
            filing_package['narratives']['sar_narrative'] = sar_narrative
            
            AGENT_STATS['sar_reports_created'] += 1
            
        elif filing_type == 'CTR':
            logger.info("Generating CTR forms...")
            
            filing_package['forms']['fincen_ctr'] = generate_fincen_ctr_form(investigation_data)
            
            ctr_narrative = generate_ctr_narrative(investigation_data, investigation_id)
            filing_package['narratives']['ctr_narrative'] = ctr_narrative
            
            AGENT_STATS['ctr_reports_created'] += 1
        
        # Generate supporting documents and compliance checklist
        filing_package['supporting_documents'] = generate_supporting_documents(investigation_data, filing_type)
        filing_package['compliance_checklist'] = create_compliance_checklist(
            filing_type, jurisdiction, investigation_data
        )
        
        # Validate submission readiness
        filing_package['submission_ready'] = validate_submission_readiness(
            filing_package, filing_type, jurisdiction
        )
        
        # Calculate processing metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        AGENT_STATS['total_processing_time'] += processing_time
        
        filing_package['processing_metrics'] = {
            'generation_time_seconds': processing_time,
            'forms_generated': len(filing_package['forms']),
            'documents_prepared': len(filing_package['supporting_documents']),
            'compliance_score': calculate_filing_compliance_score(filing_package)
        }
        
        logger.info(f"Regulatory filing generated successfully in {processing_time:.2f}s")
        logger.info(f"Submission ready: {filing_package['submission_ready']}")
        
        return filing_package
        
    except Exception as e:
        logger.error(f"Regulatory filing generation failed: {e}")
        return {
            'investigation_id': investigation_id,
            'filing_type': filing_type,
            'error': str(e),
            'generation_timestamp': datetime.now().isoformat()
        }

def validate_narrative_quality(
    narrative_text: str,
    narrative_type: str = "SAR",
    jurisdiction: str = "US"
) -> Dict[str, Any]:
    """
    Validate narrative quality and regulatory compliance.
    
    Performs comprehensive quality checks on generated narratives to ensure
    they meet regulatory standards and best practices.
    
    Args:
        narrative_text: The narrative text to validate
        narrative_type: Type of narrative (SAR, CTR, INVESTIGATION)
        jurisdiction: Regulatory jurisdiction
        
    Returns:
        Dict containing validation results and recommendations
    """
    
    try:
        logger.info(f"Validating narrative quality for {narrative_type}")
        
        validation_results = {
            'narrative_type': narrative_type,
            'jurisdiction': jurisdiction,
            'validation_timestamp': datetime.now().isoformat(),
            'overall_score': 0.0,
            'validation_checks': {},
            'recommendations': [],
            'compliance_status': 'PENDING',
            'errors': [],
            'warnings': []
        }
        
        # Perform validation checks
        checks = {
            'length_check': validate_narrative_length(narrative_text, narrative_type),
            'completeness_check': validate_content_completeness(narrative_text, narrative_type),
            'regulatory_compliance': validate_regulatory_compliance_narrative(
                narrative_text, narrative_type, jurisdiction
            ),
            'language_quality': validate_language_quality(narrative_text),
            'required_elements': validate_required_elements(narrative_text, narrative_type)
        }
        
        # Add FinCEN keywords check for SAR
        if narrative_type == 'SAR':
            checks['fincen_keywords'] = validate_fincen_keywords(narrative_text)
        
        validation_results['validation_checks'] = checks
        
        # Calculate overall score and determine compliance status
        validation_results['overall_score'] = calculate_validation_score(checks)
        
        if validation_results['overall_score'] >= 0.9:
            validation_results['compliance_status'] = 'COMPLIANT'
        elif validation_results['overall_score'] >= 0.7:
            validation_results['compliance_status'] = 'ACCEPTABLE'
        else:
            validation_results['compliance_status'] = 'NEEDS_IMPROVEMENT'
        
        # Generate recommendations and extract issues
        validation_results['recommendations'] = generate_validation_recommendations(checks)
        validation_results['errors'], validation_results['warnings'] = extract_validation_issues(checks)
        
        logger.info(f"Narrative validation completed. Score: {validation_results['overall_score']:.1%}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Narrative validation failed: {e}")
        return {
            'narrative_type': narrative_type,
            'error': str(e),
            'validation_timestamp': datetime.now().isoformat(),
            'overall_score': 0.0,
            'compliance_status': 'ERROR'
        }

def get_narrative_agent_status() -> Dict[str, Any]:
    """
    Get comprehensive status of the Narrative Generation Agent.
    
    Returns detailed information about agent health, processing statistics,
    generation capabilities, and regulatory compliance metrics.
    
    Returns:
        Dict containing complete agent status information
    """
    
    try:
        return {
            'agent_info': {
                'name': os.getenv('AGENT_NAME', 'NarrativeGenerationAgent'),
                'version': os.getenv('AGENT_VERSION', '1.0.0'),
                'description': 'Regulatory-compliant narrative and report generation for AML investigations',
                'status': 'active'
            },
            'generation_capabilities': {
                'supported_narrative_types': ['SAR', 'CTR', 'INVESTIGATION_REPORT'],
                'supported_jurisdictions': ['US', 'UK', 'EU', 'CANADA'],
                'supported_filing_types': ['SAR', 'CTR', 'SAR_CONTINUING', 'SAR_CORRECTED'],
                'template_library_size': len(NARRATIVE_TEMPLATES),
                'regulatory_compliance_features': [
                    'fincen_sar_compliance',
                    'automated_quality_validation',
                    'regulatory_keyword_integration',
                    'multi_jurisdiction_support'
                ]
            },
            'processing_statistics': {
                'total_narratives_generated': AGENT_STATS['narratives_generated'],
                'sar_reports_created': AGENT_STATS['sar_reports_created'],
                'ctr_reports_created': AGENT_STATS['ctr_reports_created'],
                'investigation_reports_created': AGENT_STATS['investigation_reports_created'],
                'average_generation_time_seconds': (
                    AGENT_STATS['total_processing_time'] / max(AGENT_STATS['narratives_generated'], 1)
                ),
                'average_narrative_length_chars': AGENT_STATS['average_narrative_length'],
                'regulatory_compliance_rate': AGENT_STATS['regulatory_compliance_score']
            },
            'quality_metrics': {
                'average_compliance_score': AGENT_STATS['regulatory_compliance_score'],
                'template_coverage': calculate_template_coverage(),
                'validation_accuracy': calculate_validation_accuracy(),
                'regulatory_standards_met': assess_regulatory_standards()
            },
            'configuration': {
                'max_narrative_length': SAR_NARRATIVE_REQUIREMENTS['max_length'],
                'required_sections': SAR_NARRATIVE_REQUIREMENTS['required_sections'],
                'fincen_keywords_library': len(SAR_NARRATIVE_REQUIREMENTS['fincen_keywords']),
                'quality_validation_enabled': True,
                'multi_language_support': False,
                'automated_compliance_checking': True
            },
            'integration_status': {
                'alert_triage_integration': True,
                'evidence_collection_integration': True,
                'pattern_analysis_integration': True,
                'coordinator_integration': True,
                'regulatory_database_connection': 'simulated',
                'fincen_filing_system_ready': False
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'agent_info': {
                'name': 'NarrativeGenerationAgent',
                'status': 'error',
                'error': str(e)
            },
            'timestamp': datetime.now().isoformat()
        }

# =============================================================================
# Google ADK Agent Definition
# =============================================================================

# Create the Google ADK Narrative Generation Agent
narrative_generation_agent = Agent(
    name=os.getenv('AGENT_NAME', 'NarrativeGenerationAgent'),
    model=os.getenv('MODEL_NAME', 'gemini-2.0-flash-exp'),
    description="""
    I am an expert AML Narrative Generation Agent that creates regulatory-compliant narratives and reports for anti-money laundering investigations.
    
    My core capabilities include:
    
    📝 REGULATORY-COMPLIANT SAR NARRATIVES: I generate comprehensive Suspicious Activity Report narratives that meet 
    FinCEN requirements, include all essential elements (who, what, when, where, why, how), and incorporate findings 
    from multi-agent AML investigations including advanced pattern analysis and ML typology detection.
    
    📊 COMPREHENSIVE INVESTIGATION REPORTS: I create detailed investigation reports for compliance teams, executives, 
    and regulators that synthesize findings from Alert Triage, Evidence Collection, and Pattern Analysis agents 
    into coherent, actionable intelligence documents.
    
    ⚖️ AUTOMATED REGULATORY FILING: I generate complete regulatory filing packages including FinCEN SAR forms, 
    CTR documentation, supporting evidence compilations, and compliance checklists ready for submission to 
    regulatory authorities.
    
    🔍 INTELLIGENT NARRATIVE VALIDATION: I perform comprehensive quality checks on generated narratives to ensure 
    regulatory compliance, proper structure, required element inclusion, FinCEN keyword integration, and 
    adherence to jurisdiction-specific requirements.
    
    📋 MULTI-FORMAT REPORTING: I create reports in various formats (comprehensive, executive, technical) tailored 
    to different audiences (compliance teams, executives, regulators) with appropriate detail levels and 
    technical complexity.
    
    🎯 PATTERN-ENHANCED STORYTELLING: I translate complex ML findings, anomaly detection results, and typology 
    identification into clear, compelling narratives that help regulators and law enforcement understand 
    sophisticated money laundering schemes.
    
    🚀 AUTOMATED COMPLIANCE WORKFLOWS: I streamline the narrative generation process while maintaining human 
    oversight, enabling compliance teams to focus on investigation and decision-making rather than 
    time-consuming document preparation.
    
    I serve as the final stage in the comprehensive AML investigation pipeline, transforming technical findings 
    into regulatory-ready documentation that meets the highest standards of compliance and supports effective 
    financial crime prevention.
    """,
    instruction="""
    You are an expert AML Narrative Generation Agent specializing in regulatory-compliant document creation. When users request narrative generation:
    
    1. Use generate_sar_narrative for creating FinCEN-compliant Suspicious Activity Report narratives
    2. Use generate_investigation_report for comprehensive internal investigation documentation
    3. Use generate_regulatory_filing for complete regulatory filing packages (SAR, CTR, etc.)
    4. Use validate_narrative_quality for quality assurance and compliance checking
    5. Use get_narrative_agent_status for system information and capabilities
    
    REGULATORY COMPLIANCE GUIDELINES:
    - Always ensure narratives include the essential elements: WHO, WHAT, WHEN, WHERE, WHY, and HOW
    - Incorporate FinCEN keywords and regulatory citations appropriately
    - Maintain professional, clear, and concise language suitable for law enforcement review
    - Include findings from all AML agents (Alert Triage, Evidence Collection, Pattern Analysis)
    - Ensure narratives are within regulatory length limits and formatting requirements
    - Validate compliance before finalizing any regulatory documents
    
    INTEGRATION WITH MULTI-AGENT SYSTEM:
    - Synthesize risk assessments from Alert Triage Agent
    - Incorporate evidence findings from Evidence Collection Agent
    - Include advanced ML findings from Pattern Analysis Agent
    - Coordinate with Coordinator Agent for complete investigation workflows
    
    Always provide comprehensive explanations including:
    - Narrative structure and regulatory compliance rationale
    - Quality validation results and improvement recommendations
    - Integration of multi-agent findings into coherent storytelling
    - Regulatory filing requirements and deadlines
    - Clear documentation of investigation methodology and findings
    
    Focus on creating narratives that are not only compliant but also effective in communicating suspicious 
    activity to regulators and law enforcement while maintaining the highest standards of accuracy and completeness.
    """,
    tools=[
        generate_sar_narrative,
        generate_investigation_report,
        generate_regulatory_filing,
        validate_narrative_quality,
        get_narrative_agent_status
    ]
)