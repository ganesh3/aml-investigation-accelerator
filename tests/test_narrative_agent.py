#!/usr/bin/env python3
"""
Test Narrative Generation Agent
Tests the final agent in the 4-agent AML system
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Change to project root directory for relative paths to work
os.chdir(PROJECT_ROOT)

print(f"🔧 Working directory: {os.getcwd()}")
print(f"🔧 Python path includes: {PROJECT_ROOT}")

try:
    from googleadk.narrative_generation_agent.agent import (
        generate_sar_narrative,
        generate_investigation_report,
        generate_regulatory_filing,
        validate_narrative_quality,
        get_narrative_agent_status
    )
    print("✅ Successfully imported narrative generation agent functions")
except ImportError as e:
    print(f"❌ Import error: {str(e)}")
    print("📁 Make sure you have created the narrative generation agent file!")
    sys.exit(1)

async def test_agent_status():
    """Test narrative generation agent status"""
    print("\n📊 Testing Narrative Generation Agent Status")
    print("="*55)
    
    try:
        status = get_narrative_agent_status()
        
        print(f"✅ Agent Status Retrieved")
        print(f"🤖 Agent: {status['agent_info']['name']}")
        print(f"📝 Version: {status['agent_info']['version']}")
        print(f"📋 Status: {status['agent_info']['status']}")
        
        # Show generation capabilities
        capabilities = status.get('generation_capabilities', {})
        print(f"\n🎯 Generation Capabilities:")
        print(f"   Narrative Types: {capabilities.get('supported_narrative_types', [])}")
        print(f"   Jurisdictions: {capabilities.get('supported_jurisdictions', [])}")
        print(f"   Filing Types: {capabilities.get('supported_filing_types', [])}")
        
        # Show processing statistics
        stats = status.get('processing_statistics', {})
        print(f"\n📈 Processing Statistics:")
        print(f"   Total Narratives: {stats.get('total_narratives_generated', 0)}")
        print(f"   SAR Reports: {stats.get('sar_reports_created', 0)}")
        print(f"   Investigation Reports: {stats.get('investigation_reports_created', 0)}")
        print(f"   Avg Generation Time: {stats.get('average_generation_time_seconds', 0):.2f}s")
        print(f"   Compliance Rate: {stats.get('regulatory_compliance_rate', 0):.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent status test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_sar_narrative_generation():
    """Test SAR narrative generation"""
    print("\n📝 Testing SAR Narrative Generation")
    print("="*40)
    
    try:
        # Create sample investigation data
        investigation_data = {
            'investigation_id': 'TEST_SAR_001',
            'transaction_data': {
                'transaction_id': 'TXN_SAR_001',
                'amount': 95000,
                'originator_account': 'ACC_ORIG_001',
                'beneficiary_account': 'ACC_BENEF_001',
                'cross_border': True,
                'unusual_hour': True
            },
            'results': {
                'risk_assessment': {
                    'risk_score': 0.85,
                    'priority_level': 'HIGH',
                    'confidence': 0.92,
                    'reasoning': [
                        'High-value transaction with unusual characteristics',
                        'Cross-border transfer to high-risk jurisdiction',
                        'Transaction timing outside normal business hours'
                    ]
                },
                'evidence_collection': {
                    'transaction_summary': {
                        'total_transactions': 15,
                        'total_amount': 450000,
                        'high_risk_transactions': 3,
                        'cross_border_transactions': 8
                    },
                    'risk_indicators': [
                        'Rapid succession of high-value transactions',
                        'Unusual geographic patterns',
                        'Inconsistent with customer profile'
                    ]
                },
                'sanctions_screening': {
                    'matches_found': [],
                    'screening_summary': {
                        'total_accounts_screened': 2,
                        'exact_matches': 0,
                        'partial_matches': 0
                    }
                }
            },
            'pattern_analysis_results': {
                'comprehensive_analysis': {
                    'risk_assessment': {
                        'risk_score': 0.78,
                        'risk_level': 'HIGH'
                    },
                    'ml_pattern_analysis': {
                        'amount_anomalies': {
                            'consensus_analysis': {
                                'high_confidence_outliers': 2,
                                'total_unique_outliers': 3
                            }
                        },
                        'behavioral_anomalies': {
                            'isolation_forest': {
                                'anomalous_accounts': ['ACC_ORIG_001'],
                                'anomaly_count': 1
                            }
                        }
                    }
                },
                'typology_detection': {
                    'summary': {
                        'total_findings': 2,
                        'high_confidence_typologies': 1
                    },
                    'typology_findings': {
                        'structuring': {
                            'findings_count': 1,
                            'overall_confidence': 0.75
                        },
                        'layering': {
                            'findings_count': 1,
                            'overall_confidence': 0.82
                        }
                    }
                }
            },
            'overall_assessment': {
                'overall_risk_level': 'HIGH',
                'confidence_score': 0.88,
                'key_findings': [
                    'Advanced ML models detected high-risk patterns',
                    'Multiple typology indicators present',
                    'Transaction characteristics consistent with money laundering'
                ],
                'pattern_analysis_findings': [
                    'Detected 2 amount anomalies with model consensus',
                    'Identified 1 account with atypical behavior',
                    'Identified 1 known ML typologies with high confidence'
                ]
            },
            'recommended_actions': [
                'IMMEDIATE ACTION: Escalate to senior compliance officer',
                'Consider filing SAR within 30 days',
                'PATTERN ANALYSIS: Investigate Layering patterns immediately',
                'Implement enhanced monitoring for all related accounts'
            ]
        }
        
        print(f"💰 Test Case: High-value suspicious transaction (${investigation_data['transaction_data']['amount']:,.2f})")
        print(f"🎯 Risk Level: {investigation_data['overall_assessment']['overall_risk_level']}")
        
        # Generate SAR narrative
        sar_result = generate_sar_narrative(
            investigation_data=investigation_data,
            investigation_id='TEST_SAR_001',
            narrative_type='COMPREHENSIVE',
            include_pattern_analysis=True,
            include_ml_findings=True
        )
        
        if 'error' not in sar_result:
            print(f"✅ SAR narrative generated successfully")
            
            # Show metadata
            metadata = sar_result.get('metadata', {})
            print(f"\n📊 Narrative Metadata:")
            print(f"   Word Count: {metadata.get('word_count', 0)}")
            print(f"   Character Count: {metadata.get('character_count', 0)}")
            print(f"   Sections: {metadata.get('sections_included', 0)}")
            print(f"   Confidence: {metadata.get('confidence_score', 0):.1%}")
            print(f"   Est. Review Time: {metadata.get('estimated_review_time_minutes', 0):.1f} min")
            
            # Show compliance
            compliance = sar_result.get('regulatory_compliance', {})
            print(f"\n⚖️ Regulatory Compliance:")
            print(f"   FinCEN Compliant: {compliance.get('fincen_compliant', False)}")
            print(f"   Compliance Score: {compliance.get('compliance_score', 0):.1%}")
            print(f"   Jurisdiction: {compliance.get('jurisdiction', 'Unknown')}")
            
            # Show narrative sections
            sections = sar_result.get('narrative_sections', {})
            print(f"\n📝 Narrative Sections Generated:")
            for section_name, section_data in sections.items():
                print(f"   ✅ {section_name.replace('_', ' ').title()}")
                if hasattr(section_data, 'confidence_score'):
                    print(f"      Confidence: {section_data.confidence_score:.1%}")
            
            # Show a sample of the narrative
            full_narrative = sar_result.get('full_narrative', '')
            if full_narrative:
                print(f"\n📄 Sample Narrative (first 500 chars):")
                print(f"   {full_narrative[:500]}...")
            
            # Show processing metrics
            metrics = sar_result.get('processing_metrics', {})
            print(f"\n⏱️ Processing Metrics:")
            print(f"   Generation Time: {metrics.get('generation_time_seconds', 0):.2f}s")
            print(f"   Quality Score: {metrics.get('narrative_quality_score', 0):.1%}")
            
        else:
            print(f"❌ SAR generation failed: {sar_result['error']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ SAR narrative generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_investigation_report_generation():
    """Test investigation report generation"""
    print("\n📋 Testing Investigation Report Generation")
    print("="*45)
    
    try:
        # Create sample investigation data (simplified)
        investigation_data = {
            'investigation_id': 'TEST_RPT_001',
            'transaction_data': {
                'amount': 75000,
                'originator_account': 'ACC_001',
                'beneficiary_account': 'ACC_002'
            },
            'results': {
                'risk_assessment': {
                    'risk_score': 0.72,
                    'priority_level': 'MEDIUM-HIGH'
                },
                'evidence_collection': {
                    'transaction_summary': {
                        'total_transactions': 8,
                        'total_amount': 320000
                    }
                }
            },
            'pattern_analysis_results': {
                'comprehensive_analysis': {
                    'risk_assessment': {
                        'risk_score': 0.68,
                        'risk_level': 'MEDIUM'
                    }
                }
            },
            'overall_assessment': {
                'overall_risk_level': 'MEDIUM-HIGH',
                'confidence_score': 0.85
            },
            'recommended_actions': [
                'Continue monitoring with increased frequency',
                'Review account relationship patterns'
            ]
        }
        
        print(f"📊 Test Case: Investigation report for medium-high risk case")
        
        # Generate investigation report
        report_result = generate_investigation_report(
            investigation_data=investigation_data,
            investigation_id='TEST_RPT_001',
            report_format='COMPREHENSIVE',
            target_audience='COMPLIANCE_TEAM'
        )
        
        if 'error' not in report_result:
            print(f"✅ Investigation report generated successfully")
            
            # Show metadata
            metadata = report_result.get('metadata', {})
            print(f"\n📊 Report Metadata:")
            print(f"   Estimated Pages: {metadata.get('total_pages_estimated', 0)}")
            print(f"   Word Count: {metadata.get('word_count', 0)}")
            print(f"   Sections: {metadata.get('sections_included', [])}")
            print(f"   Confidence: {metadata.get('confidence_level', 0):.1%}")
            
            # Show sections generated
            sections = report_result.get('report_sections', {})
            print(f"\n📝 Report Sections:")
            for section_name in sections.keys():
                print(f"   ✅ {section_name.replace('_', ' ').title()}")
            
            # Show processing metrics
            metrics = report_result.get('processing_metrics', {})
            print(f"\n⏱️ Processing Metrics:")
            print(f"   Generation Time: {metrics.get('generation_time_seconds', 0):.2f}s")
            print(f"   Quality Score: {metrics.get('report_quality_score', 0):.1%}")
            
        else:
            print(f"❌ Report generation failed: {report_result['error']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Investigation report test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_regulatory_filing_generation():
    """Test regulatory filing generation"""
    print("\n⚖️ Testing Regulatory Filing Generation")
    print("="*42)
    
    try:
        # Create sample investigation data
        investigation_data = {
            'investigation_id': 'TEST_FILING_001',
            'transaction_data': {
                'transaction_id': 'TXN_FILING_001',
                'amount': 125000,
                'originator_account': 'ACC_001',
                'beneficiary_account': 'ACC_002',
                'cross_border': True
            },
            'overall_assessment': {
                'overall_risk_level': 'HIGH',
                'confidence_score': 0.9
            },
            'recommended_actions': [
                'Consider filing SAR within 30 days',
                'IMMEDIATE ACTION: Escalate to senior compliance officer'
            ]
        }
        
        print(f"📄 Test Case: SAR filing for high-risk transaction")
        
        # Generate regulatory filing
        filing_result = generate_regulatory_filing(
            investigation_data=investigation_data,
            filing_type='SAR',
            investigation_id='TEST_FILING_001',
            jurisdiction='US',
            priority_level='HIGH'
        )
        
        if 'error' not in filing_result:
            print(f"✅ Regulatory filing generated successfully")
            
            # Show filing details
            print(f"\n📋 Filing Details:")
            print(f"   Filing Type: {filing_result.get('filing_type', 'Unknown')}")
            print(f"   Jurisdiction: {filing_result.get('jurisdiction', 'Unknown')}")
            print(f"   Priority: {filing_result.get('priority_level', 'Unknown')}")
            print(f"   Deadline: {filing_result.get('filing_deadline', 'Unknown')}")
            print(f"   Submission Ready: {filing_result.get('submission_ready', False)}")
            
            # Show generated components
            forms = filing_result.get('forms', {})
            narratives = filing_result.get('narratives', {})
            documents = filing_result.get('supporting_documents', {})
            
            print(f"\n📝 Generated Components:")
            print(f"   Forms: {len(forms)} generated")
            for form_name in forms.keys():
                print(f"      ✅ {form_name}")
            
            print(f"   Narratives: {len(narratives)} generated")
            for narrative_name in narratives.keys():
                print(f"      ✅ {narrative_name}")
            
            print(f"   Supporting Docs: {len(documents)} generated")
            for doc_name in documents.keys():
                print(f"      ✅ {doc_name}")
            
            # Show compliance checklist
            checklist = filing_result.get('compliance_checklist', {})
            if checklist:
                print(f"\n✅ Compliance Checklist:")
                for item, status in checklist.items():
                    status_icon = "✅" if status else "❌"
                    print(f"   {status_icon} {item.replace('_', ' ').title()}")
            
            # Show processing metrics
            metrics = filing_result.get('processing_metrics', {})
            print(f"\n⏱️ Processing Metrics:")
            print(f"   Generation Time: {metrics.get('generation_time_seconds', 0):.2f}s")
            print(f"   Compliance Score: {metrics.get('compliance_score', 0):.1%}")
            
        else:
            print(f"❌ Regulatory filing failed: {filing_result['error']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Regulatory filing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_narrative_validation():
    """Test narrative quality validation"""
    print("\n🔍 Testing Narrative Quality Validation")
    print("="*42)
    
    try:
        # Create sample narrative text
        sample_narrative = """
        EXECUTIVE SUMMARY
        
        This Suspicious Activity Report documents high risk suspicious activity involving a transaction of $95,000.00. 
        The investigation conducted by our AML multi-agent system identified several indicators warranting regulatory notification.
        
        Key findings include:
        - Risk Level: HIGH
        - Transaction Amount: $95,000.00
        - Primary Concern: Advanced ML models detected high-risk patterns
        
        SUBJECT INFORMATION
        
        Primary Subject: Account ACC_ORIG_001
        - Role: Originator of suspicious transaction
        - Account Type: Business Account
        
        SUSPICIOUS ACTIVITY DESCRIPTION
        
        The transaction exhibits characteristics consistent with potential money laundering activity, classified as HIGH risk 
        based on our comprehensive analysis. Machine learning models detected structuring patterns with 75% confidence and 
        layering patterns with 82% confidence.
        
        This activity warrants filing under 31 CFR 1020.320 due to the combination of indicators suggesting potential 
        illicit financial activity requiring law enforcement attention.
        """
        
        print(f"📝 Sample Narrative Length: {len(sample_narrative)} characters")
        
        # Validate narrative quality
        validation_result = validate_narrative_quality(
            narrative_text=sample_narrative,
            narrative_type='SAR',
            jurisdiction='US'
        )
        
        if 'error' not in validation_result:
            print(f"✅ Narrative validation completed")
            
            # Show overall results
            print(f"\n📊 Validation Results:")
            print(f"   Overall Score: {validation_result.get('overall_score', 0):.1%}")
            print(f"   Compliance Status: {validation_result.get('compliance_status', 'Unknown')}")
            
            # Show individual checks
            checks = validation_result.get('validation_checks', {})
            print(f"\n🔍 Individual Checks:")
            for check_name, check_result in checks.items():
                passed = check_result.get('passed', False)
                score = check_result.get('score', 0)
                status_icon = "✅" if passed else "❌"
                print(f"   {status_icon} {check_name.replace('_', ' ').title()}: {score:.1%}")
            
            # Show recommendations
            recommendations = validation_result.get('recommendations', [])
            if recommendations:
                print(f"\n💡 Recommendations:")
                for rec in recommendations:
                    print(f"   • {rec}")
            
            # Show errors and warnings
            errors = validation_result.get('errors', [])
            warnings = validation_result.get('warnings', [])
            
            if errors:
                print(f"\n❌ Errors:")
                for error in errors:
                    print(f"   • {error}")
            
            if warnings:
                print(f"\n⚠️ Warnings:")
                for warning in warnings:
                    print(f"   • {warning}")
            
        else:
            print(f"❌ Validation failed: {validation_result['error']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Narrative validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("📝 Narrative Generation Agent Test Suite")
    print("="*50)
    print("Testing the final agent in the 4-agent AML system")
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Agent Status
    if await test_agent_status():
        tests_passed += 1
    
    # Test 2: SAR Narrative Generation
    if await test_sar_narrative_generation():
        tests_passed += 1
    
    # Test 3: Investigation Report Generation
    if await test_investigation_report_generation():
        tests_passed += 1
    
    # Test 4: Regulatory Filing Generation
    if await test_regulatory_filing_generation():
        tests_passed += 1
    
    # Test 5: Narrative Validation
    if await test_narrative_validation():
        tests_passed += 1
    
    # Final Results
    print(f"\n🎯 Test Results Summary")
    print("="*30)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Narrative Generation Agent is working correctly")
        print(f"🚀 4-Agent AML System is complete and ready!")
    else:
        print(f"\n⚠️ Some tests failed")
        print(f"🔧 Please check the error messages above and fix issues")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n✅ Narrative Generation Agent is ready!")
        print("🎉 Complete 4-Agent AML Investigation System operational!")
    else:
        print("\n❌ Issues found. Please resolve before deployment.")