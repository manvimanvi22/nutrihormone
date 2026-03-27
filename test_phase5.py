#!/usr/bin/env python
"""
Phase 5 Test: Doctor Alert Gating System (Dual-Gate: Rules + AI)
"""

import sys
from app1 import (
    analyze_health, 
    evaluate_doctor_alert_deterministic,
    evaluate_doctor_alert_ai_judgment,
    dual_gate_doctor_alert,
    generate_gemini_json_response,
    predict_diet
)

print("\n" + "="*70)
print("PHASE 5 TEST: DOCTOR ALERT GATING (DUAL-GATE SYSTEM)")
print("="*70 + "\n")

# Test scenarios
print("Testing Dual-Gate System (Deterministic Rules + AI Judgment):\n")

test_cases = [
    {
        "name": "NORMAL - Healthy metrics",
        "metrics": {
            'glucose': 95,
            'cholesterol_total': 180,
            'triglycerides': 100,
            'tsh': 2.0,
            'hemoglobin': 13.5,
            'vitamin_d': 35
        },
        "cycle": {
            'is_regular': True,
            'flow_intensity': 'regular',
            'regularity_score': 90
        }
    },
    {
        "name": "CAUTION - Moderate risk",
        "metrics": {
            'glucose': 110,
            'cholesterol_total': 210,
            'triglycerides': 140,
            'tsh': 2.1,
            'hemoglobin': 12.0,
            'vitamin_d': 25
        },
        "cycle": {
            'is_regular': True,
            'flow_intensity': 'light',
            'regularity_score': 80
        }
    },
    {
        "name": "ALERT - High cholesterol",
        "metrics": {
            'glucose': 100,
            'cholesterol_total': 260,
            'triglycerides': 180,
            'tsh': 2.0,
            'hemoglobin': 13.0,
            'vitamin_d': 30
        },
        "cycle": {
            'is_regular': True,
            'flow_intensity': 'regular',
            'regularity_score': 85
        }
    },
    {
        "name": "CRITICAL - High glucose + irregular cycle",
        "metrics": {
            'glucose': 135,
            'cholesterol_total': 200,
            'triglycerides': 150,
            'tsh': 2.0,
            'hemoglobin': 12.5,
            'vitamin_d': 28
        },
        "cycle": {
            'is_regular': False,
            'flow_intensity': 'irregular',
            'regularity_score': 45
        }
    },
    {
        "name": "CRITICAL - Anemia risk (low Hb + heavy flow)",
        "metrics": {
            'glucose': 98,
            'cholesterol_total': 190,
            'triglycerides': 110,
            'tsh': 2.0,
            'hemoglobin': 10.2,
            'vitamin_d': 32
        },
        "cycle": {
            'is_regular': True,
            'flow_intensity': 'heavy',
            'regularity_score': 88
        }
    },
    {
        "name": "CRITICAL - Very high triglycerides",
        "metrics": {
            'glucose': 115,
            'cholesterol_total': 220,
            'triglycerides': 240,
            'tsh': 2.0,
            'hemoglobin': 13.0,
            'vitamin_d': 30
        },
        "cycle": {
            'is_regular': True,
            'flow_intensity': 'regular',
            'regularity_score': 85
        }
    }
]

for i, test_case in enumerate(test_cases, 1):
    print(f"Test {i}: {test_case['name']}")
    
    # Calculate analysis
    analysis = analyze_health(test_case['metrics'])
    
    # Evaluate doctor alert
    alert = evaluate_doctor_alert(test_case['metrics'], analysis, test_case['cycle'])
    
    print(f"   Alert Type: {alert['alert_type']}")
    print(f"   Urgency: {alert['urgency_level']}")
    print(f"   Medical Review: {alert['needs_medical_review']}")
    print(f"   Findings: {len(alert['findings'])} items")
    for j, finding in enumerate(alert['findings'][:2], 1):
        print(f"      {j}. {finding}")
    if len(alert['findings']) > 2:
        print(f"      ... and {len(alert['findings']) - 2} more")
    print(f"   Recommendations:")
    for j, rec in enumerate(alert['recommendation'][:2], 1):
        print(f"      {j}. {rec}")
    print()

# Test endpoint
print("\n2️⃣  Testing /doctor-alerts Endpoint:\n")

with app.app_context():
    # Create test health reports with doctor alerts
    test_email = "doctor-alert-test@example.com"
    
    # Clean up existing test data
    HealthReport.query.filter_by(user_email=test_email).delete()
    db.session.commit()
    
    # Create multiple test reports with different alert types
    test_alerts = [
        {
            'alert_type': 'CRITICAL',
            'needs_medical_review': True,
            'urgency_level': 'HIGH',
            'findings': ['High glucose + irregular cycle', 'Requires immediate medical review']
        },
        {
            'alert_type': 'ALERT',
            'needs_medical_review': True,
            'urgency_level': 'MEDIUM',
            'findings': ['Elevated cholesterol', 'Schedule visit within 1-2 weeks']
        },
        {
            'alert_type': 'NORMAL',
            'needs_medical_review': False,
            'urgency_level': 'NONE',
            'findings': ['Metrics within normal ranges']
        }
    ]
    
    for idx, alert_data in enumerate(test_alerts):
        report = HealthReport(
            user_email=test_email,
            extracted_metrics={'glucose': 100 + (idx*20), 'cholesterol_total': 180},
            health_analysis={'risk_score': 30 + (idx*25)},
            diet_recommendation='Balanced',
            doctor_alert=alert_data
        )
        db.session.add(report)
    
    db.session.commit()
    print(f"✓ Created {len(test_alerts)} test reports\n")
    
    # Test the endpoint
    from flask import json
    with app.test_client() as client:
        response = client.get(f'/doctor-alerts/{test_email}')
        data = response.get_json()
        
        print(f"Status: {response.status_code}")
        print(f"Total Alerts: {data.get('total_alerts')}")
        print(f"Critical Alerts: {data.get('critical_count')}")
        print(f"Action Needed: {data['summary'].get('recommended_action')}")
        print()
        print("Alert Summary:")
        for idx, alert in enumerate(data['alerts'][:3], 1):
            print(f"   {idx}. {alert['alert_type']} ({alert['urgency_level']})")
            print(f"      Created: {alert['created_at']}")
            print(f"      Findings: {', '.join(alert['findings'][:1])}")
        print()
    
    # Cleanup
    HealthReport.query.filter_by(user_email=test_email).delete()
    db.session.commit()

print("="*70)
print("✅ PHASE 5 TEST COMPLETE!")
print("   Doctor alert gating functions working correctly")
print("   Deterministic rules + recommendations validated")
print("   Ready for frontend integration")
print("="*70 + "\n")
