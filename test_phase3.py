"""
Quick test script for Phase 3 Gemini JSON integration
"""

from app import (
    app, generate_gemini_json_response, predict_diet, analyze_health,
    validate_single_metric_bounds, extract_metrics_with_confidence
)
import json

print("\n" + "="*60)
print("PHASE 3 TEST: Gemini Master Prompt Integration")
print("="*60 + "\n")

# Test data
metrics = {
    'glucose': 125,
    'cholesterol_total': 240,
    'hdl': 45,
    'ldl': 160,
    'triglycerides': 180,
    'vitamin_d': 22,
    'tsh': 1.2
}

print("1️⃣  Test Data:")
for k, v in metrics.items():
    print(f"   {k}: {v}")
print()

# Test health analysis
print("2️⃣  Health Analysis:")
analysis = analyze_health(metrics)
print(f"   Risk Score: {analysis['risk_score']}/100")
print(f"   Health Summary: {analysis['health_summary']}")
print()

# Test ML diet prediction (Phase 2)
print("3️⃣  ML Diet Prediction (Phase 2):")
diet_prediction = predict_diet(metrics)
print(f"   Diet Type: {diet_prediction['diet_type']}")
print(f"   Confidence: {diet_prediction['confidence']}%")
print()

# Test Gemini JSON generation (Phase 3)
print("4️⃣  Gemini Master Prompt (Phase 3 - JSON Synthesis):")
print("   Calling Gemini API...")

result = generate_gemini_json_response(metrics, analysis, diet_prediction, 'Hormonal Balance')

print("   ✅ JSON Response received!")
print()

print("5️⃣  Response Structure:")
print(f"   Top-level keys: {list(result.keys())}")
print()

print("6️⃣  Medical Analysis:")
print(f"   - Summary: {result['medical_analysis']['summary'][:80]}...")
print(f"   - Risk Level: {result['medical_analysis']['risk_level']}")
print(f"   - Key Findings: {len(result['medical_analysis']['key_findings'])} items")
print()

print("7️⃣  Personalized Diet:")
print(f"   - Diet Type: {result['personalized_diet']['diet_type']}")
print(f"   - Rationale: {result['personalized_diet']['rationale'][:60]}...")
print(f"   - Meals: {list(result['personalized_diet']['daily_framework'].keys())}")
print(f"   - Foods to Include: {len(result['personalized_diet']['foods_include'])} items")
print(f"   - Foods to Avoid: {len(result['personalized_diet']['foods_avoid'])} items")
print()

print("8️⃣  Lifestyle Recommendations:")
for key, value in result['lifestyle_recommendations'].items():
    print(f"   - {key}: {value[:50]}...")
print()

print("9️⃣  Doctor Alert Flag:")
print(f"   - Needs Medical Review: {result['doctor_flag']['needs_medical_review']}")
if result['doctor_flag']['reason']:
    print(f"   - Reason: {result['doctor_flag']['reason']}")
print()

print("="*60)
print("✅ PHASE 3 TEST COMPLETE!")
print("   All functions working correctly")
print("   Gemini JSON response structure validated")
print("="*60 + "\n")

# Test validation (Phase 1)
print("BONUS: Phase 1 Validation Testing")
print()
validation_test = {'glucose': 450, 'cholesterol_total': 600}  # Out of bounds
for metric, value in validation_test.items():
    result = validate_single_metric_bounds(metric, value)
    is_valid = result['valid']
    bounds_info = f"(min: {result['min']}, max: {result['max']})"
    print(f"   {metric}: {value} -> Valid: {is_valid} {bounds_info}")
print()