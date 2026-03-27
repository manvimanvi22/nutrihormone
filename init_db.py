#!/usr/bin/env python
"""Initialize database with app context"""
import sys
sys.path.insert(0, '/Users/manvithag/Documents/IOMP')

from app import app, db

with app.app_context():
    db.create_all()
    print("✓ Database initialized successfully with all tables")
    print("✓ HealthReport model: Updated with extraction_confidence, validation_errors, ml_features_used, gemini_response_json, doctor_alert, related_cycle_id")
    print("✓ Cycle model: Updated with flow_intensity, period_duration, symptoms, mood, exercise_minutes, sleep_hours, metabolic_risk_at_upload")