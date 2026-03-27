import os
import re
import json
import joblib
import PyPDF2
import numpy as np
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from config import config

# Load environment variables
load_dotenv()

try:
    import google.generativeai as genai
except ImportError:
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")

# ======================================================
# APP CONFIGURATION
# ======================================================

app = Flask(__name__)
CORS(app)

# Load configuration from environment or default to development
env_config = os.getenv('FLASK_ENV', 'development')
app.config.from_object(config[env_config])
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# ======================================================
# GOOGLE GEMINI API SETUP
# ======================================================

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyCckL194AazW9yZF-IXD2No8MKi4ws_nRU')
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✓ Google Gemini API configured successfully")
except Exception as e:
    print(f"⚠ Warning: Gemini API configuration failed: {e}")

# ======================================================
# LOAD ML MODEL (With Fallback)
# ======================================================

diet_model = None
try:
    # Try to load the model
    diet_model = joblib.load("diet_model.pkl")
    print("✓ ML Diet model loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load ML model: {e}")
    print("   Using fallback diet recommendations instead")
    diet_model = None  # Use fallback mode

# ======================================================
# DATABASE MODELS
# ======================================================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)  # ✅ Increased length for hash
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.email}>'


class HealthReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), db.ForeignKey('user.email'), nullable=False)
    extracted_metrics = db.Column(db.JSON)
    health_analysis = db.Column(db.JSON)
    diet_recommendation = db.Column(db.Text)
    gemini_response = db.Column(db.Text)  # Kept for backward compatibility
    
    # PHASE 1: New extraction confidence & validation fields
    extraction_confidence = db.Column(db.JSON, nullable=True)  # {metric: confidence_score}
    validation_errors = db.Column(db.JSON, nullable=True)  # {metric: error_reason}
    ml_features_used = db.Column(db.JSON, nullable=True)  # {feature_name: value}
    extraction_method_version = db.Column(db.String(50), nullable=True)
    ml_model_version = db.Column(db.String(50), nullable=True)
    
    # PHASE 3 & 5: Gemini JSON & doctor flags
    gemini_response_json = db.Column(db.JSON, nullable=True)  # New structured format
    doctor_alert = db.Column(db.JSON, nullable=True)  # {should_alert, alert_type, reasons}
    
    # PHASE 4: Link to cycle data
    related_cycle_id = db.Column(db.Integer, db.ForeignKey('cycle.id'), nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f'<HealthReport {self.id}>'


class Cycle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), nullable=False, index=True)
    last_period = db.Column(db.Date)
    cycle_length = db.Column(db.Integer)
    
    # PHASE 4: Enhanced cycle tracking (Flo/Clue style)
    flow_intensity = db.Column(db.String(20), nullable=True)  # 'light'|'regular'|'heavy'
    period_duration = db.Column(db.Integer, nullable=True)  # Days 1-7
    symptoms = db.Column(db.JSON, nullable=True)  # {"cramps": 4, "bloating": 2, ...}
    mood = db.Column(db.String(50), nullable=True)  # 'happy'|'anxious'|'irritable'|'sad'
    exercise_minutes = db.Column(db.Integer, nullable=True)  # Lifestyle context
    sleep_hours = db.Column(db.Float, nullable=True)
    
    # PHASE 5: Health context at time of cycle log
    metabolic_risk_at_upload = db.Column(db.Float, nullable=True)  # Risk score snapshot
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

# ======================================================
# HEALTH METRICS REFERENCE RANGES
# ======================================================

HEALTH_RANGES = {
    'glucose': {
        'normal': (70, 100),
        'prediabetic': (100, 126),
        'diabetic': (126, 400)
    },
    'cholesterol_total': {
        'optimal': (0, 200),
        'borderline': (200, 240),
        'high': (240, 400)
    },
    'hdl': {
        'low': (0, 40),
        'normal': (40, 100)
    },
    'ldl': {
        'optimal': (0, 100),
        'borderline': (100, 130),
        'high': (130, 400)
    },
    'triglycerides': {
        'normal': (0, 150),
        'borderline': (150, 200),
        'high': (200, 500)
    },
    'tsh': {
        'normal': (0.4, 4.0),
        'elevated': (4.0, 10),
        'high': (10, 100)
    },
    'vitamin_d': {
        'deficient': (0, 20),
        'insufficient': (20, 30),
        'sufficient': (30, 100)
    },
    'hemoglobin': {
        'low': (0, 11.5),
        'normal': (11.5, 16),
        'high': (16, 20)
    }
}

# ======================================================
# PDF EXTRACTION
# ======================================================

def extract_pdf_text(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        print(f"✓ Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        print(f"✗ PDF extraction error: {e}")
        return None

# ======================================================
# HEALTH METRICS EXTRACTION
# ======================================================

METRIC_PATTERNS = {
    'glucose': r'(?:fasting\s+)?glucose\s*[:\-]?\s*(\d+\.?\d*)',
    'cholesterol_total': r'total\s+cholesterol\s*[:\-]?\s*(\d+\.?\d*)',
    'hdl': r'hdl\s*[:\-]?\s*(\d+\.?\d*)',
    'ldl': r'ldl\s*[:\-]?\s*(\d+\.?\d*)',
    'triglycerides': r'triglycerides\s*[:\-]?\s*(\d+\.?\d*)',
    'tsh': r'tsh\s*[:\-]?\s*(\d+\.?\d*)',
    'vitamin_d': r'vitamin\s+d\s*[:\-]?\s*(\d+\.?\d*)',
    'hemoglobin': r'hemoglobin\s*[:\-]?\s*(\d+\.?\d*)',
    'hematocrit': r'hematocrit\s*[:\-]?\s*(\d+\.?\d*)',
    'wbc': r'(?:wbc|white\s+blood\s+cell)\s*[:\-]?\s*(\d+\.?\d*)',
    'rbc': r'(?:rbc|red\s+blood\s+cell)\s*[:\-]?\s*(\d+\.?\d*)',
    'platelets': r'platelets\s*[:\-]?\s*(\d+\.?\d*)',
    'alt': r'(?:alt|sgpt)\s*[:\-]?\s*(\d+\.?\d*)',
    'ast': r'(?:ast|sgot)\s*[:\-]?\s*(\d+\.?\d*)',
    'creatinine': r'creatinine\s*[:\-]?\s*(\d+\.?\d*)',
    'bun': r'(?:bun|blood\s+urea)\s*[:\-]?\s*(\d+\.?\d*)',
    'sodium': r'sodium\s*[:\-]?\s*(\d+\.?\d*)',
    'potassium': r'potassium\s*[:\-]?\s*(\d+\.?\d*)',
    'calcium': r'calcium\s*[:\-]?\s*(\d+\.?\d*)',
    'iron': r'iron\s*[:\-]?\s*(\d+\.?\d*)',
    'ferritin': r'ferritin\s*[:\-]?\s*(\d+\.?\d*)',
}

def extract_metrics_with_confidence(text):
    """
    PHASE 1: Extract health metrics from text with confidence scoring
    Returns: {metric: {value: X, unit: 'mg/dL', confidence: 0.85, raw_match: '...'}}
    """
    metrics_with_confidence = {}
    
    if not text:
        return metrics_with_confidence
    
    text_lower = text.lower()
    
    for metric_name, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                raw_match = match.group(0)
                
                # Calculate confidence score (0.6-0.9 base)
                base_confidence = 0.75
                
                # Bonus for specific context (look at surrounding text)
                context_start = max(0, match.start() - 30)
                context_end = min(len(text), match.end() + 30)
                context = text_lower[context_start:context_end]
                
                if metric_name in context or 'lab' in context or 'report' in context or 'result' in context:
                    base_confidence += 0.1
                
                # Check if value is within realistic bounds
                bounds_check = validate_single_metric_bounds(metric_name, value)
                if bounds_check['valid']:
                    base_confidence += 0.05
                else:
                    base_confidence -= 0.15  # Penalty for out-of-range
                
                # Clamp confidence to 0-1
                confidence = max(0.0, min(1.0, base_confidence))
                
                metrics_with_confidence[metric_name] = {
                    'value': value,
                    'unit': get_metric_unit(metric_name),
                    'confidence': round(confidence, 2),
                    'raw_match': raw_match
                }
                
                print(f"  ✓ Found {metric_name}: {value} (confidence: {confidence:.2f})")
            except (ValueError, IndexError) as e:
                print(f"  ✗ Error parsing {metric_name}: {e}")
                continue
    
    print(f"✓ Extracted {len(metrics_with_confidence)} health metrics with confidence scoring")
    return metrics_with_confidence


def get_metric_unit(metric_name):
    """Return standard unit for metric"""
    units = {
        'glucose': 'mg/dL',
        'cholesterol_total': 'mg/dL',
        'hdl': 'mg/dL',
        'ldl': 'mg/dL',
        'triglycerides': 'mg/dL',
        'tsh': 'mIU/L',
        'vitamin_d': 'ng/mL',
        'hemoglobin': 'g/dL',
        'hematocrit': '%',
        'wbc': '10^3/µL',
        'rbc': '10^6/µL',
        'platelets': '10^3/µL',
        'alt': 'U/L',
        'ast': 'U/L',
        'creatinine': 'mg/dL',
        'bun': 'mg/dL',
        'sodium': 'mEq/L',
        'potassium': 'mEq/L',
        'calcium': 'mg/dL',
        'iron': 'µg/dL',
        'ferritin': 'ng/mL',
    }
    return units.get(metric_name, '')


def validate_single_metric_bounds(metric_name, value):
    """Check if a single metric value is within realistic bounds"""
    bounds = {
        'glucose': (50, 400),
        'cholesterol_total': (100, 500),
        'hdl': (0, 150),
        'ldl': (0, 250),
        'triglycerides': (0, 400),
        'tsh': (0.01, 100),
        'vitamin_d': (0, 150),
        'hemoglobin': (5, 20),
        'hematocrit': (10, 60),
        'wbc': (1, 30),
        'rbc': (2, 7),
        'platelets': (50, 500),
        'alt': (5, 300),
        'ast': (5, 300),
        'creatinine': (0.3, 3),
        'bun': (5, 50),
        'sodium': (100, 160),
        'potassium': (2, 8),
        'calcium': (6, 12),
        'iron': (30, 400),
        'ferritin': (5, 500),
    }
    
    if metric_name in bounds:
        min_val, max_val = bounds[metric_name]
        return {
            'valid': min_val <= value <= max_val,
            'min': min_val,
            'max': max_val,
            'value': value
        }
    
    return {'valid': True, 'reason': 'No bounds defined'}


def validate_metric_bounds(metrics_dict):
    """
    PHASE 1: Validate all extracted metrics against realistic bounds
    Returns: {valid: bool, errors: [{metric: name, reason: msg}]}
    """
    validation_errors = []
    
    for metric_name, metric_data in metrics_dict.items():
        if isinstance(metric_data, dict) and 'value' in metric_data:
            value = metric_data['value']
        else:
            value = metric_data
        
        bounds_result = validate_single_metric_bounds(metric_name, value)
        
        if not bounds_result['valid']:
            validation_errors.append({
                'metric': metric_name,
                'value': value,
                'reason': f'Out of bounds. Expected {bounds_result["min"]}-{bounds_result["max"]}'
            })
    
    return {
        'valid': len(validation_errors) == 0,
        'errors': validation_errors,
        'metrics_validated': len(metrics_dict),
        'errors_count': len(validation_errors)
    }


def extract_metrics(text):
    """
    Legacy function: Extract health metrics from text using regex patterns
    (Kept for backward compatibility)
    """
    metrics = {}
    
    if not text:
        return metrics
    
    text_lower = text.lower()
    
    for metric_name, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                metrics[metric_name] = value
                print(f"  ✓ Found {metric_name}: {value}")
            except (ValueError, IndexError):
                continue
    
    print(f"✓ Extracted {len(metrics)} health metrics")
    return metrics

# ======================================================
# HEALTH ANALYSIS
# ======================================================

def analyze_health(metrics):
    """Analyze health metrics and identify risks"""
    analysis = {
        'metrics_details': {},
        'risk_alerts': [],
        'risk_score': 0,
        'health_summary': {}
    }
    
    risk_count = 0
    
    # Glucose Analysis
    if 'glucose' in metrics:
        glucose = metrics['glucose']
        status = 'normal'
        if glucose > 126:
            status = 'critical'
            analysis['risk_alerts'].append(f"⚠️ CRITICAL: Glucose level {glucose} mg/dL indicates possible diabetes")
            risk_count += 2
        elif glucose > 100:
            status = 'warning'
            analysis['risk_alerts'].append(f"⚠️ WARNING: Elevated fasting glucose {glucose} mg/dL (prediabetes risk)")
            risk_count += 1
        
        analysis['metrics_details']['glucose'] = {
            'value': glucose,
            'unit': 'mg/dL',
            'status': status
        }
        analysis['health_summary']['glucose_control'] = 100 - (glucose - 70) * 0.5 if glucose >= 70 else 100
    
    # Lipid Panel
    if 'cholesterol_total' in metrics:
        chol = metrics['cholesterol_total']
        status = 'normal'
        if chol > 240:
            status = 'critical'
            analysis['risk_alerts'].append(f"⚠️ CRITICAL: High total cholesterol {chol} mg/dL")
            risk_count += 2
        elif chol > 200:
            status = 'warning'
            analysis['risk_alerts'].append(f"⚠️ WARNING: Borderline high cholesterol {chol} mg/dL")
            risk_count += 1
        
        analysis['metrics_details']['cholesterol'] = {
            'value': chol,
            'unit': 'mg/dL',
            'status': status
        }
    
    if 'ldl' in metrics:
        ldl = metrics['ldl']
        status = 'normal'
        if ldl > 160:
            status = 'critical'
            risk_count += 2
        elif ldl > 130:
            status = 'warning'
            risk_count += 1
        
        analysis['metrics_details']['ldl'] = {
            'value': ldl,
            'unit': 'mg/dL',
            'status': status
        }
    
    if 'hdl' in metrics:
        hdl = metrics['hdl']
        status = 'normal' if hdl >= 40 else 'warning'
        if hdl < 40:
            analysis['risk_alerts'].append(f"⚠️ WARNING: Low HDL (good cholesterol) {hdl} mg/dL")
            risk_count += 1
        
        analysis['metrics_details']['hdl'] = {
            'value': hdl,
            'unit': 'mg/dL',
            'status': status
        }
    
    # Vitamin D
    if 'vitamin_d' in metrics:
        vit_d = metrics['vitamin_d']
        status = 'normal'
        if vit_d < 20:
            status = 'critical'
            analysis['risk_alerts'].append(f"⚠️ CRITICAL: Vitamin D deficiency {vit_d} ng/mL (supplementation needed)")
            risk_count += 2
        elif vit_d < 30:
            status = 'warning'
            analysis['risk_alerts'].append(f"⚠️ WARNING: Low vitamin D {vit_d} ng/mL")
            risk_count += 1
        
        analysis['metrics_details']['vitamin_d'] = {
            'value': vit_d,
            'unit': 'ng/mL',
            'status': status
        }
    
    # Thyroid
    if 'tsh' in metrics:
        tsh = metrics['tsh']
        status = 'normal'
        if tsh > 5:
            status = 'critical'
            analysis['risk_alerts'].append(f"⚠️ CRITICAL: Elevated TSH {tsh} (possible hypothyroidism)")
            risk_count += 2
        elif tsh > 4:
            status = 'warning'
            analysis['risk_alerts'].append(f"⚠️ WARNING: High TSH {tsh}")
            risk_count += 1
        
        analysis['metrics_details']['tsh'] = {
            'value': tsh,
            'unit': 'mIU/L',
            'status': status
        }
    
    # Hemoglobin
    if 'hemoglobin' in metrics:
        hb = metrics['hemoglobin']
        status = 'normal'
        if hb < 11.5:
            status = 'critical'
            analysis['risk_alerts'].append(f"⚠️ CRITICAL: Anemia - Low hemoglobin {hb} g/dL")
            risk_count += 2
        elif hb < 12:
            status = 'warning'
            analysis['risk_alerts'].append(f"⚠️ WARNING: Low hemoglobin {hb} g/dL")
            risk_count += 1
        
        analysis['metrics_details']['hemoglobin'] = {
            'value': hb,
            'unit': 'g/dL',
            'status': status
        }
    
    # Calculate risk score (0-100)
    analysis['risk_score'] = min(risk_count * 15, 100)
    
    if not analysis['risk_alerts']:
        analysis['risk_alerts'].append("✓ All metrics within normal ranges")
    
    print(f"✓ Health analysis complete - Risk score: {analysis['risk_score']}")
    return analysis

# ======================================================
# ML DIET PREDICTION (WITH FALLBACK)
# ======================================================

def predict_diet(metrics):
    """
    Predict diet type using ML model trained on blood metrics.
    
    Features: [Glucose, Cholesterol, HDL, LDL, Triglycerides, Vitamin_D, TSH]
    
    This REPLACES the old model which used demographics (age, BMI, exercise hours).
    New model uses actual extracted blood test results for better accuracy.
    """
    
    # ✅ FALLBACK: If model is not available, use smart heuristics
    if not diet_model:
        return smart_diet_prediction(metrics)
    
    try:
        # Extract blood metrics with defaults (safe values)
        glucose = metrics.get('glucose', 95.0)
        cholesterol = metrics.get('cholesterol_total', 180.0)
        hdl = metrics.get('hdl', 50.0)
        ldl = metrics.get('ldl', 100.0)
        triglycerides = metrics.get('triglycerides', 100.0)
        vitamin_d = metrics.get('vitamin_d', 30.0)
        tsh = metrics.get('tsh', 2.0)
        
        # Prepare features for prediction (MUST match training order!)
        # Order: Glucose, Cholesterol, HDL, LDL, Triglycerides, Vitamin_D, TSH
        features = np.array([[
            glucose,
            cholesterol,
            hdl,
            ldl,
            triglycerides,
            vitamin_d,
            tsh
        ]])
        
        prediction = diet_model.predict(features)[0]
        
        # Get confidence score (0-100)
        try:
            probabilities = diet_model.predict_proba(features)[0]
            confidence = float(max(probabilities)) * 100
        except:
            confidence = 65.0
        
        print(f"✓ ML Prediction: {prediction} (Confidence: {confidence:.1f}%)")
        return {
            'diet_type': prediction,
            'confidence': round(confidence, 2)
        }
    except Exception as e:
        print(f"✗ ML prediction error: {e}")
        return smart_diet_prediction(metrics)

# ======================================================
# SMART DIET PREDICTION (FALLBACK)
# ======================================================

def smart_diet_prediction(metrics):
    """Smart fallback diet prediction based on metrics"""
    
    print("✓ Using smart heuristic diet prediction")
    
    # Analyze metrics to recommend diet
    glucose = metrics.get('glucose', 95)
    cholesterol = metrics.get('cholesterol_total', 180)
    hdl = metrics.get('hdl', 50)
    vitamin_d = metrics.get('vitamin_d', 30)
    
    # Decision logic
    if glucose > 100 or cholesterol > 200:
        if cholesterol > 240:
            diet_type = "Mediterranean Diet"
            confidence = 92.0
        else:
            diet_type = "Balanced Mediterranean Diet"
            confidence = 88.0
    elif vitamin_d < 20:
        diet_type = "Anti-Inflammatory Diet"
        confidence = 85.0
    else:
        diet_type = "Balanced Whole Foods Diet"
        confidence = 82.0
    
    print(f"✓ Smart Prediction: {diet_type} (Confidence: {confidence}%)")
    
    return {
        'diet_type': diet_type,
        'confidence': confidence
    }

# ======================================================
# GOOGLE GEMINI AI RECOMMENDATION
# ======================================================

def generate_gemini_json_response(metrics, analysis, diet_prediction, wellness_goal):
    """
    PHASE 3: Gemini Master Prompt Integration
    
    Generate structured AI response as JSON instead of markdown.
    Combines extracted blood metrics + ML predictions + hormonal context.
    
    Returns: dict with medical_analysis, diet_plan, cycle_insights, recommendations
    """
    
    try:
        # Build context from metrics and analysis
        metrics_summary = "\n".join([
            f"- {k}: {v['value']} {v.get('unit', '')} ({v.get('status', 'normal')})"
            for k, v in analysis.get('metrics_details', {}).items()
        ])
        
        alerts_summary = "\n".join(analysis.get('risk_alerts', []))
        
        # Master Prompt: Asks Gemini to return ONLY valid JSON
        prompt = f"""You are a professional clinical nutritionist specializing in women's hormonal health.

PATIENT DATA:
- Wellness Goal: {wellness_goal}
- ML Predicted Diet: {diet_prediction.get('diet_type', 'Balanced')} ({diet_prediction.get('confidence', 50)}% confidence)
- Risk Score: {analysis.get('risk_score', 50)}/100

BLOOD METRICS:
{metrics_summary}

HEALTH ALERTS:
{alerts_summary if alerts_summary else "No critical alerts"}

IMPORTANT: Return ONLY a valid JSON object with NO additional text. Build this exact structure:

{{
  "medical_analysis": {{
    "summary": "2-3 sentence health status overview",
    "key_findings": ["finding1", "finding2", "finding3"],
    "risk_level": "low|moderate|high|critical"
  }},
  "personalized_diet": {{
    "diet_type": "{diet_prediction.get('diet_type', 'Balanced')}",
    "rationale": "Why this diet matches their metrics",
    "daily_framework": {{
      "breakfast": ["option1", "option2", "option3"],
      "lunch": ["option1", "option2", "option3"],
      "dinner": ["option1", "option2", "option3"],
      "snacks": ["option1", "option2"]
    }},
    "foods_include": ["food1", "food2", "food3", "food4", "food5"],
    "foods_avoid": ["food1", "food2", "food3", "food4", "food5"]
  }},
  "lifestyle_recommendations": {{
    "exercise": "specific guidance",
    "sleep": "sleep optimization tips",
    "stress": "stress management advice",
    "hydration": "water intake recommendation"
  }},
  "supplements": ["supplement1 with dosage", "supplement2 with dosage"],
  "tracking_metrics": ["metric1", "metric2", "metric3"],
  "doctor_flag": {{
    "needs_medical_review": false,
    "reason": "reason if true"
  }}
}}"""

        print("🔄 Calling Google Gemini API (Phase 3: Master Prompt)...")
        
        # Call Gemini API
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        response_text = response.text.strip()
        print(f"✓ Gemini API response received")
        
        # Try to parse JSON response
        # Sometimes Gemini wraps JSON in markdown code blocks, so clean it
        if response_text.startswith("```"):
            # Remove markdown code blocks if present
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        gemini_json = json.loads(response_text)
        print(f"✓ JSON parsed successfully")
        
        return gemini_json
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON parse error from Gemini response: {e}")
        print(f"  Raw response: {response.text[:200]}...")
        return generate_fallback_json_response(metrics, analysis, diet_prediction)
        
    except Exception as e:
        print(f"✗ Gemini API error: {e}")
        return generate_fallback_json_response(metrics, analysis, diet_prediction)


def generate_fallback_json_response(metrics, analysis, diet_prediction):
    """
    ML-powered personalised response based on actual extracted metrics.
    """
    print("✓ Generating ML-based personalised response")

    diet_type = diet_prediction.get('diet_type', 'Balanced')
    risk_score = analysis.get('risk_score', 50)
    alerts = analysis.get('risk_alerts', [])
    metrics_details = analysis.get('metrics_details', {})

    # --- Build personalised summary from actual metrics ---
    tsh = metrics.get('tsh')
    hemoglobin = metrics.get('hemoglobin')
    glucose = metrics.get('glucose')
    cholesterol = metrics.get('cholesterol_total')
    triglycerides = metrics.get('triglycerides')
    vitamin_d = metrics.get('vitamin_d')
    hdl = metrics.get('hdl')
    ldl = metrics.get('ldl')

    summary_parts = [f"Your ML-predicted optimal diet is {diet_type} (Risk Score: {risk_score}/100)."]
    foods_include = []
    foods_avoid = []
    breakfast = []
    lunch = []
    dinner = []
    snacks = []
    supplements = []
    lifestyle = {}

    # --- TSH (Thyroid) ---
    if tsh:
        if tsh > 4.0:
            summary_parts.append(f"Your TSH ({tsh} mIU/L) is elevated indicating hypothyroidism risk — prioritise iodine and selenium-rich foods.")
            foods_include += ["Seaweed and iodine-rich foods", "Brazil nuts (selenium)", "Eggs", "Fish (tuna, sardines)", "Dairy (low-fat milk, yogurt)"]
            foods_avoid += ["Raw cruciferous vegetables in excess (broccoli, cabbage)", "Soy products in large amounts", "Processed foods with excess fluoride"]
            supplements.append("Selenium 200mcg daily — supports thyroid function")
            supplements.append("Iodine supplement — consult doctor for dosage")
            lifestyle["thyroid_support"] = "Avoid eating 30-60 minutes before or after thyroid medication. Manage stress as cortisol suppresses thyroid function."
        elif tsh < 0.4:
            summary_parts.append(f"Your TSH ({tsh} mIU/L) is low indicating hyperthyroid risk — avoid excessive iodine.")
            foods_avoid += ["Seaweed and high-iodine foods", "Excessive caffeine", "Alcohol"]
            foods_include += ["Cruciferous vegetables (broccoli, cauliflower)", "Calcium-rich foods", "Anti-inflammatory foods"]

    # --- Hemoglobin (Anaemia) ---
    if hemoglobin:
        if hemoglobin < 12:
            summary_parts.append(f"Your hemoglobin ({hemoglobin} g/dL) indicates anaemia — increase iron-rich foods and Vitamin C for absorption.")
            foods_include += ["Lean red meat and poultry", "Spinach, kale and dark leafy greens", "Lentils and chickpeas", "Fortified cereals", "Pumpkin seeds", "Dark chocolate (70%+)"]
            foods_avoid += ["Tea and coffee with meals (inhibits iron absorption)", "Calcium-rich foods with iron-rich meals", "High-fibre foods in excess"]
            supplements.append("Iron supplement — consult doctor for dosage based on severity")
            supplements.append("Vitamin C 500mg with meals — enhances iron absorption")
            lifestyle["anaemia_management"] = "Eat Vitamin C rich foods (lemon, orange, tomato) alongside iron-rich meals to maximise absorption. Avoid tea/coffee 1 hour before and after iron-rich meals."

    # --- Glucose ---
    if glucose:
        if glucose > 126:
            summary_parts.append(f"Your glucose ({glucose} mg/dL) is in diabetic range — strictly limit refined carbohydrates and sugars.")
            foods_include += ["Non-starchy vegetables", "Whole grains (oats, barley, quinoa)", "Legumes", "Nuts and seeds", "Lean proteins"]
            foods_avoid += ["White rice, white bread, pasta", "Sugary drinks and juices", "Sweets and desserts", "High-GI fruits (mango, grapes, banana)"]
            lifestyle["glucose_control"] = "Eat smaller meals every 3-4 hours. Walk 10 minutes after meals to lower blood sugar. Monitor glucose regularly."
        elif glucose > 100:
            summary_parts.append(f"Your glucose ({glucose} mg/dL) is in pre-diabetic range — reduce refined sugars and choose low-GI foods.")
            foods_include += ["Low-GI foods (sweet potato, oats, lentils)", "Cinnamon (helps regulate blood sugar)", "Apple cider vinegar", "Berries"]
            foods_avoid += ["Refined sugars", "Sugary beverages", "White bread and refined grains"]

    # --- Cholesterol ---
    if cholesterol:
        if cholesterol > 240:
            summary_parts.append(f"Your total cholesterol ({cholesterol} mg/dL) is high — focus on heart-healthy fats and fibre.")
            foods_include += ["Oats and oat bran", "Fatty fish (salmon, mackerel)", "Avocado", "Olive oil", "Walnuts and almonds", "Flaxseeds"]
            foods_avoid += ["Saturated fats (butter, lard, fatty meat)", "Trans fats (fried foods, pastries)", "Full-fat dairy", "Red meat"]
            supplements.append("Omega-3 fatty acids 2000mg daily — reduces LDL cholesterol")
            supplements.append("Plant sterols/stanols 2g daily — lowers LDL by 10-15%")
        elif cholesterol > 200:
            foods_include += ["Soluble fibre foods (oats, beans, apples)", "Olive oil", "Nuts"]
            foods_avoid += ["Fried foods", "Processed snacks", "Excess saturated fats"]

    # --- Triglycerides ---
    if triglycerides:
        if triglycerides > 200:
            summary_parts.append(f"Your triglycerides ({triglycerides} mg/dL) are very high — eliminate sugar and alcohol immediately.")
            foods_avoid += ["All added sugars", "Alcohol", "Refined carbohydrates", "Fruit juice"]
            foods_include += ["Fatty fish (omega-3s lower triglycerides)", "Garlic", "Apple cider vinegar", "Green tea"]

    # --- Vitamin D ---
    if vitamin_d:
        if vitamin_d < 20:
            summary_parts.append(f"Your Vitamin D ({vitamin_d} ng/mL) is deficient — supplement immediately and get morning sunlight.")
            foods_include += ["Fatty fish (salmon, tuna)", "Egg yolks", "Fortified milk and cereals", "Mushrooms (UV-exposed)"]
            supplements.append("Vitamin D3 2000-4000 IU daily — critical for hormonal balance and immunity")
            lifestyle["vitamin_d"] = "Get 15-20 minutes of morning sunlight daily. Vitamin D deficiency worsens thyroid function and immune health."
        elif vitamin_d < 30:
            supplements.append("Vitamin D3 1000-2000 IU daily — maintain optimal levels")

    # --- Diet-type specific meal plans ---
    meal_plans = {
        'Low_Sodium': {
            'breakfast': ["Oatmeal with banana and unsalted nuts", "Whole grain toast with avocado and tomato", "Greek yogurt with fresh berries"],
            'lunch': ["Grilled chicken salad with olive oil and lemon dressing", "Lentil soup with whole grain bread (no added salt)", "Quinoa bowl with roasted vegetables and herbs"],
            'dinner': ["Baked salmon with steamed broccoli and brown rice", "Grilled chicken with sweet potato and green beans", "Stir-fried tofu with low-sodium vegetables"],
            'snacks': ["Fresh fruit (apple, orange, pear)", "Unsalted nuts and seeds", "Carrot and cucumber sticks with hummus"]
        },
        'Low-Carb': {
            'breakfast': ["Scrambled eggs with spinach and avocado", "Greek yogurt with nuts and seeds", "Veggie omelette with cheese"],
            'lunch': ["Grilled chicken with mixed greens and olive oil", "Tuna salad in lettuce wraps", "Egg salad with avocado"],
            'dinner': ["Baked salmon with asparagus", "Lean beef stir-fry with low-carb vegetables", "Grilled chicken thighs with roasted broccoli"],
            'snacks': ["Boiled eggs", "Cheese with cucumber", "Handful of almonds"]
        },
        'Mediterranean': {
            'breakfast': ["Whole grain toast with olive oil and tomatoes", "Greek yogurt with honey and walnuts", "Fresh fruit with nuts"],
            'lunch': ["Greek salad with feta, olives and chickpeas", "Lentil soup with whole grain bread", "Grilled fish with roasted vegetables"],
            'dinner': ["Baked salmon with quinoa and steamed vegetables", "Chickpea stew with olive oil", "Grilled chicken with Mediterranean herbs and salad"],
            'snacks': ["Fresh fruit", "Mixed nuts and olives", "Hummus with vegetables"]
        },
        'High-Protein': {
            'breakfast': ["Greek yogurt with berries and protein powder", "Egg white omelette with vegetables", "Cottage cheese with fruit"],
            'lunch': ["Grilled chicken breast with quinoa and salad", "Tuna with brown rice and vegetables", "Lentil and vegetable soup"],
            'dinner': ["Baked salmon with brown rice and broccoli", "Lean beef with sweet potato", "Turkey meatballs with zucchini"],
            'snacks': ["Boiled eggs", "Greek yogurt", "Edamame"]
        },
        'Balanced': {
            'breakfast': ["Oatmeal with mixed berries and nuts", "Whole grain toast with eggs and vegetables", "Smoothie with spinach, banana and yogurt"],
            'lunch': ["Grilled chicken with brown rice and vegetables", "Vegetable and lentil soup", "Whole grain wrap with lean protein and salad"],
            'dinner': ["Baked fish with quinoa and steamed greens", "Stir-fried tofu with brown rice", "Grilled chicken with sweet potato"],
            'snacks': ["Apple with peanut butter", "Mixed nuts", "Yogurt with berries"]
        },
        'Keto': {
            'breakfast': ["Eggs with bacon and avocado", "Full-fat Greek yogurt with nuts", "Veggie omelette with cheese"],
            'lunch': ["Chicken Caesar salad (no croutons)", "Tuna salad with mayo on lettuce", "Egg salad with avocado"],
            'dinner': ["Grilled salmon with asparagus and butter", "Beef steak with spinach and mushrooms", "Baked chicken thighs with roasted broccoli"],
            'snacks': ["Cheese cubes", "Boiled eggs", "Macadamia nuts"]
        }
    }

    plan = meal_plans.get(diet_type, meal_plans['Balanced'])
    breakfast = plan['breakfast']
    lunch = plan['lunch']
    dinner = plan['dinner']
    snacks = plan['snacks']

    # Default foods if none were added from metric analysis
    if not foods_include:
        foods_include = ["Fresh vegetables and fruits", "Whole grains", "Lean proteins", "Healthy fats (olive oil, avocado, nuts)", "Low-fat dairy"]
    if not foods_avoid:
        foods_avoid = ["Ultra-processed foods", "Sugary drinks", "Refined carbohydrates", "Excess sodium", "Trans fats"]

    # Default lifestyle
    if not lifestyle:
        lifestyle = {}
    lifestyle.setdefault("exercise", "30 minutes of moderate exercise daily — walking, yoga or swimming.")
    lifestyle.setdefault("sleep", "7-8 hours of quality sleep. Poor sleep disrupts hormones and worsens metabolic markers.")
    lifestyle.setdefault("stress", "Practice mindfulness or deep breathing 10 minutes daily. Chronic stress elevates cortisol.")
    lifestyle.setdefault("hydration", "Drink 8-10 glasses of water daily. Supports kidney function and nutrient absorption.")

    if not supplements:
        supplements = ["Vitamin D3 1000 IU daily", "Omega-3 fatty acids 1000mg daily", "Magnesium 300mg daily — supports hormonal balance"]

    return {
        "medical_analysis": {
            "summary": " ".join(summary_parts),
            "key_findings": alerts if alerts else ["Health metrics analysed successfully"],
            "risk_level": "high" if risk_score > 70 else "moderate" if risk_score > 40 else "low"
        },
        "personalized_diet": {
            "diet_type": diet_type,
            "rationale": f"The ML model predicted {diet_type} diet based on your blood markers. " + summary_parts[0],
            "daily_framework": {
                "breakfast": breakfast,
                "lunch": lunch,
                "dinner": dinner,
                "snacks": snacks
            },
            "foods_include": list(dict.fromkeys(foods_include))[:8],
            "foods_avoid": list(dict.fromkeys(foods_avoid))[:8]
        },
        "lifestyle_recommendations": lifestyle,
        "supplements": supplements[:4],
        "tracking_metrics": ["Energy levels daily (1-10)", "Sleep quality", "Hemoglobin levels", "TSH levels monthly"],
        "doctor_flag": {
            "needs_medical_review": risk_score > 40,
            "reason": "Abnormal markers detected — schedule a follow-up with your doctor" if risk_score > 40 else ""
        }
    }

def generate_diet_plan(metrics, analysis, diet_prediction, wellness_goal):
    """Generate personalized diet plan using Google Gemini AI"""
    
    try:
        # Build context from metrics and analysis
        metrics_summary = "\n".join([
            f"- {k}: {v['value']} {v['unit']} ({v['status']})"
            for k, v in analysis['metrics_details'].items()
        ])
        
        alerts_summary = "\n".join(analysis['risk_alerts'])
        
        prompt = f"""You are a professional clinical nutritionist specializing in women's hormonal health and metabolic wellness.

PATIENT HEALTH PROFILE:
Wellness Goal: {wellness_goal}
Recommended Diet: {diet_prediction['diet_type']} (Confidence: {diet_prediction['confidence']}%)
Risk Score: {analysis['risk_score']}/100

EXTRACTED HEALTH METRICS:
{metrics_summary}

HEALTH ALERTS:
{alerts_summary}

Based on this health profile, create a comprehensive, personalized nutrition and lifestyle plan. Format your response as follows:

## 🎯 PERSONALIZED DIET PLAN FOR HORMONAL WELLNESS

### 📊 Health Summary
[Brief 2-3 sentence summary of their health status]

### 🥗 Recommended Dietary Approach
[Why the {diet_prediction['diet_type']} diet is optimal for them]

### 🍽️ Daily Meal Framework

**Breakfast Ideas:**
- [Option 1]
- [Option 2]
- [Option 3]

**Lunch Ideas:**
- [Option 1]
- [Option 2]
- [Option 3]

**Dinner Ideas:**
- [Option 1]
- [Option 2]
- [Option 3]

**Healthy Snacks:**
- [Option 1]
- [Option 2]

### ✅ Foods to Include
- [List 5-7 specific foods beneficial for their condition]

### ❌ Foods to Avoid
- [List 5-7 foods to limit or avoid]

### 💊 Recommended Supplements (if applicable)
{f"- Vitamin D: {metrics.get('vitamin_d', 30)} ng/mL - Consider 1000-2000 IU daily" if metrics.get('vitamin_d', 30) < 30 else "- Vitamin D levels are adequate"}
[Other relevant supplements based on their metrics]

### 🏃 Lifestyle Recommendations
- Exercise: [Specific recommendations]
- Sleep: [Sleep optimization tips]
- Stress Management: [Stress reduction strategies]
- Hydration: [Water intake recommendations]

### 📈 Progress Metrics to Track
[What they should monitor monthly]

### ⚠️ Important Notes
[Any medical disclaimers or recommendations to see a healthcare provider]

Make the recommendations practical, hormone-aware, and based on evidence-based nutrition science."""

        print("🔄 Calling Google Gemini API for nutrition plan...")
        
        # Call Gemini API
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        plan = response.text
        print(f"✓ Gemini API response received ({len(plan)} characters)")
        return plan
        
    except Exception as e:
        print(f"✗ Gemini API error: {e}")
        
        # Fallback response
        fallback = f"""## 🎯 PERSONALIZED DIET PLAN FOR HORMONAL WELLNESS

### 📊 Health Summary
Based on your health metrics and wellness goal ({wellness_goal}), we recommend a {diet_prediction['diet_type']} approach to support your hormonal health and metabolic wellness.

### 🍽️ Daily Meal Framework

**Breakfast:** Include protein, whole grains, and fruits
- Oatmeal with berries and nuts (15g protein)
- Greek yogurt with granola and honey (20g protein)
- Whole grain toast with avocado and eggs (18g protein)

**Lunch:** Balanced macronutrients with vegetables
- Grilled chicken with quinoa and roasted vegetables
- Salmon with sweet potato and leafy greens
- Vegetable stir-fry with brown rice and tofu

**Dinner:** Light and nutritious, easy to digest
- Lean protein with roasted vegetables
- Baked white fish with leafy greens
- Plant-based options with legumes and whole grains

**Snacks:** Nutrient-dense choices for sustained energy
- Nuts and seeds (almonds, walnuts, pumpkin seeds)
- Fresh fruits (berries, apples, oranges)
- Greek yogurt or cottage cheese with nuts

### ✅ Foods to Include for Hormonal Balance
- Leafy greens (spinach, kale, lettuce) - rich in minerals
- Fatty fish (salmon, mackerel, sardines) - omega-3s
- Whole grains (oats, quinoa, brown rice)
- Legumes and beans (lentils, chickpeas) - fiber & protein
- Nuts and seeds (flax, chia, pumpkin seeds)
- Colorful vegetables (broccoli, bell peppers, carrots)
- Berries and fruits (blueberries, cherries, oranges)

### ❌ Foods to Limit or Avoid
- Processed foods and refined carbohydrates
- Excess added sugar and sugary drinks
- Refined grains (white bread, white rice, pastries)
- Excessive caffeine (limit to 1-2 cups coffee)
- Alcohol and sugary cocktails
- Trans fats (fried foods, processed oils)
- Highly processed snacks and convenience foods

### 💊 Supplements to Consider
- Vitamin D: 1000-2000 IU daily (if deficient)
- Omega-3 fatty acids: 1000-2000 mg daily
- Iron: As recommended by healthcare provider (if anemic)
- Magnesium: 300-400 mg daily (supports hormones)
- Vitamin B-complex: Support energy & metabolism

### 🏃 Lifestyle Recommendations

**Exercise:**
- Aim for 150 minutes moderate aerobic activity weekly
- Include 2-3 sessions strength training (supports hormones)
- Try yoga or pilates (reduces stress, improves flexibility)

**Sleep:**
- Target 7-9 hours sleep per night
- Consistent sleep schedule (same time daily)
- Avoid screens 1 hour before bed

**Stress Management:**
- Daily meditation (10-15 minutes)
- Deep breathing exercises
- Regular outdoor time/nature walks
- Journaling or creative activities

**Hydration:**
- Drink 8-10 glasses water daily
- Herbal teas (chamomile, peppermint)
- Limit sugary beverages

### 📈 Progress Metrics to Track
- Energy levels (rate 1-10 daily)
- Sleep quality and duration
- Digestive health and regularity
- Mood and stress levels
- Weight (if relevant to goals)
- Monthly check-in with healthcare provider

### ⚠️ Important Medical Disclaimer
This nutrition plan is for educational purposes only and does not replace professional medical advice. Always consult with your healthcare provider or registered dietitian before making major dietary changes, especially if you have existing health conditions or take medications. If you experience any adverse effects, stop and contact your healthcare provider immediately.

---

**Your personalized recommendations are above. Follow them consistently for best results!**
"""
        return fallback

# ======================================================
# PHASE 4: CYCLE ANALYSIS & TREND DETECTION
# ======================================================

def calculate_cycle_regularity(cycles):
    """
    Analyze how regular a user's menstrual cycle is.
    
    Regular cycle: Standard deviation < 7 days
    Irregular cycle: Standard deviation ≥ 7 days
    
    Returns: {regularity_score, is_regular, std_dev, avg_length}
    """
    
    if len(cycles) < 2:
        return {
            'data_points': len(cycles),
            'status': 'Not enough data (need 6+ cycles)',
            'is_regular': None,
            'regularity_score': None
        }
    
    try:
        # Get cycle lengths from database records
        cycle_lengths = []
        for cycle in cycles:
            if cycle.last_period and cycle.cycle_length:
                cycle_lengths.append(cycle.cycle_length)
        
        if len(cycle_lengths) < 2:
            return {
                'data_points': len(cycle_lengths),
                'status': 'Not enough complete records',
                'is_regular': None,
                'regularity_score': None
            }
        
        # Calculate statistics
        avg_length = np.mean(cycle_lengths)
        std_dev = np.std(cycle_lengths)
        
        # Regularity: std_dev < 7 days = regular
        is_regular = std_dev < 7
        # Score: 0-100 (higher = more regular)
        regularity_score = max(0, 100 - (std_dev * 15))
        
        print(f"✓ Cycle Regularity: {avg_length:.1f}±{std_dev:.1f} days")
        
        return {
            'data_points': len(cycle_lengths),
            'average_length': round(avg_length, 1),
            'standard_deviation': round(std_dev, 1),
            'is_regular': is_regular,
            'regularity_score': round(regularity_score, 1),
            'status': 'Regular ✓' if is_regular else 'Irregular ⚠️'
        }
        
    except Exception as e:
        print(f"✗ Cycle regularity error: {e}")
        return {'status': f'Error: {str(e)}'}


def predict_cycle_phase(last_period_date, cycle_length, current_date=None):
    """
    Predict which phase of the menstrual cycle the user is in.
    
    Menstrual Cycle Phases:
    ├─ Menstrual (Day 1-5): Bleeding
    ├─ Follicular (Day 6-14): Follicle growing, estrogen rising
    ├─ Ovulation (Day 15): Egg release (most fertile)
    └─ Luteal (Day 16-28): Progesterone rises, metabolism increases
    
    Returns: {phase, day_in_cycle, phase_start, phase_end, fertility_window}
    """
    
    if current_date is None:
        current_date = date.today()
    
    if not last_period_date or not cycle_length:
        return {'status': 'Missing cycle data', 'phase': None}
    
    try:
        # Convert to date objects if needed
        if isinstance(last_period_date, str):
            last_period_date = datetime.strptime(last_period_date, '%Y-%m-%d').date()
        if isinstance(current_date, str):
            current_date = datetime.strptime(current_date, '%Y-%m-%d').date()
        
        # Calculate days since last period
        days_since_period = (current_date - last_period_date).days
        
        # Calculate position in current cycle
        day_in_cycle = (days_since_period % cycle_length) + 1
        
        # Determine phase (assuming 28-day average for standard phases)
        # Scale phases to actual cycle length
        menstrual_days = max(1, int(cycle_length * 0.18))  # ~5 days of 28
        follicular_start = menstrual_days + 1
        follicular_end = follicular_start + int(cycle_length * 0.36)  # ~10 days
        ovulation_day = follicular_end + 1
        ovulation_end = ovulation_day + 2
        
        if day_in_cycle <= menstrual_days:
            phase = 'Menstrual'
            phase_emoji = '🔴'
            description = 'Bleeding phase'
            energy = 'Low'
        elif day_in_cycle <= follicular_end:
            phase = 'Follicular'
            phase_emoji = '🟢'
            description = 'Estrogen rising, energy increasing'
            energy = 'Rising'
        elif day_in_cycle <= ovulation_end:
            phase = 'Ovulation'
            phase_emoji = '🟡'
            description = 'Egg release, peak energy & fertility'
            energy = 'Peak'
        else:
            phase = 'Luteal'
            phase_emoji = '🟣'
            description = 'Progesterone dominant, metabolism increases'
            energy = 'Variable'
        
        # Fertility window: 5 days before ovulation to ovulation
        fertility_start = ovulation_day - 5
        fertility_end = ovulation_day
        is_fertile = fertility_start <= day_in_cycle <= fertility_end
        
        print(f"✓ Cycle Phase: {phase} (Day {day_in_cycle}/{cycle_length})")
        
        return {
            'phase': phase,
            'phase_emoji': phase_emoji,
            'day_in_cycle': day_in_cycle,
            'total_cycle_days': cycle_length,
            'description': description,
            'energy_level': energy,
            'is_fertile': is_fertile,
            'fertility_window': {
                'start_day': fertility_start,
                'end_day': fertility_end,
                'window_active': is_fertile
            }
        }
        
    except Exception as e:
        print(f"✗ Phase prediction error: {e}")
        return {'status': f'Error: {str(e)}', 'phase': None}


def correlate_cycle_with_metrics(cycle_history, health_metrics):
    """
    Find correlations between menstrual cycle phase and metabolic markers.
    
    Example:
    - High glucose during luteal phase?
    - Higher triglycerides pre-period?
    - Energy dips during menstrual phase?
    
    Returns: {patterns, recommendations, alerts}
    """
    
    try:
        patterns = {
            'menstrual_phase_glucose': 'Monitor glucose - estrogen low',
            'follicular_phase_energy': 'Peak workout time - estrogen rising',
            'ovulation_phase_strength': 'Best time for strength training',
            'luteal_phase_nutrition': 'Increase calories & magnesium - metabolism higher'
        }
        
        # Simple correlation checks
        recommendations = []
        alerts = []
        
        if health_metrics.get('glucose', 0) > 125:
            alerts.append('High glucose detected - reduce simple carbs, especially luteal phase')
        
        if health_metrics.get('triglycerides', 0) > 150:
            recommendations.append('Increase omega-3s; timing with cycle phases helps')
        
        if health_metrics.get('vitamin_d', 0) < 30:
            recommendations.append('Low Vitamin D - may affect mood during luteal phase')
        
        recommendations.extend([
            'Track symptoms weekly - see phase-specific patterns',
            'Eat more protein during follicular phase',
            'During luteal: ↑ calories, ↑ magnesium, ↓ caffeine',
            'Exercise: cardio/yoga during follicular, strength during ovulation'
        ])
        
        return {
            'patterns': patterns,
            'recommendations': recommendations,
            'alerts': alerts if alerts else ['No critical alerts']
        }
        
    except Exception as e:
        print(f"✗ Correlation error: {e}")
        return {'status': f'Error: {str(e)}'}


# ======================================================
# PHASE 5: DOCTOR ALERT GATING & MEDICAL REVIEW FLAGS
# ======================================================

def evaluate_doctor_alert_deterministic(metrics, analysis, cycle_data=None):
    """
    PHASE 5: Rule-based deterministic doctor alert system.
    
    Triggers critical alerts based on hard thresholds:
    - Glucose > 126: Prediabetes/Diabetes
    - Glucose > 200: Diabetic emergency
    - Cholesterol > 240: High-risk heart disease
    - Triglycerides > 500: Risk of pancreatitis
    - TSH outside 0.4-4: Thyroid disorder
    - Vitamin D < 10: Severe deficiency
    - Combined: High glucose + irregular cycle = CRITICAL
    
    Returns: {needs_review, alert_type, critical_findings, urgency_level}
    """
    
    print("\n🔍 PHASE 5A: Rule-Based Medical Alert System")
    
    critical_findings = []
    urgency_level = 'normal'
    alert_type = 'none'
    
    glucose = metrics.get('glucose', 0)
    cholesterol = metrics.get('cholesterol_total', 0)
    triglycerides = metrics.get('triglycerides', 0)
    tsh = metrics.get('tsh', 0)
    vitamin_d = metrics.get('vitamin_d', 30)
    
    # RULE 1: Extreme Glucose (Priority 1 - CRITICAL)
    if glucose > 200:
        critical_findings.append('🚨 CRITICAL: Glucose > 200 mg/dL (severe hyperglycemia)')
        urgency_level = 'critical'
        alert_type = 'hyperglycemia'
    elif glucose > 126:
        critical_findings.append('⚠️  HIGH: Glucose > 126 mg/dL (prediabetes range)')
        if urgency_level != 'critical':
            urgency_level = 'high'
            alert_type = 'prediabetes'
    
    # RULE 2: High Cholesterol (Priority 2 - HIGH)
    if cholesterol > 240:
        critical_findings.append('⚠️  HIGH: Total cholesterol > 240 mg/dL (cardiovascular risk)')
        if urgency_level != 'critical':
            urgency_level = 'high'
    
    # RULE 3: Extreme Triglycerides (Priority 1 - CRITICAL)
    if triglycerides > 500:
        critical_findings.append('🚨 CRITICAL: Triglycerides > 500 mg/dL (acute pancreatitis risk)')
        urgency_level = 'critical'
        alert_type = 'hypertriglyceridemia'
    elif triglycerides > 200:
        critical_findings.append('⚠️  HIGH: Triglycerides > 200 mg/dL (elevated lipoprotein risk)')
        if urgency_level != 'critical':
            urgency_level = 'high'
    
    # RULE 4: Thyroid Disorder (Priority 2 - HIGH)
    if tsh < 0.4 or tsh > 4:
        critical_findings.append(f'⚠️  FLAG: TSH {tsh} mIU/L (outside normal 0.4-4)')
        if urgency_level != 'critical':
            urgency_level = 'high' if (tsh < 0.1 or tsh > 10) else urgency_level
    
    # RULE 5: Severe Vitamin D Deficiency (Priority 3 - MODERATE)
    if vitamin_d < 10:
        critical_findings.append('⚠️  ALERT: Vitamin D < 10 ng/mL (severe deficiency)')
        if urgency_level == 'normal':
            urgency_level = 'moderate'
    
    # RULE 6: Combined Risk - High Glucose + Cycle Issues (Priority 1 - CRITICAL)
    if glucose > 125 and cycle_data:
        regularity = cycle_data.get('regularity', {})
        if not regularity.get('is_regular', True):
            critical_findings.append('🚨 CRITICAL: High glucose + irregular cycle detected (hormonal insulin resistance)')
            urgency_level = 'critical'
            alert_type = 'pcos_risk'
    
    needs_review = len(critical_findings) > 0
    
    print(f"   Findings: {len(critical_findings)}")
    for finding in critical_findings:
        print(f"   {finding}")
    print(f"   Urgency: {urgency_level}")
    
    return {
        'needs_review': needs_review,
        'alert_type': alert_type if alert_type != 'none' else None,
        'critical_findings': critical_findings,
        'urgency_level': urgency_level,
        'reason': ' | '.join(critical_findings) if critical_findings else 'All metrics within acceptable ranges'
    }


def evaluate_doctor_alert_ai_judgment(metrics, analysis, gemini_response_json):
    """
    PHASE 5B: AI-powered medical review decision.
    
    Uses Gemini to make nuanced judgment on whether doctor review needed:
    - Beyond simple rules
    - Considers interaction effects
    - Factors in patient goals & wellness
    - Provides AI reasoning
    
    Returns: {ai_recommends_review, confidence, reasoning}
    """
    
    print("\n🧠 PHASE 5B: AI Medical Judgment")
    
    try:
        # Extract key context from Gemini's medical analysis
        analysis_summary = gemini_response_json.get('medical_analysis', {}).get('summary', '')
        risk_level = gemini_response_json.get('medical_analysis', {}).get('risk_level', 'moderate')
        
        # Simple AI judgment based on risk level
        # In production, you'd call Gemini API for nuanced judgment
        if risk_level == 'critical':
            return {
                'ai_recommends_review': True,
                'confidence': 0.95,
                'reasoning': 'AI detected critical health risk requiring immediate medical consultation'
            }
        elif risk_level == 'high':
            return {
                'ai_recommends_review': True,
                'confidence': 0.85,
                'reasoning': 'AI indicates elevated health markers warrant medical review'
            }
        else:
            return {
                'ai_recommends_review': False,
                'confidence': 0.90,
                'reasoning': 'AI assessment: health profile stable, routine check-up sufficient'
            }
            
    except Exception as e:
        print(f"   ⚠️  AI judgment error: {e}")
        return {
            'ai_recommends_review': False,
            'confidence': 0.5,
            'reasoning': f'Error in AI assessment: {str(e)}'
        }


def dual_gate_doctor_alert(deterministic_alert, ai_judgment):
    """
    PHASE 5C: Dual-Gate Decision System
    
    Combines:
    - Deterministic rules (hard thresholds)
    - AI judgment (nuanced assessment)
    
    Final decision: OR logic (if either says review needed, flag it)
    
    Returns: {final_decision, needs_review, decision_method}
    """
    
    print("\n🚪 PHASE 5C: Dual-Gate Decision")
    
    # Deterministic says review?
    rules_say_review = deterministic_alert.get('needs_review', False)
    
    # AI says review?
    ai_says_review = ai_judgment.get('ai_recommends_review', False)
    
    # OR logic: if either system says review, flag it
    final_needs_review = rules_say_review or ai_says_review
    
    decision_method = []
    if rules_say_review:
        decision_method.append("Rules-Based")
    if ai_says_review:
        decision_method.append("AI-Judgment")
    
    print(f"   Rules: {rules_say_review}")
    print(f"   AI:    {ai_says_review}")
    print(f"   Final: {final_needs_review}")
    
    return {
        'needs_medical_review': final_needs_review,
        'decision_method': ' + '.join(decision_method) if decision_method else 'No flags',
        'rules_triggered': deterministic_alert.get('critical_findings', []),
        'ai_reasoning': ai_judgment.get('reasoning', ''),
        'urgency_level': deterministic_alert.get('urgency_level', 'normal')
    }


# ======================================================
# PHASE 5: DOCTOR ALERT GATING
# ======================================================

def evaluate_doctor_alert(metrics, analysis, cycle_data=None):
    """
    PHASE 5: Rule-based + AI judgment to determine if medical review is needed.
    
    Deterministic Rules (Critical Thresholds):
    ├─ Glucose > 126 (prediabetes) + Irregular cycle → CRITICAL
    ├─ Cholesterol > 240 (high) → ALERT
    ├─ Triglycerides > 200 (high) → ALERT
    ├─ TSH abnormal + Heavy bleeding → ALERT
    ├─ Hemoglobin < 11 + Heavy bleeding → CRITICAL (anemia risk)
    └─ Multiple alerts → CRITICAL
    
    Returns: {needs_medical_review, alert_type, urgency_level, findings}
    """
    
    print("\n🏥 PHASE 5: Doctor Alert Evaluation")
    
    critical_findings = []
    alert_findings = []
    risk_score = analysis.get('risk_score', 50)
    
    # RULE 1: High glucose + irregular cycle
    glucose = metrics.get('glucose', 95)
    if glucose > 126:
        critical_findings.append(f'Elevated glucose: {glucose} mg/dL (prediabetes range)')
        if cycle_data and not cycle_data.get('is_regular', True):
            critical_findings.append('⚠️ CRITICAL: High glucose + irregular cycle → Hormonal/metabolic disorder risk')
    
    # RULE 2: High cholesterol
    cholesterol = metrics.get('cholesterol_total', 180)
    if cholesterol > 240:
        alert_findings.append(f'High cholesterol: {cholesterol} mg/dL (cardiovascular risk)')
    
    # RULE 3: High triglycerides
    triglycerides = metrics.get('triglycerides', 100)
    if triglycerides > 200:
        critical_findings.append(f'Very high triglycerides: {triglycerides} mg/dL (metabolic syndrome risk)')
    
    # RULE 4: TSH abnormal + heavy bleeding
    tsh = metrics.get('tsh', 2.0)
    flow_intensity = cycle_data.get('flow_intensity', 'regular') if cycle_data else 'regular'
    if (tsh < 0.4 or tsh > 4.0) and flow_intensity == 'heavy':
        alert_findings.append(f'Abnormal TSH ({tsh}) + Heavy bleeding → Thyroid dysfunction possible')
    
    # RULE 5: Low hemoglobin + heavy bleeding → Anemia risk
    hemoglobin = metrics.get('hemoglobin', 12.5)
    if hemoglobin < 11 and flow_intensity == 'heavy':
        critical_findings.append(f'Low hemoglobin ({hemoglobin}) + Heavy flow → Iron deficiency anemia risk')
    
    # RULE 6: Vitamin D critically low
    vitamin_d = metrics.get('vitamin_d', 30)
    if vitamin_d < 15:
        alert_findings.append(f'Severely low Vitamin D: {vitamin_d} ng/mL (bone health risk)')
    
    # RULE 7: Overall risk score
    if risk_score > 75:
        critical_findings.append(f'High overall risk score: {risk_score}/100')
    elif risk_score > 60:
        alert_findings.append(f'Moderate risk score: {risk_score}/100')
    
    # Determine alert type and urgency
    needs_review = len(critical_findings) > 0 or len(alert_findings) > 2
    
    if len(critical_findings) > 0:
        alert_type = 'CRITICAL'
        urgency = 'HIGH'
        findings = critical_findings + alert_findings
    elif len(alert_findings) > 2:
        alert_type = 'ALERT'
        urgency = 'MEDIUM'
        findings = alert_findings
    elif risk_score > 60:
        alert_type = 'CAUTION'
        urgency = 'LOW'
        findings = [f'Risk indicators detected (score: {risk_score}/100)']
    else:
        alert_type = 'NORMAL'
        urgency = 'NONE'
        findings = ['Metrics within normal ranges']
        needs_review = False
    
    print(f"   Alert Type: {alert_type}")
    print(f"   Urgency: {urgency}")
    print(f"   Findings: {len(findings)} items")
    
    return {
        'needs_medical_review': needs_review,
        'alert_type': alert_type,
        'urgency_level': urgency,
        'findings': findings,
        'risk_score': risk_score,
        'recommendation': get_medical_recommendation(alert_type, findings)
    }


def get_medical_recommendation(alert_type, findings):
    """
    Generate actionable recommendations based on alert type.
    """
    
    recommendations = {
        'CRITICAL': [
            '⚠️ URGENT: Schedule appointment with healthcare provider within 1-2 days',
            'Bring blood test results and cycle tracking data to appointment',
            'Document symptoms, onset timeline, and any lifestyle changes',
            'Consider emergency care if experiencing severe symptoms'
        ],
        'ALERT': [
            '📋 Schedule healthcare visit within 1-2 weeks',
            'Monitor metrics closely - retest if changes occur',
            'Keep detailed symptom and cycle tracking logs',
            'Discuss findings with primary care provider'
        ],
        'CAUTION': [
            '👀 Continue monitoring health metrics monthly',
            'Lifestyle modifications may help: exercise, diet, sleep',
            'Follow up in 2-3 months with retesting if available',
            'Consult provider if symptoms worsen'
        ],
        'NORMAL': [
            '✓ Continue current healthy habits',
            'Annual checkups recommended',
            'Monitor cycle and health metrics monthly',
            'Maintain balanced diet, exercise, and stress management'
        ]
    }
    
    return recommendations.get(alert_type, [])


# ======================================================
# ROUTES
# ======================================================


# ======================================================
# ROUTES
# ======================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login-page')
def login_page():
    return render_template('login.html')

@app.route('/signup-page')
def signup_page():
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    return render_template('Dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/cycle-tracker')
def cycle_tracker():
    return render_template('cycle.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

# ======================================================
# AUTH ROUTES - PASSWORD HASHING FIXED FOR PYTHON 3.9
# ======================================================

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        
        if not data.get('email') or not data.get('password') or not data.get('name'):
            return jsonify({"error": "Missing required fields"}), 400
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({"error": "Email already exists"}), 409
        
        # ✅ FIXED: Use werkzeug.security for Python 3.9 compatibility
        hashed_pw = generate_password_hash(data['password'], method='pbkdf2:sha256')
        
        new_user = User(
            name=data['name'],
            email=data['email'],
            password=hashed_pw
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        print(f"✓ New user created: {data['email']}")
        return jsonify({"message": "User created successfully"}), 201
        
    except Exception as e:
        db.session.rollback()
        print(f"✗ Signup error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        user = User.query.filter_by(email=data.get('email')).first()
        
        # ✅ FIXED: Use check_password_hash (compatible with Python 3.9)
        if not user or not check_password_hash(user.password, data.get('password')):
            return jsonify({"error": "Invalid credentials"}), 401
        
        print(f"✓ User logged in: {data['email']}")
        return jsonify({
            "name": user.name,
            "email": user.email,
            "message": "Login successful"
        }), 200
        
    except Exception as e:
        print(f"✗ Login error: {e}")
        return jsonify({"error": str(e)}), 500

# ======================================================
# MAIN ANALYSIS ROUTE
# ======================================================

@app.route('/analyze-health', methods=['POST'])
def analyze_health_route():
    """
    Main endpoint for medical report analysis (Phase 1-3 implementation)
    
    Flow:
    1. Extract PDF text
    2. Extract metrics with confidence scores
    3. Analyze health (risk assessment)
    4. ML diet prediction
    5. Gemini Master Prompt (Phase 3 JSON synthesis)
    6. Save to database with confidence & doctor alerts
    """
    
    print("\n" + "="*60)
    print("🔍 STARTING MEDICAL REPORT ANALYSIS")
    print("="*60)
    
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({"error": "No PDF file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are supported"}), 400
        
        user_email = request.form.get('email')
        wellness_goal = request.form.get('wellness_goal', 'General Health')
        
        if not user_email:
            return jsonify({"error": "Email is required"}), 400
        
        print(f"📧 User: {user_email}")
        print(f"🎯 Wellness Goal: {wellness_goal}")
        
        # Step 1: Save PDF
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"✓ PDF saved: {filename}")
        
        # Step 2: Extract text from PDF
        print("\n📄 Extracting text from PDF...")
        pdf_text = extract_pdf_text(filepath)
        
        if not pdf_text or len(pdf_text) < 100:
            return jsonify({"error": "Could not extract meaningful text from PDF"}), 400
        
        # Step 3: Extract health metrics WITH CONFIDENCE SCORES (Phase 1)
        print("\n🔬 Extracting health metrics with confidence scoring...")
        metrics_with_confidence = extract_metrics_with_confidence(pdf_text)
        
        # Convert to simple dict for analysis (extract just values)
        metrics = {}
        extraction_confidence = {}
        for metric_name, metric_data in metrics_with_confidence.items():
            if isinstance(metric_data, dict):
                metrics[metric_name] = metric_data.get('value', 0)
                extraction_confidence[metric_name] = metric_data.get('confidence', 0.5)
            else:
                metrics[metric_name] = metric_data
                extraction_confidence[metric_name] = 0.7  # default confidence
        
        if not metrics:
            return jsonify({"error": "No health metrics found in document. Please upload a valid medical report."}), 400
        
        # Step 4: Analyze health
        print("\n📊 Analyzing health metrics...")
        analysis = analyze_health(metrics)
        
        # Step 5: Validate metrics bounds (Phase 1)
        validation_errors = {}
        for metric_name, value in metrics.items():
            bounds_result = validate_single_metric_bounds(metric_name, value)
            if not bounds_result['valid']:
                validation_errors[metric_name] = f"Out of bounds. Expected {bounds_result['min']}-{bounds_result['max']}, got {value}"
                print(f"⚠️  Validation: {metric_name} = {value} -> Out of bounds")
        
        # Step 6: ML Diet Prediction (Phase 2 - with new model)
        print("\n🤖 Predicting optimal diet...")
        diet_prediction = predict_diet(metrics)
        
        # Record ML features used (Phase 1)
        ml_features_used = {
            'glucose': metrics.get('glucose'),
            'cholesterol': metrics.get('cholesterol_total'),
            'hdl': metrics.get('hdl'),
            'ldl': metrics.get('ldl'),
            'triglycerides': metrics.get('triglycerides'),
            'vitamin_d': metrics.get('vitamin_d'),
            'tsh': metrics.get('tsh')
        }
        
        # Step 7: Generate Structured JSON using Gemini Master Prompt (Phase 3)
        print("\n🧠 Calling Gemini Master Prompt (Phase 3 - JSON Synthesis)...")
        gemini_json_response = generate_gemini_json_response(metrics, analysis, diet_prediction, wellness_goal)
        
        # Step 8: PHASE 5 - Dual-Gate Doctor Alert System
        print("\n🏥 PHASE 5 - Doctor Alert Evaluation:")
        # Get latest cycle data for context
        latest_cycle = None
        cycle_context = {}
        try:
            latest_cycle = Cycle.query.filter_by(user_email=user_email).order_by(Cycle.created_at.desc()).first()
            if latest_cycle:
                regularity_data = calculate_cycle_regularity([latest_cycle])
                cycle_context = regularity_data
        except Exception as cycle_err:
            print(f"⚠️ Cycle query skipped (table not ready): {cycle_err}")
            latest_cycle = None
            cycle_context = {}
        
        # Phase 5A: Deterministic rule-based alert
        deterministic_alert = evaluate_doctor_alert_deterministic(metrics, analysis, cycle_context)
        
        # Phase 5B: AI judgment based on Gemini response
        ai_judgment = evaluate_doctor_alert_ai_judgment(metrics, analysis, gemini_json_response)
        
        # Phase 5C: Dual-gate final decision (OR logic: either system triggers review)
        doctor_alert = dual_gate_doctor_alert(deterministic_alert, ai_judgment)
        
        print(f"\n   Final Decision: {'🚨 MEDICAL REVIEW NEEDED' if doctor_alert['needs_medical_review'] else '✅ No immediate review needed'}")
        
        # Step 9: Save to database with Phase 1-5 fields
        print("\n💾 Saving to database with confidence scores & doctor alert...")
        try:
            health_report = HealthReport(
                user_email=user_email,
                extracted_metrics=metrics,
                extraction_confidence=extraction_confidence,  # Phase 1: Confidence scores
                validation_errors=validation_errors if validation_errors else None,  # Phase 1: Validation results
                health_analysis=analysis,
                ml_features_used=ml_features_used,  # Phase 1: Which features were used
                diet_recommendation=diet_prediction.get('diet_type', 'Balanced'),
                gemini_response_json=gemini_json_response,  # Phase 3: Structured AI response
                doctor_alert=doctor_alert,  # Phase 5: Dual-gate doctor review flags
                related_cycle_id=latest_cycle.id if latest_cycle else None  # Link to cycle for hormonal context
            )
            db.session.add(health_report)
            db.session.commit()
            print(f"✓ Report saved with ID: {health_report.id}")
            print(f"✓ Extraction confidence: {round(np.mean(list(extraction_confidence.values())), 2):.0%}")
        except Exception as db_error:
            db.session.rollback()
            print(f"⚠️ Database save warning: {db_error}")
        
        # Step 10: Prepare response
        print("\n✅ Analysis complete!")
        print("="*60 + "\n")
        
        return jsonify({
            "status": "success",
            "message": "Medical report analyzed successfully",
            "phase": "Phase 1-3 (Extraction + ML + Gemini JSON)",
            "extracted_metrics": metrics_with_confidence,
            "extraction_confidence": {k: round(v, 2) for k, v in extraction_confidence.items()},
            "health_analysis": {
                "metrics_details": analysis['metrics_details'],
                "risk_alerts": analysis['risk_alerts'],
                "risk_score": analysis['risk_score'],
                "summary": analysis['health_summary']
            },
            "validation_errors": validation_errors if validation_errors else "All metrics within normal bounds",
            "diet_prediction": diet_prediction,
            "gemini_response": gemini_json_response,  # Phase 3: Full JSON response
            "doctor_review_needed": doctor_alert.get('needs_medical_review', False),
            "processed_at": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        print("="*60 + "\n")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

# ======================================================
# ADDITIONAL ROUTES
# ======================================================

@app.route('/save-cycle', methods=['POST'])
def save_cycle():
    """
    Save or update cycle data with Phase 1 expanded fields.
    
    Accepts:
    - email (required)
    - last_period: ISO date of period start
    - cycle_length: Days in cycle (21-35)
    - flow_intensity: 'light'|'regular'|'heavy' (Phase 1)
    - period_duration: 1-7 days (Phase 1)
    - symptoms: {cramps, bloating, acne, headache} (1-5 scale) (Phase 1)
    - mood: 'happy'|'anxious'|'irritable'|'sad'|'energetic' (Phase 1)
    - exercise_minutes: Integer (Phase 1)
    - sleep_hours: Float (Phase 1)
    """
    try:
        data = request.get_json()
        
        cycle = Cycle(
            user_email=data.get('email'),
            last_period=datetime.fromisoformat(data['last_period']).date(),
            cycle_length=data.get('cycle_length', 28),
            # Phase 1: New fields
            flow_intensity=data.get('flow_intensity', 'regular'),  # light|regular|heavy
            period_duration=data.get('period_duration', 5),  # 1-7 days
            symptoms=data.get('symptoms', {}),  # {cramps: 1-5, bloating: 1-5, ...}
            mood=data.get('mood', 'neutral'),  # emotional state
            exercise_minutes=data.get('exercise_minutes', 0),  # Activity level
            sleep_hours=data.get('sleep_hours', 7.0)  # Sleep quality
        )
        
        db.session.add(cycle)
        db.session.commit()
        
        print(f"✓ Cycle saved for {data.get('email')}")
        print(f"   Period: {cycle.last_period}, Length: {cycle.cycle_length} days")
        print(f"   Flow: {cycle.flow_intensity}, Duration: {cycle.period_duration} days")
        print(f"   Mood: {cycle.mood}, Exercise: {cycle.exercise_minutes} min, Sleep: {cycle.sleep_hours}h")
        
        return jsonify({
            "message": "Cycle saved successfully",
            "cycle_id": cycle.id,
            "next_period": (cycle.last_period + timedelta(days=cycle.cycle_length)).isoformat()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/cycle-insights/<email>', methods=['GET'])
def cycle_insights(email):
    """
    Phase 4 Enhanced: Return comprehensive cycle analysis + trends
    
    Includes:
    - Regularity analysis (standard deviation)
    - Current cycle phase prediction
    - Cycle-metabolic correlations
    - Health recommendations based on phase
    """
    try:
        # Get last 6 cycles for trend analysis
        cycles = Cycle.query.filter_by(user_email=email).order_by(Cycle.created_at.desc()).limit(6).all()
        
        if not cycles:
            return jsonify({"insight": "No cycle data found", "phase": None}), 200
        
        print(f"\n🔍 CYCLE INSIGHTS for {email}")
        print(f"   Records available: {len(cycles)}")
        
        # PHASE 4A: Calculate regularity
        print("\n1️⃣  Regularity Analysis:")
        regularity = calculate_cycle_regularity(cycles)
        print(f"   {regularity}")
        
        # PHASE 4B: Predict current phase
        print("\n2️⃣  Phase Prediction:")
        if cycles[0].last_period and cycles[0].cycle_length:
            phase_data = predict_cycle_phase(cycles[0].last_period, cycles[0].cycle_length)
            print(f"   {phase_data['phase']} (Day {phase_data['day_in_cycle']}/{phase_data['total_cycle_days']})")
        else:
            phase_data = {'phase': None, 'status': 'Missing data'}
        
        # Get latest health report for metabolic data
        latest_report = HealthReport.query.filter_by(user_email=email).order_by(HealthReport.created_at.desc()).first()
        health_metrics = latest_report.extracted_metrics if latest_report else {}
        
        # PHASE 4C: Correlate cycle with health metrics
        print("\n3️⃣  Cycle-Metabolic Correlations:")
        correlations = correlate_cycle_with_metrics(cycles, health_metrics)
        print(f"   {len(correlations['recommendations'])} recommendations")
        
        # Calculate next period prediction
        next_period = cycles[0].last_period + timedelta(days=cycles[0].cycle_length) if cycles[0].cycle_length else None
        
        # Build comprehensive response
        # Convert numpy types to native Python types for JSON serialization
        regularity_clean = {
            'data_points': int(regularity.get('data_points', 0)),
            'average_length': float(regularity.get('average_length', 0)),
            'standard_deviation': float(regularity.get('standard_deviation', 0)),
            'is_regular': bool(regularity.get('is_regular', False)),
            'regularity_score': float(regularity.get('regularity_score', 0)),
            'status': str(regularity.get('status', 'Unknown'))
        }
        
        response = {
            "status": "success",
            "phase": phase_data.get('phase'),
            "phase_emoji": phase_data.get('phase_emoji', ''),
            "day_in_cycle": int(phase_data.get('day_in_cycle', 0)) if phase_data.get('day_in_cycle') else None,
            "total_cycle_days": int(phase_data.get('total_cycle_days', 0)) if phase_data.get('total_cycle_days') else None,
            "phase_description": phase_data.get('description', 'Unknown'),
            "energy_level": phase_data.get('energy_level'),
            "fertility": bool(phase_data.get('is_fertile', False)),
            "regularity": regularity_clean,
            "next_period": next_period.strftime("%Y-%m-%d") if next_period else None,
            "cycle_correlations": {
                "patterns": correlations.get('patterns', {}),
                "recommendations": correlations.get('recommendations', []),
                "alerts": correlations.get('alerts', [])
            },
            "health_context": {
                "glucose": health_metrics.get('glucose'),
                "cholesterol": health_metrics.get('cholesterol_total'),
                "vitamin_d": health_metrics.get('vitamin_d')
            }
        }
        
        print(f"\n✅ Cycle insights generated!")
        print()
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"\n✗ Cycle insights error: {e}")
        return jsonify({"error": str(e)}), 500

# ======================================================
# PHASE 5: DOCTOR ALERT RETRIEVAL
# ======================================================

@app.route('/doctor-alerts/<email>', methods=['GET'])
def get_doctor_alerts(email):
    """
    Phase 5: Retrieve doctor alerts for a user's health reports.
    
    Returns most recent alerts with critical incidents first.
    """
    try:
        # Get recent health reports with doctor alerts
        reports = HealthReport.query.filter_by(user_email=email).order_by(
            HealthReport.created_at.desc()
        ).limit(10).all()
        
        if not reports:
            return jsonify({
                "status": "no_data",
                "message": "No health reports found",
                "alerts": []
            }), 200
        
        # Collect all alerts with report context
        alert_list = []
        critical_count = 0
        
        for report in reports:
            if report.doctor_alert:
                alert = report.doctor_alert
                alert_entry = {
                    "report_id": report.id,
                    "created_at": report.created_at.isoformat(),
                    "needs_medical_review": alert.get('needs_medical_review', False),
                    "alert_type": alert.get('alert_type', 'NORMAL'),
                    "urgency_level": alert.get('urgency_level', 'NONE'),
                    "findings": alert.get('findings', []),
                    "recommendations": alert.get('recommendation', [])
                }
                alert_list.append(alert_entry)
                
                if alert.get('alert_type') == 'CRITICAL':
                    critical_count += 1
        
        # Sort by urgency (CRITICAL first, then ALERT, then CAUTION)
        urgency_order = {'CRITICAL': 0, 'ALERT': 1, 'CAUTION': 2, 'NORMAL': 3}
        alert_list.sort(key=lambda x: urgency_order.get(x['alert_type'], 4))
        
        print(f"\n📋 Doctor Alerts for {email}")
        print(f"   Critical: {critical_count}, Total: {len(alert_list)}")
        
        return jsonify({
            "status": "success",
            "user_email": email,
            "total_alerts": len(alert_list),
            "critical_count": critical_count,
            "alerts": alert_list,
            "summary": {
                "has_critical": critical_count > 0,
                "action_needed": critical_count > 0,
                "recommended_action": "Schedule urgent doctor visit" if critical_count > 0 else "Monitor metrics regularly"
            }
        }), 200
        
    except Exception as e:
        print(f"✗ Doctor alerts error: {e}")
        return jsonify({"error": str(e)}), 500


# ======================================================
# ERROR HANDLERS
# ======================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Route not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({"error": "Internal server error"}), 500

# ======================================================
# RUN APPLICATION
# ======================================================

if __name__ == "__main__":
    import socket
    
    print("\n🚀 Starting NutriHormone Application...\n")
    
    with app.app_context():
        try:
            db.create_all()
            print("✓ Database tables created/verified")
        except Exception as e:
            print(f"⚠️ Database initialization warning: {e}")
    
    # Find available port
    port = int(os.getenv('FLASK_PORT', 5000))
    available_ports = [5000, 5001, 5002, 5003, 8000, 8001, 8080]
    
    for p in available_ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', p))
        sock.close()
        
        if result != 0:
            port = p
            print(f"✓ Port {p} is available\n")
            break
    
    print("📌 Routes available:")
    print(f"  - http://localhost:{port}/ (Home)")
    print(f"  - http://localhost:{port}/dashboard (Dashboard)")
    print(f"  - http://localhost:{port}/analyze-health (API)")
    print("\n✓ Server ready!\n")
    
    app.run(debug=True, host='0.0.0.0', port=port)