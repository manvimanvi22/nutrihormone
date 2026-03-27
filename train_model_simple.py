"""
Simple ML Model Training for Diet Recommendations
3rd Year Undergraduate Level - Easy to Understand

Features: Glucose, Cholesterol, HDL, LDL, Triglycerides, Vitamin D, TSH
Target: Diet Category (Balanced, High-Protein, Mediterranean, Low-Carb, Keto)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ============================================================================
# STEP 1: CREATE SIMPLE TRAINING DATA
# ============================================================================
# Real doctors use these blood metrics to recommend diets
# Example: High glucose + high triglycerides = recommend Low-Carb diet

def create_training_data():
    """
    Creates 300 simple examples of blood metrics and recommended diets.
    Each row = one patient's blood test results + their recommended diet.
    """
    
    np.random.seed(42)  # Same random numbers every run (reproducible)
    samples = 300
    
    # Initialize arrays to store data
    data = []
    labels = []
    
    # Create 300 patient examples
    for i in range(samples):
        # Generate random but realistic blood metric values
        glucose = np.random.uniform(70, 200)        # mg/dL (normal: 70-100)
        cholesterol = np.random.uniform(120, 300)   # mg/dL (normal: <200)
        hdl = np.random.uniform(25, 70)             # mg/dL (higher is better)
        ldl = np.random.uniform(40, 200)            # mg/dL (lower is better)
        triglycerides = np.random.uniform(30, 400)  # mg/dL (normal: <150)
        vitamin_d = np.random.uniform(10, 70)       # ng/mL (optimal: >30)
        tsh = np.random.uniform(0.4, 5.0)           # mIU/L (normal: 0.4-4.0)
        
        # Create one row of features
        features = [glucose, cholesterol, hdl, ldl, triglycerides, vitamin_d, tsh]
        
        # STEP 2: SIMPLE RULES to assign diet category
        # These rules mimic what a nutritionist would recommend
        
        if glucose > 125 and triglycerides > 150:
            # High glucose + High triglycerides = Low-Carb diet
            diet = "Low-Carb"
        elif ldl > 130 and cholesterol > 240:
            # High LDL cholesterol = Mediterranean diet
            diet = "Mediterranean"
        elif glucose < 90 and hdl > 50:
            # Good glucose control + high HDL = Balanced diet
            diet = "Balanced"
        elif triglycerides > 200 and vitamin_d < 20:
            # Very high triglycerides = Keto diet (strict)
            diet = "Keto"
        else:
            # Normal range = High-Protein (for muscle maintenance)
            diet = "High-Protein"
        
        data.append(features)
        labels.append(diet)
    
    # Convert to pandas DataFrame (easier to work with)
    df = pd.DataFrame(data, columns=['Glucose', 'Cholesterol', 'HDL', 'LDL', 
                                      'Triglycerides', 'Vitamin_D', 'TSH'])
    df['Diet'] = labels
    
    print("✅ Training Data Created: 300 samples")
    print(f"   Features: {list(df.columns[:-1])}")
    print(f"   Diet Categories: {df['Diet'].unique()}")
    print(f"\n   Sample Distribution:")
    print(df['Diet'].value_counts())
    print()
    
    return df


# ============================================================================
# STEP 3: PREPARE DATA FOR MACHINE LEARNING
# ============================================================================

def prepare_data(df):
    """
    Separate features (X) from target (y)
    Split into training (80%) and testing (20%)
    """
    
    # X = features (what we feed to the model)
    # y = target (what we want to predict)
    X = df.drop('Diet', axis=1)  # All columns except 'Diet'
    y = df['Diet']                # Just the 'Diet' column
    
    # Split: 80% for training models, 20% for testing it
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("✅ Data Prepared:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    print()
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# STEP 4: TRAIN RANDOM FOREST MODEL (Simple & Effective)
# ============================================================================

def train_model(X_train, y_train):
    """
    Random Forest = Like asking 100 decision trees for their opinion,
    then picking the most common answer. Simple but powerful!
    
    Why Random Forest?
    - Easy to understand
    - Works well with blood metrics (nonlinear relationships)
    - Handles multiple diet categories naturally
    - No complex tuning needed
    """
    
    # Create Random Forest with 100 trees
    # max_depth=10: Each tree can be at most 10 levels deep (prevents overfitting)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # Use all CPU cores (faster training)
    )
    
    # Train the model (learn patterns from data)
    model.fit(X_train, y_train)
    
    print("✅ Model Trained (Random Forest with 100 trees)")
    print()
    
    return model


# ============================================================================
# STEP 5: TEST MODEL & SHOW RESULTS
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Check how accurate the model is on both training and testing data.
    """
    
    # Predictions on training data
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Predictions on testing data (more important - unseen patients)
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Training Accuracy: {train_accuracy:.1%}  (model memorized training data)")
    print(f"Testing Accuracy:  {test_accuracy:.1%}  (how it works on NEW patients)")
    print()
    
    # Show detailed breakdown for each diet type
    print("Performance by Diet Category:")
    print(classification_report(y_test, y_test_pred))
    print()
    
    # Feature importance: Which blood metrics matter most?
    print("Most Important Blood Metrics:")
    feature_names = ['Glucose', 'Cholesterol', 'HDL', 'LDL', 
                     'Triglycerides', 'Vitamin_D', 'TSH']
    importances = model.feature_importances_
    
    for name, importance in sorted(zip(feature_names, importances), 
                                   key=lambda x: x[1], reverse=True):
        print(f"   • {name}: {importance:.1%}")
    print()
    
    return test_accuracy


# ============================================================================
# STEP 6: SAVE MODEL FOR LATER USE IN FLASK APP
# ============================================================================

def save_model(model, filepath='diet_model.pkl'):
    """
    Save the trained model to a file so Flask app can load it later.
    joblib = Python library for saving ML models
    """
    
    joblib.dump(model, filepath)
    file_size_kb = os.path.getsize(filepath) / 1024
    
    print(f"✅ Model Saved: {filepath} ({file_size_kb:.1f} KB)")
    print()


# ============================================================================
# STEP 7: DEMONSTRATION - MAKE A PREDICTION
# ============================================================================

def demo_prediction(model):
    """
    Show how to use the trained model to predict for a new patient.
    """
    
    print("=" * 60)
    print("DEMO: Predicting Diet for a New Patient")
    print("=" * 60)
    
    # Example patient's blood test results
    # Format: [Glucose, Cholesterol, HDL, LDL, Triglycerides, Vitamin_D, TSH]
    patient_metrics = np.array([[
        125,    # Glucose: slightly elevated
        240,    # Cholesterol: high
        45,     # HDL: low (bad)
        160,    # LDL: high (bad)
        180,    # Triglycerides: high
        22,     # Vitamin D: low (deficient)
        1.2     # TSH: normal
    ]])
    
    prediction = model.predict(patient_metrics)[0]
    confidence = model.predict_proba(patient_metrics)[0].max()
    
    print(f"Patient's Blood Metrics:")
    print(f"   Glucose: 125 mg/dL (↑ elevated)")
    print(f"   Cholesterol: 240 mg/dL (↑ high)")
    print(f"   HDL: 45 mg/dL (↓ low)")
    print(f"   LDL: 160 mg/dL (↑ high)")
    print(f"   Triglycerides: 180 mg/dL (↑ high)")
    print(f"   Vitamin D: 22 ng/mL (↓ deficient)")
    print(f"   TSH: 1.2 mIU/L (normal)")
    print()
    print(f"🎯 Recommended Diet: {prediction}")
    print(f"   Confidence: {confidence:.0%}")
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NUTRIHORMONE - ML MODEL TRAINING (Phase 2)")
    print("Simple Student-Level Implementation")
    print("=" * 60 + "\n")
    
    # Step 1: Create training data
    df = create_training_data()
    
    # Step 2: Prepare data (split into train/test)
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Step 3: Train the model
    model = train_model(X_train, y_train)
    
    # Step 4: Evaluate the model
    evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Step 5: Save the model
    save_model(model, 'diet_model.pkl')
    
    # Step 6: Demo prediction
    demo_prediction(model)
    
    print("=" * 60)
    print("✅ PHASE 2 COMPLETE!")
    print("   diet_model.pkl is ready for Flask app")
    print("=" * 60 + "\n")