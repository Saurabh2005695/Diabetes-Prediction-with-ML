import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Paths to models
MODEL_PATH = 'diabetes_model.pkl'
SCALER_PATH = 'scaler.pkl'
IMPUTER_PATH = 'imputer.pkl'
METRICS_PATH = 'model_metrics.json'

# Global model state
active_model = None
active_scaler = None
active_imputer = None
active_metrics = {}
custom_model_loaded = False
custom_model_filename = ""

FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

FEATURE_BASELINES = {
    'Pregnancies': {'normal_min': 0, 'normal_max': 5, 'unit': 'count', 'label': 'Pregnancies'},
    'Glucose': {'normal_min': 70, 'normal_max': 125, 'unit': 'mg/dL', 'label': 'Fasting Glucose'},
    'BloodPressure': {'normal_min': 60, 'normal_max': 80, 'unit': 'mmHg', 'label': 'Diastolic BP'},
    'SkinThickness': {'normal_min': 10, 'normal_max': 30, 'unit': 'mm', 'label': 'Triceps Fold'},
    'Insulin': {'normal_min': 15, 'normal_max': 160, 'unit': 'mu U/ml', 'label': '2-Hr Insulin'},
    'BMI': {'normal_min': 18.5, 'normal_max': 24.9, 'unit': 'kg/m²', 'label': 'Body Mass Index'},
    'DiabetesPedigreeFunction': {'normal_min': 0.0, 'normal_max': 0.5, 'unit': 'score', 'label': 'Family Pedigree'},
    'Age': {'normal_min': 21, 'normal_max': 45, 'unit': 'yrs', 'label': 'Age'}
}

def load_system_assets():
    global active_model, active_scaler, active_imputer, active_metrics, custom_model_loaded
    try:
        if os.path.exists(MODEL_PATH):
            active_model = joblib.load(MODEL_PATH)
        if os.path.exists(SCALER_PATH):
            active_scaler = joblib.load(SCALER_PATH)
        if os.path.exists(IMPUTER_PATH):
            active_imputer = joblib.load(IMPUTER_PATH)
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                active_metrics = json.load(f)
        custom_model_loaded = False
        print("[+] Core ML model assets loaded successfully.")
    except Exception as e:
        print(f"[-] Error loading core assets: {e}")

# Load models on server boot
load_system_assets()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    global active_model, active_scaler, active_imputer
    
    if active_model is None:
        return jsonify({'error': 'ML model is not loaded. Please train or upload a model.'}), 500
    
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid JSON request payload.'}), 400

    try:
        # Extract features in required exact order
        raw_values = []
        patient_dict = {}
        for feat in FEATURE_NAMES:
            val = float(data.get(feat, 0))
            raw_values.append(val)
            patient_dict[feat] = val

        raw_df = pd.DataFrame([raw_values], columns=FEATURE_NAMES)
        
        # Apply imputer if present, otherwise pass raw_df to scaler
        if active_imputer is not None:
            zero_fields = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            raw_df_imp = raw_df.copy()
            for col in zero_fields:
                if col in raw_df_imp.columns and raw_df_imp.loc[0, col] == 0:
                    raw_df_imp.loc[0, col] = np.nan
            processed_input = active_imputer.transform(raw_df_imp)
        else:
            processed_input = raw_df

        # Scale data if scaler is available
        if active_scaler is not None:
            if not isinstance(processed_input, pd.DataFrame):
                processed_input = pd.DataFrame(processed_input, columns=FEATURE_NAMES)
            scaled_input = active_scaler.transform(processed_input)
        else:
            scaled_input = processed_input.values if isinstance(processed_input, pd.DataFrame) else processed_input

        # Make prediction
        pred_class = int(active_model.predict(scaled_input)[0])
        
        if hasattr(active_model, 'predict_proba'):
            probabilities = active_model.predict_proba(scaled_input)[0]
            diabetic_prob = float(probabilities[1])
        else:
            diabetic_prob = 1.0 if pred_class == 1 else 0.0

        risk_percentage = round(diabetic_prob * 100, 1)

        # Risk severity classification
        if risk_percentage < 30.0:
            severity = 'Low'
            color_code = '#10b981'  # Green
            assessment = "Low likelihood of diabetes. Maintain standard healthy lifestyle and regular checkups."
        elif risk_percentage < 65.0:
            severity = 'Moderate'
            color_code = '#f59e0b'  # Amber/Orange
            assessment = "Moderate risk detected. Elevated glycemic or metabolic markers observed. Dietary adjustments and preventive screening recommended."
        else:
            severity = 'High'
            color_code = '#ef4444'  # Red
            assessment = "High risk of diabetes. Clinical evaluation with an endocrinologist and a Formal HbA1c test are strongly recommended."

        # Risk factors analysis
        risk_contributions = []
        for feat in FEATURE_NAMES:
            val = patient_dict[feat]
            meta = FEATURE_BASELINES[feat]
            norm_max = meta['normal_max']
            status = 'Normal'
            pct_diff = 0
            
            if val > norm_max:
                status = 'Elevated'
                pct_diff = round(((val - norm_max) / norm_max) * 100, 1)
            elif val < meta['normal_min'] and val > 0:
                status = 'Below Normal'
                
            risk_contributions.append({
                'feature': feat,
                'label': meta['label'],
                'value': val,
                'unit': meta['unit'],
                'normal_range': f"{meta['normal_min']} - {meta['normal_max']}",
                'status': status,
                'pct_diff': pct_diff
            })

        # Sort by elevation
        risk_contributions.sort(key=lambda x: x['pct_diff'], reverse=True)

        model_name = getattr(active_model, '__class__', {}).__name__ if active_model else 'Custom ML Model'

        return jsonify({
            'success': True,
            'prediction': pred_class,
            'label': 'Diabetic' if pred_class == 1 else 'Non-Diabetic',
            'risk_percentage': risk_percentage,
            'risk_severity': severity,
            'color_code': color_code,
            'assessment': assessment,
            'risk_contributions': risk_contributions,
            'patient_inputs': patient_dict,
            'model_used': custom_model_filename if custom_model_loaded else active_metrics.get('best_model', model_name),
            'is_custom_model': custom_model_loaded
        })

    except Exception as e:
        return jsonify({'error': f'Prediction execution failed: {str(e)}'}), 500

@app.route('/api/upload-model', methods=['POST'])
def upload_model():
    global active_model, custom_model_loaded, custom_model_filename
    
    if 'file' not in request.files:
        return jsonify({'error': 'No model file provided in request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file submitted.'}), 400

    if not (file.filename.endswith('.pkl') or file.filename.endswith('.joblib') or file.filename.endswith('.sav')):
        return jsonify({'error': 'Unsupported file format. Please upload a .pkl, .joblib, or .sav file.'}), 400

    temp_path = os.path.join('scratch_custom_model.pkl')
    try:
        file.save(temp_path)
        loaded_obj = joblib.load(temp_path)
        
        # Validate if object has predict method
        if not hasattr(loaded_obj, 'predict'):
            os.remove(temp_path)
            return jsonify({'error': 'Uploaded object does not contain a valid scikit-learn predict() method.'}), 400

        active_model = loaded_obj
        custom_model_loaded = True
        custom_model_filename = file.filename

        return jsonify({
            'success': True,
            'message': f'Model "{file.filename}" loaded successfully!',
            'model_type': type(loaded_obj).__name__,
            'filename': file.filename
        })

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'Failed to parse uploaded model: {str(e)}'}), 400

@app.route('/api/reset-model', methods=['POST'])
def reset_model():
    load_system_assets()
    return jsonify({
        'success': True,
        'message': 'Reset to system default trained ML model (Logistic Regression / SVM).'
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    if active_metrics:
        return jsonify({
            'success': True,
            'custom_model_active': custom_model_loaded,
            'custom_filename': custom_model_filename,
            'metrics': active_metrics
        })
    else:
        return jsonify({'error': 'Model metrics file not found.'}), 444

@app.route('/api/samples', methods=['GET'])
def get_samples():
    samples = [
        {
            'name': 'Colab Notebook Sample (Diabetic)',
            'description': 'Exact test patient from Google Colab CELL 35 (Pregnancies: 5, Glucose: 166, Age: 51).',
            'values': {
                'Pregnancies': 5,
                'Glucose': 166,
                'BloodPressure': 72,
                'SkinThickness': 19,
                'Insulin': 175,
                'BMI': 25.8,
                'DiabetesPedigreeFunction': 0.587,
                'Age': 51
            }
        },
        {
            'name': 'Low Risk (Healthy Adult)',
            'description': 'Normal glucose, optimal BMI, healthy blood pressure.',
            'values': {
                'Pregnancies': 1,
                'Glucose': 88,
                'BloodPressure': 66,
                'SkinThickness': 18,
                'Insulin': 64,
                'BMI': 22.4,
                'DiabetesPedigreeFunction': 0.185,
                'Age': 26
            }
        },
        {
            'name': 'High Risk (Diabetic Profile)',
            'description': 'Elevated fasting glucose (168 mg/dL), high BMI, high pedigree score.',
            'values': {
                'Pregnancies': 6,
                'Glucose': 168,
                'BloodPressure': 86,
                'SkinThickness': 36,
                'Insulin': 190,
                'BMI': 35.2,
                'DiabetesPedigreeFunction': 0.825,
                'Age': 48
            }
        }
    ]
    return jsonify({'success': True, 'samples': samples})

if __name__ == '__main__':
    print("[*] Starting Diabetes ML Web Application on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
