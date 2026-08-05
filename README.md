# DiaPredict AI: Diabetes Prediction with Machine Learning

DiaPredict AI is an end-to-end Machine Learning web application designed for interactive diabetes risk prediction, physiological factor analysis, and custom model evaluation. The platform combines a Linear Support Vector Machine (SVM) classifier trained on clinical data with a Flask backend REST API and a modern web dashboard.

---

## Application Demo Showcase

<!-- Replace the path below with your demo GIF or video file path -->
![Application Demo Showcase](![Uploading ezgif.com-video-to-gif-converter.gif…]()
)

*Note: Replace `assets/demo.gif` with the path or URL of your recorded demonstration GIF.*

---

## Key Features

- **Clinical Risk Assessment Engine**: Calculates diabetes risk probability and severity tier (Low, Moderate, High) based on eight medical parameters.
- **Physiological Driver Breakdown**: Identifies specific clinical factors that deviate from standard healthy baselines (such as elevated Fasting Glucose or BMI).
- **Custom Trained Model Uploader**: Allows users to dynamically drag and drop custom serialized machine learning models (`.pkl`, `.joblib`, `.sav`) to perform real-time predictions without restarting the application.
- **Model Analytics Dashboard**: Displays feature importance weights and classifier performance comparison metrics using interactive Chart.js visualizations.
- **Preset Clinical Profiles**: Includes one-click test profile presets for rapid testing, including standard low-risk, high-risk, and Google Colab validation test cases.

---

## Dataset Description

The machine learning model is trained on the Pima Indians Diabetes Dataset, consisting of 768 patient entries with 8 numerical physiological features:

| Parameter | Feature Name | Description | Clinical Normal Range | Unit |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `Pregnancies` | Number of times pregnant | 0 - 5 | Count |
| 2 | `Glucose` | Fasting plasma glucose concentration | 70 - 125 | mg/dL |
| 3 | `BloodPressure` | Diastolic blood pressure | 60 - 80 | mmHg |
| 4 | `SkinThickness` | Triceps skin fold thickness | 10 - 30 | mm |
| 5 | `Insulin` | 2-Hour serum insulin | 15 - 160 | mu U/ml |
| 6 | `BMI` | Body Mass Index | 18.5 - 24.9 | kg/m² |
| 7 | `DiabetesPedigreeFunction` | Diabetes pedigree function (family genetic score) | 0.08 - 0.50 | Score |
| 8 | `Age` | Patient age | 21 - 45 | Years |

---

## Machine Learning Architecture

The underlying predictive pipeline follows standard machine learning best practices:

1. **Preprocessing and Feature Scaling**: Features are standardized using `StandardScaler` to ensure zero mean and unit variance across features.
2. **Train-Test Stratification**: The dataset is split into 80% training data (614 samples) and 20% testing data (154 samples) with stratification on the binary target (`Outcome`).
3. **Classifier**: Support Vector Machine (SVM) with a Linear Kernel (`svm.SVC(kernel='linear')`).
4. **Performance Metrics**:
   - Training Accuracy: 78.66%
   - Testing Accuracy: 77.27%
   - ROC-AUC Score: 0.7920

---

## Technology Stack

- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Backend API Server**: Python 3.10, Flask
- **Frontend Dashboard**: HTML5, Vanilla CSS3 (Glassmorphism design), JavaScript (ES6+), Chart.js
- **Testing**: Python Unittest framework

---

## Project Structure

```
Diabetes Prediction with ML/
├── app.py                      # Primary Flask web server and REST API endpoints
├── main.py                     # Application entry point script
├── train_model.py              # ML pipeline for dataset loading, scaling, and model training
├── test_app.py                 # Automated unit tests for API endpoints
├── generate_test_models.py     # Script to generate sample custom .pkl models for testing
├── diabetes.csv                # Pima Indians Diabetes dataset (768 records)
├── diabetes_model.pkl          # Trained Linear SVM model binary
├── scaler.pkl                  # Fitted StandardScaler binary
├── model_metrics.json          # Serialized model performance metrics and feature importances
├── templates/
│   └── index.html              # Main web interface template
├── static/
│   ├── style.css               # Styling, layout, and glassmorphism design system
│   └── app.js                  # Frontend state management, sliders, gauge, and API calls
├── sample_models/              # Sample custom .pkl models for testing upload functionality
│   ├── custom_random_forest.pkl
│   ├── custom_gradient_boosting.pkl
│   └── custom_logistic_regression.pkl
└── README.md                   # Project documentation
```

---

## Installation and Setup

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Steps

1. Clone or navigate to the project directory:
   ```cmd
   cd "Diabetes Prediction with ML"
   ```

2. Install required Python packages:
   ```cmd
   pip install flask scikit-learn pandas numpy joblib
   ```

---

## Running the Application

### 1. Launch Web Dashboard
To start the Flask development server, execute:
```cmd
python app.py
```
*(Alternative command: `python main.py`)*

Once running, access the web application by opening your browser at:
`http://127.0.0.1:5000`

### 2. Train or Retrain the Model
To re-run the training pipeline and generate updated model binaries:
```cmd
python train_model.py
```

### 3. Generate Sample Upload Models
To generate additional sample `.pkl` model files for testing the upload feature:
```cmd
python generate_test_models.py
```

---

## API Reference

### `POST /api/predict`
Calculates diabetes risk probability and assessment for a set of input parameters.

- **Request Body (JSON)**:
  ```json
  {
    "Pregnancies": 5,
    "Glucose": 166,
    "BloodPressure": 72,
    "SkinThickness": 19,
    "Insulin": 175,
    "BMI": 25.8,
    "DiabetesPedigreeFunction": 0.587,
    "Age": 51
  }
  ```

- **Response (JSON)**:
  ```json
  {
    "success": true,
    "prediction": 1,
    "label": "Diabetic",
    "risk_percentage": 78.4,
    "risk_severity": "High",
    "color_code": "#ef4444",
    "assessment": "High risk of diabetes. Clinical evaluation recommended.",
    "model_used": "Linear SVM (Colab Model)"
  }
  ```

### `POST /api/upload-model`
Accepts a multipart file upload (`.pkl`, `.joblib`, `.sav`) and dynamically updates the active model in memory.

### `GET /api/metrics`
Returns current model training metrics, accuracy scores, and feature importance weights.

### `GET /api/samples`
Returns pre-configured clinical test patient profiles.

---

## Testing and Verification

Run the automated test suite to verify that all API endpoints and prediction calculations execute correctly:

```cmd
python test_app.py
```

Expected Output:
```
Ran 5 tests in 0.039s

OK
```

---

## License

This project is open-source and available under the MIT License.
