import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_exact_colab_model():
    print("="*65)
    print("  TRAINING EXACT DIABETES MODEL FROM USER'S GOOGLE COLAB  ")
    print("="*65)
    
    csv_path = "diabetes.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset {csv_path} not found.")

    # 1. Load exact Pima Indians Diabetes Dataset (768 rows, 9 columns)
    diabetes_dataset = pd.read_csv(csv_path)
    print(f"[*] Loaded dataset: {diabetes_dataset.shape[0]} rows, {diabetes_dataset.shape[1]} columns")

    # 2. Separate Data and Labels (as done in Colab CELL 12)
    X = diabetes_dataset.drop(columns='Outcome', axis=1)
    Y = diabetes_dataset['Outcome']

    # 3. Data Imputation & Standardization with StandardScaler (Colab CELL 16-18)
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    # Fit imputer on raw features with 0 replaced by NaN for zero-invalid physiological features
    zero_fields = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    X_imp = X.copy()
    for col in zero_fields:
        X_imp[col] = X_imp[col].replace(0, np.nan)
    
    imputer.fit(X_imp)
    X_imputed = imputer.transform(X_imp)

    scaler = StandardScaler()
    scaler.fit(X_imputed)
    standardized_data = scaler.transform(X_imputed)
    X = standardized_data

    # 4. Train Test Split (Colab CELL 23: test_size=0.2, stratify=Y, random_state=2)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    print(f"[*] Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # 5. Support Vector Machine Classifier with Linear Kernel (Colab CELL 26)
    # Note: enable probability=True to allow smooth risk percentage calculation in UI
    classifier = SVC(kernel='linear', probability=True, random_state=2)
    classifier.fit(X_train, Y_train)
    print("[+] Trained SVM model with linear kernel.")

    # 6. Evaluate Model Accuracy (Colab CELL 30-33)
    train_pred = classifier.predict(X_train)
    train_acc = accuracy_score(Y_train, train_pred)

    test_pred = classifier.predict(X_test)
    test_acc = accuracy_score(Y_test, test_pred)
    
    y_proba = classifier.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(Y_test, y_proba)
    test_prec = precision_score(Y_test, test_pred)
    test_rec = recall_score(Y_test, test_pred)
    test_f1 = f1_score(Y_test, test_pred)

    print(f"\n[*] Colab Training Data Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"[*] Colab Test Data Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"[*] ROC-AUC Score:                {test_auc:.4f}")

    # 7. Test sample from Colab (CELL 35): (5, 166, 72, 19, 175, 25.8, 0.587, 51)
    colab_sample = np.asarray((5, 166, 72, 19, 175, 25.8, 0.587, 51)).reshape(1, -1)
    std_colab_sample = scaler.transform(imputer.transform(colab_sample))
    colab_pred = classifier.predict(std_colab_sample)[0]
    print(f"\n[*] Testing Colab sample (5, 166, 72, 19, 175, 25.8, 0.587, 51):")
    print(f"    Prediction: {colab_pred} -> {'Diabetic' if colab_pred == 1 else 'Non-Diabetic'}")

    # 8. Save Trained Model, Scaler, and Imputer
    joblib.dump(classifier, 'diabetes_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(imputer, 'imputer.pkl')
    print("\n[+] Saved 'diabetes_model.pkl', 'scaler.pkl', and 'imputer.pkl'")

    # Save metadata
    feature_names = list(diabetes_dataset.columns[:-1])
    # Extract linear SVM coefficients as feature importances
    coefs = np.abs(classifier.coef_[0])
    feat_imp = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)

    metadata = {
        'best_model': 'Linear SVM (Colab Model)',
        'metrics': {
            'accuracy': float(test_acc),
            'train_accuracy': float(train_acc),
            'precision': float(test_prec),
            'recall': float(test_rec),
            'f1': float(test_f1),
            'roc_auc': float(test_auc),
            'confusion_matrix': confusion_matrix(Y_test, test_pred).tolist()
        },
        'all_models': {
            'Linear SVM (Colab)': {
                'accuracy': float(test_acc),
                'roc_auc': float(test_auc)
            }
        },
        'features': feature_names,
        'feature_importances': [{'feature': f, 'importance': round(float(imp), 4)} for f, imp in feat_imp]
    }

    with open('model_metrics.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("[+] Saved metadata to 'model_metrics.json'")
    print("="*65)
    print("Colab model integration complete successfully!")

if __name__ == '__main__':
    train_exact_colab_model()
