import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def create_sample_custom_models():
    os.makedirs('sample_models', exist_ok=True)
    
    df = pd.read_csv('diabetes.csv')
    X = df.drop(columns='Outcome')
    y = df['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 1. Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_path = os.path.join('sample_models', 'custom_random_forest.pkl')
    joblib.dump(rf_model, rf_path)
    print(f"[+] Created custom test model: {rf_path}")

    # 2. Gradient Boosting Model
    gb_model = GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_path = os.path.join('sample_models', 'custom_gradient_boosting.pkl')
    joblib.dump(gb_model, gb_path)
    print(f"[+] Created custom test model: {gb_path}")

    # 3. Logistic Regression Model
    lr_model = LogisticRegression(C=0.5, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_path = os.path.join('sample_models', 'custom_logistic_regression.pkl')
    joblib.dump(lr_model, lr_path)
    print(f"[+] Created custom test model: {lr_path}")

if __name__ == '__main__':
    create_sample_custom_models()
