import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns


# 1. Paths
AUGMENTED_PATH = r"C:\Users\ajith\Desktop\Diabetic Predictor\data\processed\augmented_data.csv"
MODEL_DIR = r"C:\Users\ajith\Desktop\Diabetic Predictor\models"
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

def train_final_model():
    print("\n" + "="*60)
    print("MODULE 6: Model Training Random Forest")
    print("="*60)

    if not os.path.exists(AUGMENTED_PATH):
        print(f"❌ Error: {AUGMENTED_PATH} not found!")
        return
    
    # Load and Shuffle
    df = pd.read_csv(AUGMENTED_PATH)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = df.drop('risk_level', axis=1)
    y = df['risk_level']

    # 2. Train/Test Split (30% test is harder than 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=25, stratify=y
    )

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)


    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=3,            
        min_samples_leaf=12,   
        max_features=0.3,       
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, MODEL_PATH)

    # 5. Evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ FINAL ACCURACY: {acc*100:.2f}%")
    print("\n📊 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred))

    

    # 6. Save Importance Plot
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    feat_imp.plot(kind='barh', color='navy')
    plt.title(' Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'final_importance.png'))
    plt.show()

    # 7.Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # 8. Plot Confusion Matrix

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Medium', 'High'], 
                yticklabels=['Low', 'Medium', 'High'])
    plt.title('Confusion Matrix: Diabetic Risk Prediction')
    plt.ylabel('Actual Risk Level')
    plt.xlabel('Predicted Risk Level')
    plt.tight_layout()
    
    # Save the matrix for your presentation
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    print(f"✓ Confusion Matrix saved to {MODEL_DIR}")
    plt.show()

    # Add this to the end of your Module 6 Training Script:
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "features.pkl"))
    print("✓ Feature list saved for Predictor alignment.")

if __name__ == "__main__":
    train_final_model()

