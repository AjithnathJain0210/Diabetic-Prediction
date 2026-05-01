import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from config import TRAIN_DATA_PATH, MODEL_PATH, SCALER_PATH, MODELS_DIR

def evaluate_system():
    print("\n" + "="*60)
    print("📊 MODULE 8: FINAL SYSTEM EVALUATION & METRICS")
    print("="*60)

    # 1. Load Data and Model Artifacts
    if not all(os.path.exists(p) for p in [TRAIN_DATA_PATH, MODEL_PATH, SCALER_PATH]):
        print("❌ Error: Training data or Model files missing. Check your 'models' and 'data/processed' folders.")
        return

    df = pd.read_csv(TRAIN_DATA_PATH)
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 2. Prepare Features and Target
    X = df.drop('risk_level', axis=1)
    y = df['risk_level']

    # 3. Apply Scaling
    X_scaled = scaler.transform(X)

    # 4. Generate Predictions
    y_pred = model.predict(X_scaled)
    acc = accuracy_score(y, y_pred)

    # 5. Print Technical Report
    print(f"\n✅ Overall System Accuracy: {acc*100:.2f}%")
    print("\n📝 Detailed Classification Report:")
    # [Image of a classification report showing precision, recall, and f1-score for a machine learning model]
    print(classification_report(y, y_pred, target_names=['Low Risk', 'Medium Risk', 'High Risk']))

    # 6. Generate and Save Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Medium', 'High'], 
                yticklabels=['Low', 'Medium', 'High'])
    
    plt.title('Final Validation: Diabetic Risk Confusion Matrix')
    plt.ylabel('Actual Medical Status')
    plt.xlabel('System Predicted Status')
    
    # Save the plot for the project documentation
    plot_path = os.path.join(MODELS_DIR, 'final_evaluation_matrix.png')
    plt.savefig(plot_path)
    print(f"\n✓ Confusion Matrix saved at: {plot_path}")
    
    # 7. Feature Importance Verification
    # This proves the multimodal nature (Clinical + Fingerprint)
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 6))
    feat_imp.plot(kind='barh', color='teal')
    plt.title('Multimodal Feature Contribution')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'feature_contribution.png'))
    
    print("✓ Feature Contribution plot saved.")
    print("="*60 + "\n")
    plt.show()

if __name__ == "__main__":
    evaluate_system()