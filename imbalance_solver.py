import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('Pima.csv')

# Separate features and target
X = df.drop('diabetes_class', axis=1)
y = df['diabetes_class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 80)
print("CLASS IMBALANCE ANALYSIS")
print("=" * 80)
print(f"\nOriginal Dataset Distribution:")
print(f"  Class 0 (No Diabetes): {(y == 0).sum()} samples ({(y == 0).sum()/len(y)*100:.2f}%)")
print(f"  Class 1 (Diabetes): {(y == 1).sum()} samples ({(y == 1).sum()/len(y)*100:.2f}%)")
print(f"  Imbalance Ratio: {(y == 0).sum() / (y == 1).sum():.2f}:1")

print(f"\nTraining Set Distribution (Before Resampling):")
print(f"  Class 0: {(y_train == 0).sum()} samples ({(y_train == 0).sum()/len(y_train)*100:.2f}%)")
print(f"  Class 1: {(y_train == 1).sum()} samples ({(y_train == 1).sum()/len(y_train)*100:.2f}%)")

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"\nTraining Set Distribution (After SMOTE):")
print(f"  Class 0: {(y_train_smote == 0).sum()} samples ({(y_train_smote == 0).sum()/len(y_train_smote)*100:.2f}%)")
print(f"  Class 1: {(y_train_smote == 1).sum()} samples ({(y_train_smote == 1).sum()/len(y_train_smote)*100:.2f}%)")

# Apply Random Over Sampler
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train_scaled, y_train)

print(f"\nTraining Set Distribution (After Random Over Sampler):")
print(f"  Class 0: {(y_train_ros == 0).sum()} samples ({(y_train_ros == 0).sum()/len(y_train_ros)*100:.2f}%)")
print(f"  Class 1: {(y_train_ros == 1).sum()} samples ({(y_train_ros == 1).sum()/len(y_train_ros)*100:.2f}%)")

# Train models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
}

results = []

print("\n" + "=" * 80)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 80 + "\n")

for model_name, model in models.items():
    print(f"\n{model_name}:")
    print("-" * 60)
    
    # Original data
    model.fit(X_train_scaled, y_train)
    y_pred_original = model.predict(X_test_scaled)
    
    acc_orig = accuracy_score(y_test, y_pred_original)
    prec_orig = precision_score(y_test, y_pred_original)
    rec_orig = recall_score(y_test, y_pred_original)
    f1_orig = f1_score(y_test, y_pred_original)
    
    print(f"  Original Data:")
    print(f"    Accuracy:  {acc_orig:.4f}")
    print(f"    Precision: {prec_orig:.4f}")
    print(f"    Recall:    {rec_orig:.4f}")
    print(f"    F1-Score:  {f1_orig:.4f}")
    
    # SMOTE
    model.fit(X_train_smote, y_train_smote)
    y_pred_smote = model.predict(X_test_scaled)
    
    acc_smote = accuracy_score(y_test, y_pred_smote)
    prec_smote = precision_score(y_test, y_pred_smote)
    rec_smote = recall_score(y_test, y_pred_smote)
    f1_smote = f1_score(y_test, y_pred_smote)
    
    print(f"  SMOTE:")
    print(f"    Accuracy:  {acc_smote:.4f}")
    print(f"    Precision: {prec_smote:.4f}")
    print(f"    Recall:    {rec_smote:.4f}")
    print(f"    F1-Score:  {f1_smote:.4f}")
    
    # Random Over Sampler
    model.fit(X_train_ros, y_train_ros)
    y_pred_ros = model.predict(X_test_scaled)
    
    acc_ros = accuracy_score(y_test, y_pred_ros)
    prec_ros = precision_score(y_test, y_pred_ros)
    rec_ros = recall_score(y_test, y_pred_ros)
    f1_ros = f1_score(y_test, y_pred_ros)
    
    print(f"  Random Over Sampler:")
    print(f"    Accuracy:  {acc_ros:.4f}")
    print(f"    Precision: {prec_ros:.4f}")
    print(f"    Recall:    {rec_ros:.4f}")
    print(f"    F1-Score:  {f1_ros:.4f}")
    
    results.append({
        'Model': model_name,
        'Original_Accuracy': f"{acc_orig:.4f}",
        'Original_Precision': f"{prec_orig:.4f}",
        'Original_Recall': f"{rec_orig:.4f}",
        'Original_F1': f"{f1_orig:.4f}",
        'SMOTE_Accuracy': f"{acc_smote:.4f}",
        'SMOTE_Precision': f"{prec_smote:.4f}",
        'SMOTE_Recall': f"{rec_smote:.4f}",
        'SMOTE_F1': f"{f1_smote:.4f}",
        'ROS_Accuracy': f"{acc_ros:.4f}",
        'ROS_Precision': f"{prec_ros:.4f}",
        'ROS_Recall': f"{rec_ros:.4f}",
        'ROS_F1': f"{f1_ros:.4f}"
    })

# Create results table
print("\n" + "=" * 80)
print("DETAILED COMPARISON TABLE")
print("=" * 80 + "\n")

results_df = pd.DataFrame(results)

# Display full table
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(results_df.to_string(index=False))

# Save to CSV
results_df.to_csv('imbalance_comparison_results.csv', index=False)
print("\n✓ Results saved to 'imbalance_comparison_results.csv'")

# Create summary table for easier viewing
print("\n" + "=" * 80)
print("ACCURACY COMPARISON SUMMARY")
print("=" * 80 + "\n")

summary_df = pd.DataFrame({
    'Model': [r['Model'] for r in results],
    'Original': [r['Original_Accuracy'] for r in results],
    'SMOTE': [r['SMOTE_Accuracy'] for r in results],
    'Random Over Sampler': [r['ROS_Accuracy'] for r in results]
})

print(summary_df.to_string(index=False))

# Create recall comparison (important for imbalanced datasets)
print("\n" + "=" * 80)
print("RECALL COMPARISON SUMMARY (Important for Imbalanced Data)")
print("=" * 80 + "\n")

recall_df = pd.DataFrame({
    'Model': [r['Model'] for r in results],
    'Original': [r['Original_Recall'] for r in results],
    'SMOTE': [r['SMOTE_Recall'] for r in results],
    'Random Over Sampler': [r['ROS_Recall'] for r in results]
})

print(recall_df.to_string(index=False))

# Create F1-Score comparison
print("\n" + "=" * 80)
print("F1-SCORE COMPARISON SUMMARY (Balanced Metric)")
print("=" * 80 + "\n")

f1_df = pd.DataFrame({
    'Model': [r['Model'] for r in results],
    'Original': [r['Original_F1'] for r in results],
    'SMOTE': [r['SMOTE_F1'] for r in results],
    'Random Over Sampler': [r['ROS_F1'] for r in results]
})

print(f1_df.to_string(index=False))

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print("""
1. SMOTE (Synthetic Minority Oversampling Technique):
   - Creates synthetic samples for the minority class
   - Better for preserving feature relationships
   - Often improves recall and F1-score
   
2. Random Over Sampler:
   - Randomly duplicates minority class samples
   - Simpler but may lead to overfitting
   - Quick to compute
   
3. Metric Guide for Imbalanced Data:
   - Accuracy: Can be misleading with imbalanced data
   - Recall: Important to catch positive class (Diabetes)
   - Precision: Important to avoid false positives
   - F1-Score: Balanced metric combining precision and recall
""")

print("=" * 80)
