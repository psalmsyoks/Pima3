import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, PowerTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('Pima.csv')
X = df.drop('diabetes_class', axis=1)
y = df['diabetes_class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define 5 preprocessing techniques
preprocessing_techniques = {
    'Standard Scaling': StandardScaler(),
    'Min-Max Scaling': MinMaxScaler(),
    'Robust Scaling': RobustScaler(),
    'Normalization (L2)': Normalizer(norm='l2'),
    'Power Transform': PowerTransformer(method='yeo-johnson')
}

# Store results
results = []

print("="*80)
print("PREPROCESSING TECHNIQUES COMPARISON - PIMA DIABETES DATASET")
print("="*80)
print("\nTraining models with different preprocessing techniques...\n")

# Train model with each preprocessing technique
for technique_name, scaler in preprocessing_techniques.items():
    print(f"Processing with: {technique_name}...", end=" ", flush=True)
    
    # Apply preprocessing
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, 
                                       max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        'Preprocessing Technique': technique_name,
        'Accuracy': f'{accuracy:.4f}',
        'Accuracy %': f'{accuracy*100:.2f}%',
        'Precision': f'{precision:.4f}',
        'Recall': f'{recall:.4f}',
        'F1 Score': f'{f1:.4f}',
        'ROC-AUC': f'{roc_auc:.4f}',
        'Combined Score': f'{((accuracy + precision + recall + f1 + roc_auc) / 5):.4f}'
    })
    
    print("✓ Done")

# Create DataFrame
results_df = pd.DataFrame(results)

# Display results
print("\n" + "="*80)
print("PREPROCESSING TECHNIQUES COMPARISON TABLE")
print("="*80)
print("\n")
print(results_df.to_string(index=False))

# Find best technique by different metrics
print("\n" + "="*80)
print("SUMMARY & RANKINGS")
print("="*80)

# Convert numeric columns for ranking
numeric_results = results_df.copy()
for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Combined Score']:
    numeric_results[col] = pd.to_numeric(numeric_results[col])

print("\n🏆 Best by Accuracy:")
best_accuracy_idx = numeric_results['Accuracy'].idxmax()
best_accuracy_tech = results_df.loc[best_accuracy_idx, 'Preprocessing Technique']
best_accuracy_val = numeric_results.loc[best_accuracy_idx, 'Accuracy']
print(f"   {best_accuracy_tech}: {best_accuracy_val:.4f}")

print("\n🏆 Best by ROC-AUC Score:")
best_roc_idx = numeric_results['ROC-AUC'].idxmax()
best_roc_tech = results_df.loc[best_roc_idx, 'Preprocessing Technique']
best_roc_val = numeric_results.loc[best_roc_idx, 'ROC-AUC']
print(f"   {best_roc_tech}: {best_roc_val:.4f}")

print("\n🏆 Best Combined Score:")
best_combined_idx = numeric_results['Combined Score'].idxmax()
best_combined_tech = results_df.loc[best_combined_idx, 'Preprocessing Technique']
best_combined_val = numeric_results.loc[best_combined_idx, 'Combined Score']
print(f"   {best_combined_tech}: {best_combined_val:.4f}")

print("\n" + "="*80)
print("COMBINED SCORE = (Accuracy + Precision + Recall + F1 + ROC-AUC) / 5")
print("="*80)

# Save results to CSV
results_df.to_csv('preprocessing_comparison_results.csv', index=False)
print("\n✓ Results saved to 'preprocessing_comparison_results.csv'")

# Create a detailed comparison table
print("\n" + "="*80)
print("DETAILED METRIC COMPARISON")
print("="*80)

detailed_df = results_df[['Preprocessing Technique', 'Accuracy %', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Combined Score']]
print("\n")
print(detailed_df.to_string(index=False))
