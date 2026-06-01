import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('Pima.csv')

# Prepare the data
X = df.drop('diabetes_class', axis=1)
y = df['diabetes_class']

# Handle missing values
cols_with_zeros = ['glucose_concentration', 'diastolic_bp', 'triceps_skin_fold_thickness', 
                    'two_hr_serum_insulin', 'bmi']

for col in cols_with_zeros:
    median_val = X[X[col] != 0][col].median()
    X.loc[X[col] == 0, col] = median_val

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define imbalance handling techniques
techniques = {
    'No Balancing': (X_train_scaled, y_train),
    'SMOTE': None,  # Will be computed
    'RandomOverSampler': None  # Will be computed
}

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
techniques['SMOTE'] = (X_train_smote, y_train_smote)

# Apply RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train_scaled, y_train)
techniques['RandomOverSampler'] = (X_train_ros, y_train_ros)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# Store results
all_results = []

print("="*130)
print("CLASS IMBALANCE HANDLING TECHNIQUES COMPARISON")
print("="*130)
print(f"\nOriginal Dataset - Class Distribution:")
print(f"  Class 0 (No Diabetes): {sum(y_train == 0)} samples ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
print(f"  Class 1 (Diabetes): {sum(y_train == 1)} samples ({sum(y_train == 1)/len(y_train)*100:.1f}%)")

print(f"\nAfter SMOTE - Class Distribution:")
print(f"  Class 0 (No Diabetes): {sum(y_train_smote == 0)} samples ({sum(y_train_smote == 0)/len(y_train_smote)*100:.1f}%)")
print(f"  Class 1 (Diabetes): {sum(y_train_smote == 1)} samples ({sum(y_train_smote == 1)/len(y_train_smote)*100:.1f}%)")

print(f"\nAfter RandomOverSampler - Class Distribution:")
print(f"  Class 0 (No Diabetes): {sum(y_train_ros == 0)} samples ({sum(y_train_ros == 0)/len(y_train_ros)*100:.1f}%)")
print(f"  Class 1 (Diabetes): {sum(y_train_ros == 1)} samples ({sum(y_train_ros == 1)/len(y_train_ros)*100:.1f}%)")

print("\n" + "="*130)
print("MODEL TRAINING AND EVALUATION")
print("="*130)

for tech_name, (X_train_processed, y_train_processed) in techniques.items():
    print(f"\n{'='*130}")
    print(f"Technique: {tech_name}")
    print(f"Training set size: {len(y_train_processed)} samples")
    print(f"{'='*130}")
    
    for model_name, model in models.items():
        print(f"\n  Training {model_name}...")
        
        # Train the model
        model.fit(X_train_processed, y_train_processed)
        
        # Make predictions on test set (original imbalanced test set)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        except:
            try:
                roc_auc = roc_auc_score(y_test, model.decision_function(X_test_scaled))
            except:
                roc_auc = 0.0
        
        all_results.append({
            'Technique': tech_name,
            'Model': model_name,
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(roc_auc, 4)
        })
        
        print(f"    Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

# Create results dataframe
results_df = pd.DataFrame(all_results)

print("\n" + "="*130)
print("SUMMARY - ALL RESULTS")
print("="*130)
print(results_df.to_string(index=False))

# Find best combinations
print("\n" + "="*130)
print("TOP 5 BEST COMBINATIONS (BY ACCURACY)")
print("="*130)
top_accuracy = results_df.nlargest(5, 'Accuracy')[['Technique', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]
print(top_accuracy.to_string(index=False))

print("\n" + "="*130)
print("TOP 5 BEST COMBINATIONS (BY F1-SCORE)")
print("="*130)
top_f1 = results_df.nlargest(5, 'F1-Score')[['Technique', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]
print(top_f1.to_string(index=False))

print("\n" + "="*130)
print("TOP 5 BEST COMBINATIONS (BY ROC-AUC)")
print("="*130)
top_roc = results_df.nlargest(5, 'ROC-AUC')[['Technique', 'Model', 'Accuracy', 'Recall', 'Precision', 'ROC-AUC']]
print(top_roc.to_string(index=False))

# Pivot tables
pivot_accuracy = results_df.pivot(index='Model', columns='Technique', values='Accuracy')
pivot_f1 = results_df.pivot(index='Model', columns='Technique', values='F1-Score')
pivot_roc = results_df.pivot(index='Model', columns='Technique', values='ROC-AUC')

print("\n" + "="*130)
print("PIVOT TABLE - ACCURACY BY MODEL AND TECHNIQUE")
print("="*130)
print(pivot_accuracy)

print("\n" + "="*130)
print("PIVOT TABLE - F1-SCORE BY MODEL AND TECHNIQUE")
print("="*130)
print(pivot_f1)

print("\n" + "="*130)
print("PIVOT TABLE - ROC-AUC BY MODEL AND TECHNIQUE")
print("="*130)
print(pivot_roc)

# Save results to CSV
results_df.to_csv('imbalance_solution_results.csv', index=False)
pivot_accuracy.to_csv('imbalance_accuracy_pivot.csv')
pivot_f1.to_csv('imbalance_f1_pivot.csv')
pivot_roc.to_csv('imbalance_roc_pivot.csv')

print("\n✓ Results saved to CSV files")

# Save best model
best_idx = results_df['F1-Score'].idxmax()
best_row = results_df.loc[best_idx]
best_technique = best_row['Technique']
best_model_name = best_row['Model']

print(f"\n✓ Best Model Configuration:")
print(f"  Technique: {best_technique}")
print(f"  Model: {best_model_name}")
print(f"  F1-Score: {best_row['F1-Score']:.4f}")
