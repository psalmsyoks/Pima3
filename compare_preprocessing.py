import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('Pima.csv')

# Prepare the data
X = df.drop('diabetes_class', axis=1)
y = df['diabetes_class']

# Handle missing values (0 values are biologically impossible)
cols_with_zeros = ['glucose_concentration', 'diastolic_bp', 'triceps_skin_fold_thickness', 
                    'two_hr_serum_insulin', 'bmi']

for col in cols_with_zeros:
    median_val = X[X[col] != 0][col].median()
    X.loc[X[col] == 0, col] = median_val

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define preprocessing techniques
preprocessors = {
    'Standard Scaler': StandardScaler(),
    'MinMax Scaler': MinMaxScaler(),
    'Robust Scaler': RobustScaler(),
    'MaxAbs Scaler': MaxAbsScaler(),
    'Log Transformation': None  # Special handling
}

# Define best models (from previous comparison)
models = {
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
}

# Store results
all_results = []

print("="*120)
print("PREPROCESSING TECHNIQUES COMPARISON")
print("="*120)

for prep_name, preprocessor in preprocessors.items():
    print(f"\n{'='*120}")
    print(f"Processing with: {prep_name}")
    print(f"{'='*120}")
    
    # Apply preprocessing
    if prep_name == 'Log Transformation':
        # Add 1 to avoid log(0), then apply log transformation
        X_train_transformed = np.log1p(X_train)
        X_test_transformed = np.log1p(X_test)
    else:
        preprocessor.fit(X_train)
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
    
    # Train and evaluate models with this preprocessing
    for model_name, model in models.items():
        print(f"\n  Training {model_name}...")
        
        # Train the model
        model.fit(X_train_transformed, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_transformed)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        all_results.append({
            'Preprocessing': prep_name,
            'Model': model_name,
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4)
        })
        
        print(f"    Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

# Create results dataframe
results_df = pd.DataFrame(all_results)

print("\n" + "="*120)
print("SUMMARY - ALL RESULTS")
print("="*120)
print(results_df.to_string(index=False))
print("\n")

# Find best combination
best_idx = results_df['Accuracy'].idxmax()
best_row = results_df.loc[best_idx]

print("="*120)
print(f"BEST COMBINATION: {best_row['Preprocessing']} + {best_row['Model']}")
print(f"Accuracy: {best_row['Accuracy']:.4f}")
print("="*120)

# Save results to CSV
results_df.to_csv('preprocessing_comparison_results.csv', index=False)

# Create pivot table for better visualization
pivot_accuracy = results_df.pivot(index='Model', columns='Preprocessing', values='Accuracy')
pivot_f1 = results_df.pivot(index='Model', columns='Preprocessing', values='F1-Score')

print("\n" + "="*120)
print("PIVOT TABLE - ACCURACY BY MODEL AND PREPROCESSING")
print("="*120)
print(pivot_accuracy)

print("\n" + "="*120)
print("PIVOT TABLE - F1-SCORE BY MODEL AND PREPROCESSING")
print("="*120)
print(pivot_f1)

# Save pivot tables
pivot_accuracy.to_csv('preprocessing_accuracy_pivot.csv')
pivot_f1.to_csv('preprocessing_f1_pivot.csv')

print("\n✓ Results saved to CSV files")
