import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
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

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define 10 different classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    'SVM (Linear)': SVC(kernel='linear', random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Neural Network (MLP)': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

# Train and evaluate all classifiers
results = []

print("="*100)
print("CLASSIFICATION ALGORITHMS COMPARISON")
print("="*100)

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    results.append({
        'Algorithm': name,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4),
        'CV Mean': round(cv_mean, 4),
        'CV Std': round(cv_std, 4),
        'Model': clf
    })
    
    print(f"  Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"  Cross-Val Mean: {cv_mean:.4f} ± {cv_std:.4f}")

# Create results dataframe
results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'Model'} for r in results])

# Sort by accuracy
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

print("\n" + "="*100)
print("SUMMARY - ALGORITHMS RANKED BY ACCURACY")
print("="*100)
print(results_df.to_string(index=False))
print("\n")

# Save results to CSV
results_df.to_csv('algorithms_comparison.csv', index=False)

# Save the best model
best_model_idx = results_df.index[0]
best_model_name = results_df.loc[best_model_idx, 'Algorithm']
best_model = None

for r in results:
    if r['Algorithm'] == best_model_name:
        best_model = r['Model']
        break

joblib.dump(best_model, 'best_classifier_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {results_df.loc[best_model_idx, 'Accuracy']:.4f}")
print("\nResults saved to 'algorithms_comparison.csv'")
print("Best model saved to 'best_classifier_model.pkl'")
