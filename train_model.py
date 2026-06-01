import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('Pima.csv')

print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())
print("\nTarget distribution:")
print(df['diabetes_class'].value_counts())

# Prepare features and target
X = df.drop('diabetes_class', axis=1)
y = df['diabetes_class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models and use ensemble
# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, 
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                                       min_samples_split=5, min_samples_leaf=2, random_state=42)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, gb_pred)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")
print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")

# Use the better model
if gb_accuracy >= rf_accuracy:
    best_model = gb_model
    best_pred = gb_pred
    best_name = "Gradient Boosting"
    best_accuracy = gb_accuracy
else:
    best_model = rf_model
    best_pred = rf_pred
    best_name = "Random Forest"
    best_accuracy = rf_accuracy

print(f"\nBest Model: {best_name}")
print(f"Best Accuracy: {best_accuracy:.4f}")
print(f"Precision: {precision_score(y_test, best_pred):.4f}")
print(f"Recall: {recall_score(y_test, best_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, best_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1]):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, best_pred))

# Save models and scaler
joblib.dump(best_model, 'pima_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("\n" + "="*60)
print("Model and scaler saved successfully!")
print("="*60)
