import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('Pima.csv')

print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Prepare the data
X = df.drop('diabetes_class', axis=1)
y = df['diabetes_class']

# Handle missing values (0 values in some columns are actually missing)
# Replace 0 with median for columns where 0 is biologically impossible
cols_with_zeros = ['glucose_concentration', 'diastolic_bp', 'triceps_skin_fold_thickness', 
                    'two_hr_serum_insulin', 'bmi']

for col in cols_with_zeros:
    # Calculate median before replacing 0s
    median_val = X[X[col] != 0][col].median()
    # Replace 0 with median
    X.loc[X[col] == 0, col] = median_val

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models and select the best
models = {
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
}

best_model = None
best_accuracy = 0
best_model_name = ""

print("\n" + "="*60)
print("MODEL TRAINING RESULTS")
print("="*60)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print("\n" + "="*60)
print(f"BEST MODEL: {best_model_name}")
print(f"Test Accuracy: {best_accuracy:.4f}")
print("="*60)

# Save the model and scaler
joblib.dump(best_model, 'pima_model.pkl')
joblib.dump(scaler, 'pima_scaler.pkl')

print("\nModel and scaler saved successfully!")
