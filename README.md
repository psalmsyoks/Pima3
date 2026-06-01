# Pima Indians Diabetes Classification

A comprehensive machine learning project analyzing and predicting diabetes in the Pima Indians population using multiple classification algorithms, preprocessing techniques, and imbalance handling solutions.

## 📊 Project Overview

This project compares and evaluates:
- **10 Classification Algorithms**
- **5 Preprocessing Techniques**
- **3 Class Imbalance Solutions**
- **Multiple Performance Metrics**

## 🎯 Key Results

### Best Performing Models

| Metric | Algorithm | Preprocessing | Imbalance Solution | Score |
|--------|-----------|---------------|--------------------|-------|
| **Accuracy** | AdaBoost | Standard Scaler | No Balancing | 75.97% |
| **F1-Score** | AdaBoost | - | SMOTE | 67.21% |
| **ROC-AUC** | Random Forest | - | RandomOverSampler | 82.98% |
| **Recall** | AdaBoost | - | SMOTE | 75.93% |

## 📁 Project Structure

```
Pima/
├── Pima.csv                           # Original dataset (768 samples, 9 features)
├── README.md                          # This file
│
├── Data Analysis & Preprocessing
├── compare_preprocessing.py            # Compares 5 preprocessing techniques
├── preprocessing_comparison_results.csv
├── preprocessing_accuracy_pivot.csv
├── preprocessing_f1_pivot.csv
│
├── Algorithm Comparison
├── compare_algorithms.py               # Compares 10 classification algorithms
├── algorithms_comparison.csv
│
├── Imbalance Solutions
├── compare_imbalance.py                # Compares SMOTE & RandomOverSampler
├── imbalance_solution_results.csv
├── imbalance_accuracy_pivot.csv
├── imbalance_f1_pivot.csv
├── imbalance_roc_pivot.csv
│
├── Trained Models & Scalers
├── best_classifier_model.pkl          # Best performing model
├── scaler.pkl                         # Feature scaler
├── pima_model.pkl
├── pima_scaler.pkl
│
├── Streamlit Web Applications
├── streamlit_comparison.py             # 10 Algorithms comparison app
├── streamlit_preprocessing.py          # 5 Preprocessing techniques app
├── streamlit_imbalance.py             # Imbalance solutions app
├── streamlit_app.py                   # Main prediction app
│
└── Legacy/Training Scripts
    ├── train_model.py
    ├── train_pima_model.py
    ├── imbalance_solver.py
    └── app.py
```

## 🔬 Preprocessing Techniques Tested

### 1. Standard Scaler
- **Formula:** (X - mean) / std
- **Output Range:** [-∞, +∞]
- **Best For:** Algorithms sensitive to feature scaling (SVM, Neural Networks, KNN)
- **Result:** 75.32% - 75.97% accuracy

### 2. MinMax Scaler
- **Formula:** (X - min) / (max - min)
- **Output Range:** [0, 1]
- **Best For:** Neural Networks, bounded input requirements
- **Result:** 74.03% - 75.97% accuracy

### 3. Robust Scaler
- **Formula:** (X - median) / IQR
- **Output Range:** [-∞, +∞]
- **Best For:** Data with outliers, tree-based models
- **Result:** 74.03% - 75.97% accuracy

### 4. MaxAbs Scaler
- **Formula:** X / max(|X|)
- **Output Range:** [-1, 1]
- **Best For:** Sparse data, negative values preservation
- **Result:** 74.03% - 75.97% accuracy

### 5. Log Transformation
- **Formula:** log(1 + X)
- **Output Range:** [0, +∞]
- **Best For:** Right-skewed data, wide value ranges
- **Result:** 73.38% - 75.97% accuracy

## 🤖 10 Classification Algorithms Tested

| Rank | Algorithm | Accuracy | Precision | Recall | F1-Score |
|------|-----------|----------|-----------|--------|----------|
| 1 | AdaBoost | 75.97% | 68.09% | 59.26% | 63.37% |
| 2 | Random Forest | 75.32% | 68.18% | 55.56% | 61.22% |
| 3 | K-Nearest Neighbors | 75.32% | 66.00% | 61.11% | 63.46% |
| 4 | SVM (RBF) | 74.03% | 65.22% | 55.56% | 60.00% |
| 5 | Gradient Boosting | 73.38% | 63.83% | 55.56% | 59.41% |
| 6 | Neural Network (MLP) | 72.08% | 59.32% | 64.81% | 61.95% |
| 7 | Logistic Regression | 70.78% | 60.00% | 50.00% | 54.55% |
| 8 | SVM (Linear) | 70.13% | 59.09% | 48.15% | 53.06% |
| 9 | Naive Bayes | 70.13% | 56.67% | 62.96% | 59.65% |
| 10 | Decision Tree | 67.53% | 54.55% | 44.44% | 48.98% |

## ⚖️ Class Imbalance Solutions Tested

### Original Dataset
- **Class 0 (No Diabetes):** 400 samples (65.1%)
- **Class 1 (Diabetes):** 214 samples (34.9%)

### 1. No Balancing
- **Training Size:** 614 samples
- **Best Accuracy:** 75.97% (AdaBoost)
- **Best F1-Score:** 63.37% (AdaBoost)
- **Pros:** No synthetic data, preserves original distribution
- **Cons:** Biased towards majority class

### 2. SMOTE (Synthetic Minority Oversampling)
- **Training Size:** 800 samples (50/50)
- **Best Accuracy:** 74.03% (AdaBoost)
- **Best F1-Score:** 67.21% (AdaBoost) ⭐
- **Best Recall:** 75.93% (AdaBoost)
- **Pros:** Intelligent synthetic samples, better recall
- **Cons:** Higher computation, may create overlapping samples

### 3. RandomOverSampler
- **Training Size:** 800 samples (50/50)
- **Best Accuracy:** 75.32% (Random Forest)
- **Best F1-Score:** 65.45% (Random Forest)
- **Best ROC-AUC:** 82.98% (Random Forest) ⭐
- **Pros:** Simple, fast, no information loss
- **Cons:** Creates duplicates, risk of overfitting

## 🌐 Interactive Streamlit Applications

All applications are built with Streamlit for easy visualization and interaction.

### 1. Algorithm Comparison App
```bash
streamlit run streamlit_comparison.py
```
- Compare 10 algorithms across multiple metrics
- Interactive accuracy heatmaps
- Performance analysis and rankings

### 2. Preprocessing Techniques App
```bash
streamlit run streamlit_preprocessing.py
```
- Analyze 5 preprocessing techniques
- Accuracy and F1-Score comparisons
- Detailed technique information and use cases

### 3. Imbalance Solutions App
```bash
streamlit run streamlit_imbalance.py
```
- Compare SMOTE vs RandomOverSampler
- Class distribution visualizations
- ROC-AUC analysis for imbalanced data

### 4. Main Prediction App
```bash
streamlit run streamlit_app.py
```
- Interactive diabetes prediction tool
- Real-time risk assessment
- Model performance metrics

## 📊 Dataset Information

**Pima Indians Diabetes Database**
- **Total Samples:** 768
- **Training Set:** 614 (80%)
- **Test Set:** 154 (20%)
- **Features:** 8 medical metrics
- **Target:** Binary classification (Diabetes: Yes/No)

### Features
1. **Pregnancies** - Number of times pregnant
2. **Glucose** - Plasma glucose concentration (2-hour)
3. **Blood Pressure** - Diastolic blood pressure (mmHg)
4. **Skin Fold Thickness** - Triceps skin fold thickness (mm)
5. **Insulin** - 2-hour serum insulin (μU/mL)
6. **BMI** - Body mass index (kg/m²)
7. **Diabetes Pedigree** - Diabetes pedigree function
8. **Age** - Age (years)

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/psalmsyoks/Pima.git
cd Pima

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Analysis Scripts

```bash
# Compare 10 algorithms
python compare_algorithms.py

# Compare 5 preprocessing techniques
python compare_preprocessing.py

# Compare imbalance solutions
python compare_imbalance.py
```

### Running Streamlit Apps

```bash
# Algorithm comparison
streamlit run streamlit_comparison.py

# Preprocessing techniques
streamlit run streamlit_preprocessing.py

# Imbalance solutions
streamlit run streamlit_imbalance.py

# Main prediction app
streamlit run streamlit_app.py
```

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.2
imbalanced-learn>=0.8.1
streamlit>=1.0.0
plotly>=5.0.0
joblib>=1.0.1
matplotlib>=3.3.0
```

## 💡 Key Insights

1. **AdaBoost** is the most consistent performer across all preprocessing techniques
2. **SMOTE** significantly improves recall (75.93%) for minority class detection
3. **RandomOverSampler** provides the best ROC-AUC (82.98%) with Random Forest
4. **Tree-based models** (Random Forest, Gradient Boosting) are less sensitive to feature scaling
5. **Robust Scaler & Log Transformation** show slight advantage for Random Forest
6. **Standard Scaler** is the safest choice for most algorithms

## ⚠️ Disclaimer

This model is for educational and research purposes only. It should **NOT** be used for medical diagnosis. Always consult with qualified healthcare professionals for medical advice.

## 📈 Performance Metrics Explained

- **Accuracy:** Overall correctness of predictions
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of Precision and Recall (best for imbalanced data)
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve (comprehensive metric)

## 🔗 GitHub Repository

**Repository:** https://github.com/psalmsyoks/Pima.git

**Author:** psalmsyoks  
**Email:** smmuuo@gmail.com

## 📝 License

This project is open source and available for educational purposes.

## 🎓 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Imbalanced-learn Guide](https://imbalanced-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Class Imbalance Handling](https://machinelearningmastery.com/imbalanced-classification-with-the-smote-oversampling-method-in-python/)

---

**Last Updated:** June 1, 2026  
**Project Status:** Complete ✅
