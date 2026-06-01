import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Set page config
st.set_page_config(page_title="Pima Diabetes Predictor", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .header {
        font-size: 2.5rem;
        color: #0066cc;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .accuracy-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_models():
    model = joblib.load('pima_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

model, scaler, feature_names = load_models()

# Title and description
st.markdown('<div class="header">🩺 Pima Diabetes Prediction Model</div>', unsafe_allow_html=True)

# Display model info
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="accuracy-box">
    <h3 style="color: #0066cc;">Model Accuracy</h3>
    <h2 style="color: #00aa00;">75.97%</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="accuracy-box">
    <h3 style="color: #0066cc;">ROC-AUC Score</h3>
    <h2 style="color: #00aa00;">82.81%</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="accuracy-box">
    <h3 style="color: #0066cc;">F1 Score</h3>
    <h2 style="color: #00aa00;">64.76%</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Make Prediction", "📊 Model Info", "⚙️ Preprocessing Comparison", "❓ About"])

with tab1:
    st.subheader("Enter Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        preg_count = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Concentration (mg/dL)", min_value=0, max_value=300, value=120)
        diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0, max_value=200, value=70)
        triceps = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0, max_value=900, value=85)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=60.0, value=25.0)
        diabetes_pedi = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=33)
    
    # Make prediction
    if st.button("🔍 Predict Diabetes Risk", key="predict", use_container_width=True):
        # Prepare input data
        input_data = np.array([[preg_count, glucose, diastolic_bp, triceps, 
                               insulin, bmi, diabetes_pedi, age]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📋 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.markdown("""
                <div style="background-color: #ffcccc; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #cc0000;">
                <h3 style="color: #cc0000;">⚠️ High Risk</h3>
                <p style="font-size: 1.2rem; color: #333;">Diabetes Positive</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #ccffcc; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #00aa00;">
                <h3 style="color: #00aa00;">✅ Low Risk</h3>
                <p style="font-size: 1.2rem; color: #333;">Diabetes Negative</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
            <h4>Confidence Scores</h4>
            <p><strong>Negative (No Diabetes):</strong> {probability[0]:.2%}</p>
            <p><strong>Positive (Diabetes):</strong> {probability[1]:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display confidence gauge
        st.subheader("Risk Level")
        risk_percentage = probability[1] * 100
        st.progress(probability[1], text=f"Diabetes Risk: {risk_percentage:.1f}%")

with tab2:
    st.subheader("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
        <h4>Model Type</h4>
        <p>Gradient Boosting Classifier</p>
        </div>
        
        <div class="metric-box">
        <h4>Accuracy</h4>
        <p>75.97%</p>
        </div>
        
        <div class="metric-box">
        <h4>Precision</h4>
        <p>66.67%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
        <h4>Recall</h4>
        <p>62.96%</p>
        </div>
        
        <div class="metric-box">
        <h4>F1 Score</h4>
        <p>64.76%</p>
        </div>
        
        <div class="metric-box">
        <h4>ROC-AUC</h4>
        <p>82.81%</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Features Used")
    st.markdown("""
    1. **Pregnancies**: Number of times pregnant
    2. **Glucose Concentration**: Plasma glucose concentration in mg/dL
    3. **Diastolic Blood Pressure**: Diastolic blood pressure in mmHg
    4. **Triceps Skin Fold Thickness**: Triceps skin fold thickness in mm
    5. **2-Hour Serum Insulin**: 2-hour serum insulin level in mu U/ml
    6. **BMI**: Body Mass Index
    7. **Diabetes Pedigree Function**: Diabetes pedigree function score
    8. **Age**: Age in years
    """)

with tab3:
    st.subheader("Preprocessing Techniques Comparison")
    
    # Load preprocessing results
    try:
        prep_results = pd.read_csv('preprocessing_comparison_results.csv')
        
        # Display summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
            <h4>Best Technique</h4>
            <p>Standard Scaling</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
            <h4>Best Accuracy</h4>
            <p>75.32%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-box">
            <h4>Best ROC-AUC</h4>
            <p>0.8198</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-box">
            <h4>Best Combined</h4>
            <p>0.6882</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown("""
            <div class="metric-box">
            <h4>Techniques Used</h4>
            <p>5 Methods</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display main comparison table
        st.subheader("📋 Full Comparison Table")
        
        # Create display dataframe with better formatting
        display_prep = prep_results.copy()
        display_prep = display_prep[['Preprocessing Technique', 'Accuracy %', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Combined Score']]
        
        st.dataframe(display_prep, use_container_width=True)
        
        st.markdown("---")
        
        # Create visualization tabs
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.subheader("📈 Accuracy Comparison")
            # Convert to numeric for plotting
            prep_numeric = prep_results.copy()
            prep_numeric['Accuracy'] = pd.to_numeric(prep_results['Accuracy'])
            
            chart_data = prep_numeric.set_index('Preprocessing Technique')[['Accuracy']]
            st.bar_chart(chart_data, height=400)
        
        with viz_col2:
            st.subheader("🎯 Combined Score Comparison")
            prep_numeric = prep_results.copy()
            prep_numeric['Combined Score'] = pd.to_numeric(prep_results['Combined Score'])
            
            chart_data = prep_numeric.set_index('Preprocessing Technique')[['Combined Score']]
            st.bar_chart(chart_data, height=400)
        
        st.markdown("---")
        
        # Detailed metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Precision vs Recall")
            metrics_data = prep_results.copy()
            metrics_data['Precision'] = pd.to_numeric(metrics_data['Precision'])
            metrics_data['Recall'] = pd.to_numeric(metrics_data['Recall'])
            metrics_chart = metrics_data.set_index('Preprocessing Technique')[['Precision', 'Recall']]
            st.bar_chart(metrics_chart, height=400)
        
        with col2:
            st.subheader("🔍 ROC-AUC Score Comparison")
            prep_numeric = prep_results.copy()
            prep_numeric['ROC-AUC'] = pd.to_numeric(prep_results['ROC-AUC'])
            
            chart_data = prep_numeric.set_index('Preprocessing Technique')[['ROC-AUC']]
            st.bar_chart(chart_data, height=400)
        
        st.markdown("---")
        
        # Preprocessing techniques explanation
        st.subheader("ℹ️ About Preprocessing Techniques")
        
        techniques_info = {
            "🔹 Standard Scaling": "Transforms data to have mean=0 and std=1. Best for normally distributed data. **Winner** ✅",
            "🔹 Min-Max Scaling": "Scales data to a fixed range [0,1]. Good for bounded data.",
            "🔹 Robust Scaling": "Uses median and IQR, resistant to outliers. Good for data with outliers.",
            "🔹 Normalization (L2)": "Converts each sample to unit norm. Good for sparse data and text.",
            "🔹 Power Transform": "Applies mathematical transformations to make data more Gaussian-like."
        }
        
        for technique, description in techniques_info.items():
            st.markdown(f"**{technique}**")
            st.write(description)
            st.write("")
        
        st.markdown("---")
        
        st.markdown("""
        ### Combined Score Calculation
        
        The **Combined Score** is calculated as the average of all performance metrics:
        
        $$\\text{Combined Score} = \\frac{\\text{Accuracy} + \\text{Precision} + \\text{Recall} + \\text{F1} + \\text{ROC-AUC}}{5}$$
        
        **Key Findings:**
        - **Standard Scaling** performs best with a combined score of **0.6882**
        - Most scaling techniques (Standard, Min-Max, Robust, Power) perform similarly
        - **Normalization (L2)** underperforms for this dataset (0.5821)
        - The difference between best and second-best is minimal (0.0001)
        """)
        
    except FileNotFoundError:
        st.warning("⚠️ Preprocessing comparison results not found. Please run preprocessing_comparison.py first.")

with tab4:
    st.subheader("About This Application")
    st.markdown("""
    ### Pima Diabetes Prediction Model
    
    This web application uses a machine learning model to predict the likelihood of diabetes 
    based on diagnostic measurements from the Pima Indians Diabetes Database.
    
    **Dataset Information:**
    - **Source**: Pima Indians Diabetes Database
    - **Samples**: 768 individuals
    - **Features**: 8 medical/physiological measurements
    - **Target**: Diabetes presence (0: Negative, 1: Positive)
    
    **Model Performance:**
    - **Test Set Accuracy**: 75.97%
    - **ROC-AUC Score**: 82.81% (Excellent discrimination ability)
    - **F1 Score**: 64.76%
    
    ### How to Use:
    1. Enter the patient's medical measurements in the prediction tab
    2. Click the "Predict Diabetes Risk" button
    3. View the prediction result and confidence scores
    
    ### Important Disclaimer:
    ⚠️ **This model is for educational and demonstration purposes only.**
    
    This prediction model should NOT be used for actual medical diagnosis or treatment decisions.
    Please consult with qualified healthcare professionals for medical advice.
    
    ### Features Explanation:
    - **Pregnancies**: Number of times pregnant
    - **Glucose**: Fasting blood glucose concentration
    - **Blood Pressure**: Diastolic blood pressure measurement
    - **Skin Thickness**: Triceps skin fold measurement
    - **Insulin**: 2-hour serum insulin level
    - **BMI**: Body Mass Index (weight/height²)
    - **Pedigree Function**: Genetic predisposition score
    - **Age**: Patient age in years
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
<p><strong>Pima Diabetes Prediction Model v1.0</strong></p>
<p>Built with Streamlit and Gradient Boosting | Model Accuracy: 75.97%</p>
</div>
""", unsafe_allow_html=True)
