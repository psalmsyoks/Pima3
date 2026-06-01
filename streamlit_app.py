import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Pima Diabetes Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('pima_model.pkl')
    scaler = joblib.load('pima_scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

# Page title
st.title("🏥 Pima Diabetes Classification Model")
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Model Prediction", "📈 Model Performance", "ℹ️ About"])

# TAB 1: Prediction
with tab1:
    st.header("Diabetes Risk Prediction")
    st.write("Enter the patient's health metrics to predict diabetes risk")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        preg_count = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Concentration (mg/dL)", min_value=0, max_value=300, value=120)
        diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0, max_value=150, value=70)
    
    with col2:
        triceps = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=0, max_value=100, value=30)
        insulin = st.number_input("2-Hour Serum Insulin (μU/mL)", min_value=0, max_value=900, value=100)
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0)
    
    with col3:
        diabetes_pedi = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
    
    # Create prediction button
    if st.button("🔍 Predict", use_container_width=True):
        # Prepare input data
        input_data = np.array([[preg_count, glucose, diastolic_bp, triceps, insulin, bmi, diabetes_pedi, age]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ **HIGH RISK**: Diabetes Likely Present")
                risk_level = "High"
                risk_percentage = probability[1] * 100
            else:
                st.success("✅ **LOW RISK**: Diabetes Unlikely")
                risk_level = "Low"
                risk_percentage = probability[0] * 100
        
        with col2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_percentage,
                title={'text': "Diabetes Risk Score"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ff4444" if risk_percentage > 50 else "#44aa44"},
                    'steps': [
                        {'range': [0, 25], 'color': "#e8f4f8"},
                        {'range': [25, 50], 'color': "#b3d9e6"},
                        {'range': [50, 75], 'color': "#ffd9b3"},
                        {'range': [75, 100], 'color': "#ffb3b3"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Prediction Confidence")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk of Diabetes", f"{risk_percentage:.1f}%")
        with col2:
            st.metric("No Diabetes Risk", f"{(100-risk_percentage):.1f}%")
        
        # Probability bar chart
        fig = go.Figure(data=[
            go.Bar(x=['No Diabetes', 'Diabetes'], y=[probability[0]*100, probability[1]*100],
                   marker_color=['#44aa44', '#ff4444'])
        ])
        fig.update_layout(
            title="Prediction Probability Distribution",
            yaxis_title="Probability (%)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: Model Performance
with tab2:
    st.header("Model Performance Metrics")
    
    # Load test data for evaluation
    df = pd.read_csv('Pima.csv')
    X = df.drop('diabetes_class', axis=1)
    y = df['diabetes_class']
    
    # Handle missing values
    cols_with_zeros = ['glucose_concentration', 'diastolic_bp', 'triceps_skin_fold_thickness', 
                        'two_hr_serum_insulin', 'bmi']
    for col in cols_with_zeros:
        # Calculate median before replacing 0s
        median_val = X[X[col] != 0][col].median()
        # Replace 0 with median
        X.loc[X[col] == 0, col] = median_val
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Accuracy", f"{accuracy:.2%}", delta=f"{accuracy*100:.1f}%")
    with col2:
        st.metric("Total Samples", len(y))
    with col3:
        st.metric("Diabetes Cases", f"{sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    
    st.markdown("---")
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No Diabetes', 'Predicted Diabetes'],
        y=['Actual No Diabetes', 'Actual Diabetes'],
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        colorscale='Blues'
    ))
    fig.update_layout(title="Confusion Matrix", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

# TAB 3: About
with tab3:
    st.header("About This Model")
    
    st.subheader("Dataset: Pima Indians Diabetes Database")
    st.write("""
    This model is trained on the Pima Indians Diabetes Database, which contains:
    - **768 samples** from female patients of Pima Indian heritage
    - **8 medical features** used for prediction
    - **Binary outcome**: Presence or absence of diabetes
    """)
    
    st.subheader("Features Used")
    features_info = {
        "Pregnancies": "Number of times pregnant",
        "Glucose": "Plasma glucose concentration (2-hour)",
        "Blood Pressure": "Diastolic blood pressure (mmHg)",
        "Skin Fold Thickness": "Triceps skin fold thickness (mm)",
        "Insulin": "2-hour serum insulin (μU/mL)",
        "BMI": "Body mass index (kg/m²)",
        "Diabetes Pedigree": "Diabetes pedigree function",
        "Age": "Age (years)"
    }
    
    for feature, description in features_info.items():
        st.write(f"• **{feature}**: {description}")
    
    st.subheader("Model Details")
    st.write(f"""
    - **Algorithm**: Gradient Boosting Classifier
    - **Test Accuracy**: {accuracy:.2%}
    - **Training Method**: 80-20 train-test split with stratification
    - **Feature Scaling**: StandardScaler normalization
    - **Data Preprocessing**: 
        - Missing value handling (replacing biologically impossible zeros)
        - Feature normalization
        - Class balancing with stratified split
    """)
    
    st.subheader("Disclaimer")
    st.warning("""
    ⚠️ **Important**: This model is for educational and research purposes only. 
    It should NOT be used for medical diagnosis or treatment decisions. 
    Always consult with a qualified healthcare professional for medical advice.
    """)
    
    st.subheader("Data Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature correlation with target
        correlations = df.corr()['diabetes_class'].drop('diabetes_class').sort_values(ascending=False)
        fig = go.Figure(data=[go.Bar(y=correlations.index, x=correlations.values, orientation='h')])
        fig.update_layout(title="Feature Correlation with Diabetes", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Class distribution
        class_dist = y.value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=['No Diabetes', 'Diabetes'],
            values=[class_dist[0], class_dist[1]],
            marker=dict(colors=['#44aa44', '#ff4444'])
        )])
        fig.update_layout(title="Class Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Built with Streamlit | Machine Learning Model</div>", unsafe_allow_html=True)
