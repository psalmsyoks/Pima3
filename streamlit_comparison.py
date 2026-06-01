import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Pima Diabetes - 10 Algorithms Comparison",
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
    .best-model {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('best_classifier_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@st.cache_data
def load_comparison_results():
    return pd.read_csv('algorithms_comparison.csv')

model, scaler = load_model_and_scaler()
comparison_df = load_comparison_results()

# Page title
st.title("🏥 Pima Diabetes Classification - 10 Algorithms Comparison")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Algorithm Comparison", "🔮 Best Model Prediction", "📈 Performance Analysis", "ℹ️ About"])

# TAB 1: Algorithm Comparison
with tab1:
    st.header("Algorithm Performance Comparison")
    st.write("Comparison of 10 different classification algorithms trained on the Pima dataset")
    
    # Display comparison table
    st.subheader("Accuracy Scores Table")
    
    # Create a styled dataframe for display
    display_df = comparison_df.copy()
    display_df['Rank'] = range(1, len(display_df) + 1)
    display_df = display_df[['Rank', 'Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Mean', 'CV Std']]
    
    # Display the table
    st.dataframe(display_df, use_container_width=True)
    
    st.markdown("---")
    
    # Best model highlight
    best_model_row = comparison_df.iloc[0]
    st.markdown(f"""
    <div class='best-model'>
        <h3>🏆 Best Performing Model</h3>
        <h2>{best_model_row['Algorithm']}</h2>
        <p><b>Accuracy:</b> {best_model_row['Accuracy']:.2%}</p>
        <p><b>Precision:</b> {best_model_row['Precision']:.4f} | <b>Recall:</b> {best_model_row['Recall']:.4f} | <b>F1-Score:</b> {best_model_row['F1-Score']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualization 1: Accuracy comparison bar chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(
                y=comparison_df['Algorithm'],
                x=comparison_df['Accuracy'],
                orientation='h',
                marker=dict(
                    color=comparison_df['Accuracy'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Accuracy")
                ),
                text=comparison_df['Accuracy'].round(4),
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Algorithm Accuracy Comparison",
            xaxis_title="Accuracy",
            yaxis_title="Algorithm",
            height=500,
            xaxis=dict(range=[0.65, 0.80])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Visualization 2: Multiple metrics comparison
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=comparison_df['Algorithm'],
            x=comparison_df['Accuracy'],
            name='Accuracy',
            orientation='h'
        ))
        fig.add_trace(go.Bar(
            y=comparison_df['Algorithm'],
            x=comparison_df['Precision'],
            name='Precision',
            orientation='h'
        ))
        fig.add_trace(go.Bar(
            y=comparison_df['Algorithm'],
            x=comparison_df['Recall'],
            name='Recall',
            orientation='h'
        ))
        
        fig.update_layout(
            title="Metrics Comparison",
            xaxis_title="Score",
            barmode='group',
            height=500,
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: Best Model Prediction
with tab2:
    st.header("Diabetes Risk Prediction (Best Model)")
    st.write(f"Using **{best_model_row['Algorithm']}** - Accuracy: {best_model_row['Accuracy']:.2%}")
    
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
    
    if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
        # Prepare input data
        input_data = np.array([[preg_count, glucose, diastolic_bp, triceps, insulin, bmi, diabetes_pedi, age]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Try to get probability (works for most models)
        try:
            probability = model.predict_proba(input_scaled)[0]
            has_proba = True
        except:
            has_proba = False
        
        # Display results
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ **HIGH RISK**: Diabetes Likely Present")
                risk_level = "High"
                if has_proba:
                    risk_percentage = probability[1] * 100
                else:
                    risk_percentage = 65
            else:
                st.success("✅ **LOW RISK**: Diabetes Unlikely")
                risk_level = "Low"
                if has_proba:
                    risk_percentage = probability[0] * 100
                else:
                    risk_percentage = 75
        
        with col2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percentage if has_proba else (risk_percentage if prediction == 0 else 100-risk_percentage),
                title={'text': "Diabetes Risk Score"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ff4444" if (risk_percentage if has_proba else risk_percentage) > 50 else "#44aa44"},
                    'steps': [
                        {'range': [0, 25], 'color': "#e8f4f8"},
                        {'range': [25, 50], 'color': "#b3d9e6"},
                        {'range': [50, 75], 'color': "#ffd9b3"},
                        {'range': [75, 100], 'color': "#ffb3b3"}
                    ]
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

# TAB 3: Performance Analysis
with tab3:
    st.header("Detailed Performance Analysis")
    
    # Load test data
    df = pd.read_csv('Pima.csv')
    X = df.drop('diabetes_class', axis=1)
    y = df['diabetes_class']
    
    # Handle missing values
    cols_with_zeros = ['glucose_concentration', 'diastolic_bp', 'triceps_skin_fold_thickness', 
                        'two_hr_serum_insulin', 'bmi']
    for col in cols_with_zeros:
        median_val = X[X[col] != 0][col].median()
        X.loc[X[col] == 0, col] = median_val
    
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Total Samples", len(y))
    with col3:
        st.metric("Positive Cases", f"{sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    
    st.markdown("---")
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Negative', 'Predicted Positive'],
        y=['Actual Negative', 'Actual Positive'],
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        colorscale='Blues'
    ))
    fig.update_layout(title="Confusion Matrix", height=400, width=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification Metrics")
        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format('{:.4f}'), use_container_width=True)
    
    with col2:
        st.subheader("Algorithm Accuracy Rankings")
        ranking_fig = go.Figure(data=[
            go.Bar(
                y=comparison_df['Algorithm'].head(10),
                x=comparison_df['Accuracy'].head(10),
                orientation='h',
                marker=dict(color=comparison_df['Accuracy'].head(10), colorscale='Viridis')
            )
        ])
        ranking_fig.update_layout(title="Top 10 Algorithms", xaxis_title="Accuracy", yaxis_title="Algorithm", height=400)
        st.plotly_chart(ranking_fig, use_container_width=True)

# TAB 4: About
with tab4:
    st.header("About This Comparison")
    
    st.subheader("10 Classification Algorithms Tested")
    algorithms_info = {
        "1. AdaBoost": "Adaptive Boosting - Combines weak learners to create strong classifier",
        "2. Random Forest": "Ensemble of decision trees with random feature selection",
        "3. K-Nearest Neighbors": "Classification based on k nearest neighbors distance",
        "4. SVM (RBF)": "Support Vector Machine with RBF kernel for non-linear separation",
        "5. Gradient Boosting": "Sequential tree building to minimize loss function",
        "6. Neural Network (MLP)": "Multi-layer perceptron neural network",
        "7. Logistic Regression": "Linear model for binary classification",
        "8. SVM (Linear)": "Support Vector Machine with linear kernel",
        "9. Naive Bayes": "Probabilistic classifier based on Bayes' theorem",
        "10. Decision Tree": "Tree-based model for hierarchical decision making"
    }
    
    for algo, description in algorithms_info.items():
        st.write(f"**{algo}**: {description}")
    
    st.markdown("---")
    
    st.subheader("Dataset Information")
    st.write("""
    - **Dataset**: Pima Indians Diabetes Database
    - **Total Samples**: 768
    - **Training Set**: 80% (614 samples)
    - **Test Set**: 20% (154 samples)
    - **Features**: 8 medical metrics
    - **Target**: Binary classification (Diabetes: Yes/No)
    """)
    
    st.markdown("---")
    
    st.subheader("Evaluation Metrics")
    st.write("""
    - **Accuracy**: Proportion of correct predictions
    - **Precision**: True positives / (True positives + False positives)
    - **Recall**: True positives / (True positives + False negatives)
    - **F1-Score**: Harmonic mean of Precision and Recall
    - **Cross-Validation**: 5-fold CV for robust performance estimation
    """)
    
    st.markdown("---")
    
    st.subheader("Key Findings")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Best Performing Algorithm**: {best_model_row['Algorithm']}
        
        **Accuracy**: {best_model_row['Accuracy']:.2%}
        
        **Precision**: {best_model_row['Precision']:.4f}
        
        **Recall**: {best_model_row['Recall']:.4f}
        """)
    
    with col2:
        st.warning("""
        **Important Disclaimer**
        
        ⚠️ This model is for educational and research purposes only.
        
        It should NOT be used for medical diagnosis. Always consult with a qualified healthcare professional.
        
        The predictions are probabilistic and may not reflect actual medical conditions.
        """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Pima Diabetes Prediction | 10 Algorithms Comparison | Built with Streamlit</div>", unsafe_allow_html=True)
