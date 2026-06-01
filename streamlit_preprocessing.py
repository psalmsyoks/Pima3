import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Preprocessing Techniques Comparison",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .preprocessing-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    results_df = pd.read_csv('preprocessing_comparison_results.csv')
    accuracy_pivot = pd.read_csv('preprocessing_accuracy_pivot.csv', index_col=0)
    f1_pivot = pd.read_csv('preprocessing_f1_pivot.csv', index_col=0)
    return results_df, accuracy_pivot, f1_pivot

results_df, accuracy_pivot, f1_pivot = load_data()

# Page title
st.title("🔬 Preprocessing Techniques Comparison")
st.markdown("Analyzing 5 preprocessing techniques with 3 classification models")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Results Table", "📈 Accuracy Analysis", "🎯 Performance Metrics", "ℹ️ Techniques Info"])

# TAB 1: Results Table
with tab1:
    st.header("Comprehensive Results Table")
    st.write("All preprocessing techniques with 3 models and multiple metrics")
    
    # Display results table
    display_df = results_df.copy()
    st.dataframe(display_df, use_container_width=True)
    
    st.markdown("---")
    
    # Find best combinations for each metric
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_accuracy_idx = results_df['Accuracy'].idxmax()
        best_accuracy_row = results_df.loc[best_accuracy_idx]
        st.markdown(f"""
        **🏆 Best Accuracy**
        
        {best_accuracy_row['Preprocessing']} + {best_accuracy_row['Model']}
        
        **Score: {best_accuracy_row['Accuracy']:.4f}**
        """)
    
    with col2:
        best_precision_idx = results_df['Precision'].idxmax()
        best_precision_row = results_df.loc[best_precision_idx]
        st.markdown(f"""
        **🎯 Best Precision**
        
        {best_precision_row['Preprocessing']} + {best_precision_row['Model']}
        
        **Score: {best_precision_row['Precision']:.4f}**
        """)
    
    with col3:
        best_f1_idx = results_df['F1-Score'].idxmax()
        best_f1_row = results_df.loc[best_f1_idx]
        st.markdown(f"""
        **⭐ Best F1-Score**
        
        {best_f1_row['Preprocessing']} + {best_f1_row['Model']}
        
        **Score: {best_f1_row['F1-Score']:.4f}**
        """)

# TAB 2: Accuracy Analysis
with tab2:
    st.header("Accuracy Analysis by Preprocessing Technique")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accuracy Heatmap")
        
        # Create heatmap for accuracy
        fig = go.Figure(data=go.Heatmap(
            z=accuracy_pivot.values,
            x=accuracy_pivot.columns,
            y=accuracy_pivot.index,
            colorscale='RdYlGn',
            text=np.round(accuracy_pivot.values, 4),
            texttemplate='%{text:.4f}',
            textfont={"size": 11}
        ))
        fig.update_layout(
            title="Accuracy by Model and Preprocessing",
            xaxis_title="Preprocessing Technique",
            yaxis_title="Classification Model",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Accuracy Comparison by Preprocessing")
        
        # Average accuracy for each preprocessing technique
        avg_accuracy = results_df.groupby('Preprocessing')['Accuracy'].mean().sort_values(ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(
                y=avg_accuracy.index,
                x=avg_accuracy.values,
                orientation='h',
                marker=dict(
                    color=avg_accuracy.values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Avg Accuracy")
                ),
                text=np.round(avg_accuracy.values, 4),
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Average Accuracy per Preprocessing Technique",
            xaxis_title="Average Accuracy",
            yaxis_title="Preprocessing Technique",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Accuracy comparison by model
    st.subheader("Accuracy Comparison by Model")
    
    col1, col2, col3 = st.columns(3)
    
    models = results_df['Model'].unique()
    
    for idx, model in enumerate(models):
        model_data = results_df[results_df['Model'] == model].sort_values('Accuracy', ascending=False)
        
        if idx == 0:
            col = col1
        elif idx == 1:
            col = col2
        else:
            col = col3
        
        with col:
            fig = go.Figure(data=[
                go.Bar(
                    y=model_data['Preprocessing'],
                    x=model_data['Accuracy'],
                    orientation='h',
                    marker_color='lightblue',
                    text=model_data['Accuracy'].round(4),
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title=f"{model} - Accuracy by Preprocessing",
                xaxis_title="Accuracy",
                height=400,
                xaxis=dict(range=[0.72, 0.77])
            )
            st.plotly_chart(fig, use_container_width=True)

# TAB 3: Performance Metrics
with tab3:
    st.header("Performance Metrics Comparison")
    
    # F1-Score Heatmap
    st.subheader("F1-Score by Model and Preprocessing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=go.Heatmap(
            z=f1_pivot.values,
            x=f1_pivot.columns,
            y=f1_pivot.index,
            colorscale='Blues',
            text=np.round(f1_pivot.values, 4),
            texttemplate='%{text:.4f}',
            textfont={"size": 11}
        ))
        fig.update_layout(
            title="F1-Score Heatmap",
            xaxis_title="Preprocessing Technique",
            yaxis_title="Classification Model",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Average Metrics by Preprocessing")
        
        avg_metrics = results_df.groupby('Preprocessing')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].mean()
        
        fig = go.Figure()
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            fig.add_trace(go.Bar(
                x=avg_metrics.index,
                y=avg_metrics[metric],
                name=metric
            ))
        
        fig.update_layout(
            title="Average Metrics by Preprocessing",
            xaxis_title="Preprocessing Technique",
            yaxis_title="Score",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed comparison table
    st.subheader("Detailed Metrics Table")
    
    avg_metrics_display = results_df.groupby('Preprocessing')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].mean()
    st.dataframe(avg_metrics_display, use_container_width=True)

# TAB 4: Techniques Information
with tab4:
    st.header("Preprocessing Techniques Guide")
    
    techniques_info = {
        "Standard Scaler": {
            "description": "Transforms features by subtracting mean and dividing by standard deviation",
            "formula": "(X - mean) / std",
            "range": "[-∞, +∞]",
            "best_for": "Algorithms sensitive to feature scaling (SVM, Neural Networks, KNN)",
            "pros": "✓ Zero mean and unit variance | ✓ Preserves outlier information",
            "cons": "✗ Sensitive to outliers",
            "use_case": "When data is normally distributed"
        },
        "MinMax Scaler": {
            "description": "Scales features to a fixed range [0, 1]",
            "formula": "(X - min) / (max - min)",
            "range": "[0, 1]",
            "best_for": "Neural Networks, Image processing, Algorithms with bounded inputs",
            "pros": "✓ Preserves exact relationship | ✓ Bounded range",
            "cons": "✗ Very sensitive to outliers",
            "use_case": "When you need features in a specific range"
        },
        "Robust Scaler": {
            "description": "Uses median and interquartile range (IQR) for scaling",
            "formula": "(X - median) / IQR",
            "range": "[-∞, +∞]",
            "best_for": "Data with outliers, Tree-based models",
            "pros": "✓ Robust to outliers | ✓ Good for skewed data",
            "cons": "✗ Less effective on normal distributions",
            "use_case": "When data contains significant outliers"
        },
        "MaxAbs Scaler": {
            "description": "Scales features by dividing by maximum absolute value",
            "formula": "X / max(|X|)",
            "range": "[-1, 1]",
            "best_for": "Sparse data, Negative values preservation",
            "pros": "✓ Preserves sparsity | ✓ Doesn't shift zero",
            "cons": "✗ Sensitive to outliers",
            "use_case": "Sparse matrices and data with negative values"
        },
        "Log Transformation": {
            "description": "Applies logarithmic transformation: log(1 + X)",
            "formula": "log(1 + X)",
            "range": "[0, +∞]",
            "best_for": "Right-skewed data, Reduces impact of large values",
            "pros": "✓ Reduces skewness | ✓ Handles wide value ranges",
            "cons": "✗ Cannot handle negative values (hence log1p)",
            "use_case": "Highly skewed features"
        }
    }
    
    for technique_name, info in techniques_info.items():
        with st.container():
            st.markdown(f"""
            ### {technique_name}
            
            **Description:** {info['description']}
            
            **Formula:** `{info['formula']}`
            
            **Output Range:** {info['range']}
            
            **Best For:** {info['best_for']}
            
            **Advantages:**
            {info['pros']}
            
            **Disadvantages:**
            {info['cons']}
            
            **Use Case:** {info['use_case']}
            """)
            st.markdown("---")
    
    st.info("""
    ### Key Findings from Analysis
    
    1. **AdaBoost** performs well with most preprocessing techniques
    2. **Robust Scaler and Log Transformation** show slight edge for Random Forest
    3. **MinMax Scaler** works well for tree-based models
    4. **Tree-based models** (Random Forest, Gradient Boosting) are less sensitive to feature scaling
    5. **Standard Scaler** is a safe choice for most algorithms
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Preprocessing Techniques Comparison | 5 Scalers | 3 Models | Pima Dataset</div>", unsafe_allow_html=True)
