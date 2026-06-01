import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Class Imbalance Solutions Comparison",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .technique-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 10px;
    }
    .best-card {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    results_df = pd.read_csv('imbalance_solution_results.csv')
    accuracy_pivot = pd.read_csv('imbalance_accuracy_pivot.csv', index_col=0)
    f1_pivot = pd.read_csv('imbalance_f1_pivot.csv', index_col=0)
    roc_pivot = pd.read_csv('imbalance_roc_pivot.csv', index_col=0)
    return results_df, accuracy_pivot, f1_pivot, roc_pivot

results_df, accuracy_pivot, f1_pivot, roc_pivot = load_data()

# Page title
st.title("⚖️ Class Imbalance Solutions Comparison")
st.markdown("Comparing No Balancing, SMOTE, and RandomOverSampler techniques")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Results Table", "📈 Accuracy Comparison", "🎯 F1-Score Analysis", "📉 ROC-AUC Analysis", "ℹ️ Techniques Info"])

# TAB 1: Results Table
with tab1:
    st.header("Comprehensive Results Table")
    st.write("All combinations: 3 techniques × 5 models = 15 results")
    
    # Display results table
    display_df = results_df.copy()
    st.dataframe(display_df, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_accuracy_idx = results_df['Accuracy'].idxmax()
        best_accuracy_row = results_df.loc[best_accuracy_idx]
        st.markdown(f"""
        <div class='best-card'>
        <h3>🏆 Best Accuracy</h3>
        <p><b>Technique:</b> {best_accuracy_row['Technique']}</p>
        <p><b>Model:</b> {best_accuracy_row['Model']}</p>
        <p><b>Score:</b> {best_accuracy_row['Accuracy']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        best_f1_idx = results_df['F1-Score'].idxmax()
        best_f1_row = results_df.loc[best_f1_idx]
        st.markdown(f"""
        <div class='best-card'>
        <h3>⭐ Best F1-Score</h3>
        <p><b>Technique:</b> {best_f1_row['Technique']}</p>
        <p><b>Model:</b> {best_f1_row['Model']}</p>
        <p><b>Score:</b> {best_f1_row['F1-Score']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        best_roc_idx = results_df['ROC-AUC'].idxmax()
        best_roc_row = results_df.loc[best_roc_idx]
        st.markdown(f"""
        <div class='best-card'>
        <h3>📊 Best ROC-AUC</h3>
        <p><b>Technique:</b> {best_roc_row['Technique']}</p>
        <p><b>Model:</b> {best_roc_row['Model']}</p>
        <p><b>Score:</b> {best_roc_row['ROC-AUC']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Top 5 by each metric
    st.subheader("Top 5 Best Combinations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**By Accuracy**")
        top_acc = results_df.nlargest(5, 'Accuracy')[['Technique', 'Model', 'Accuracy']]
        st.dataframe(top_acc, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**By F1-Score**")
        top_f1 = results_df.nlargest(5, 'F1-Score')[['Technique', 'Model', 'F1-Score']]
        st.dataframe(top_f1, use_container_width=True, hide_index=True)
    
    with col3:
        st.write("**By ROC-AUC**")
        top_roc = results_df.nlargest(5, 'ROC-AUC')[['Technique', 'Model', 'ROC-AUC']]
        st.dataframe(top_roc, use_container_width=True, hide_index=True)

# TAB 2: Accuracy Comparison
with tab2:
    st.header("Accuracy Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accuracy Heatmap")
        
        fig = go.Figure(data=go.Heatmap(
            z=accuracy_pivot.values,
            x=accuracy_pivot.columns,
            y=accuracy_pivot.index,
            colorscale='RdYlGn',
            text=np.round(accuracy_pivot.values, 4),
            texttemplate='%{text:.4f}',
            textfont={"size": 10}
        ))
        fig.update_layout(
            title="Accuracy by Model and Technique",
            xaxis_title="Imbalance Handling Technique",
            yaxis_title="Classification Model",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Average Accuracy per Technique")
        
        avg_acc = results_df.groupby('Technique')['Accuracy'].mean().sort_values(ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(
                y=avg_acc.index,
                x=avg_acc.values,
                orientation='h',
                marker=dict(
                    color=avg_acc.values,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=np.round(avg_acc.values, 4),
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Average Accuracy per Technique",
            xaxis_title="Average Accuracy",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # By model
    st.subheader("Accuracy by Model Across Techniques")
    
    col1, col2, col3 = st.columns(3)
    models = results_df['Model'].unique()
    
    for idx, model in enumerate(models):
        model_data = results_df[results_df['Model'] == model].sort_values('Accuracy', ascending=False)
        
        if idx % 3 == 0:
            col = col1
        elif idx % 3 == 1:
            col = col2
        else:
            col = col3
        
        with col:
            fig = go.Figure(data=[
                go.Bar(
                    y=model_data['Technique'],
                    x=model_data['Accuracy'],
                    orientation='h',
                    marker_color='lightblue',
                    text=model_data['Accuracy'].round(4),
                    textposition='auto'
                )
            ])
            fig.update_layout(
                title=f"{model}",
                xaxis_title="Accuracy",
                height=350,
                xaxis=dict(range=[0.68, 0.78])
            )
            st.plotly_chart(fig, use_container_width=True)

# TAB 3: F1-Score Analysis
with tab3:
    st.header("F1-Score Analysis (Best for Imbalanced Data)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("F1-Score Heatmap")
        
        fig = go.Figure(data=go.Heatmap(
            z=f1_pivot.values,
            x=f1_pivot.columns,
            y=f1_pivot.index,
            colorscale='Blues',
            text=np.round(f1_pivot.values, 4),
            texttemplate='%{text:.4f}',
            textfont={"size": 10}
        ))
        fig.update_layout(
            title="F1-Score by Model and Technique",
            xaxis_title="Imbalance Handling Technique",
            yaxis_title="Classification Model",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Average F1-Score per Technique")
        
        avg_f1 = results_df.groupby('Technique')['F1-Score'].mean().sort_values(ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(
                y=avg_f1.index,
                x=avg_f1.values,
                orientation='h',
                marker=dict(
                    color=avg_f1.values,
                    colorscale='Blues',
                    showscale=True
                ),
                text=np.round(avg_f1.values, 4),
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Average F1-Score per Technique",
            xaxis_title="Average F1-Score",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Comparison of metrics for better understanding
    st.subheader("Precision vs Recall vs F1-Score")
    
    avg_metrics = results_df.groupby('Technique')[['Precision', 'Recall', 'F1-Score']].mean()
    
    fig = go.Figure()
    for metric in ['Precision', 'Recall', 'F1-Score']:
        fig.add_trace(go.Bar(
            x=avg_metrics.index,
            y=avg_metrics[metric],
            name=metric
        ))
    
    fig.update_layout(
        title="Average Metrics by Technique",
        xaxis_title="Imbalance Handling Technique",
        yaxis_title="Score",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# TAB 4: ROC-AUC Analysis
with tab4:
    st.header("ROC-AUC Analysis")
    st.write("ROC-AUC is a comprehensive metric for evaluating classifier performance on imbalanced datasets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROC-AUC Heatmap")
        
        fig = go.Figure(data=go.Heatmap(
            z=roc_pivot.values,
            x=roc_pivot.columns,
            y=roc_pivot.index,
            colorscale='Greens',
            text=np.round(roc_pivot.values, 4),
            texttemplate='%{text:.4f}',
            textfont={"size": 10}
        ))
        fig.update_layout(
            title="ROC-AUC by Model and Technique",
            xaxis_title="Imbalance Handling Technique",
            yaxis_title="Classification Model",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Average ROC-AUC per Technique")
        
        avg_roc = results_df.groupby('Technique')['ROC-AUC'].mean().sort_values(ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(
                y=avg_roc.index,
                x=avg_roc.values,
                orientation='h',
                marker=dict(
                    color=avg_roc.values,
                    colorscale='Greens',
                    showscale=True
                ),
                text=np.round(avg_roc.values, 4),
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Average ROC-AUC per Technique",
            xaxis_title="Average ROC-AUC",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

# TAB 5: Techniques Information
with tab5:
    st.header("Class Imbalance Handling Techniques")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### No Balancing
        
        **Description:** Training on original imbalanced data
        
        **Class Distribution:**
        - Class 0: 400 samples (65.1%)
        - Class 1: 214 samples (34.9%)
        
        **Pros:**
        - ✓ No synthetic data created
        - ✓ Lower computational cost
        - ✓ Preserves original data
        
        **Cons:**
        - ✗ Biased towards majority class
        - ✗ Low recall for minority class
        - ✗ High false negatives
        
        **Best For:** Balanced datasets
        """)
    
    with col2:
        st.markdown("""
        ### SMOTE (Synthetic Minority Oversampling)
        
        **Description:** Creates synthetic samples in feature space
        
        **Class Distribution:**
        - Class 0: 400 samples (50%)
        - Class 1: 400 samples (50%)
        
        **Pros:**
        - ✓ Intelligent synthetic samples
        - ✓ Addresses overfitting
        - ✓ Better recall
        - ✓ K-NN based approach
        
        **Cons:**
        - ✗ Higher computation
        - ✗ May create overlapping samples
        - ✗ Doesn't solve all issues
        
        **Best For:** Severely imbalanced data
        """)
    
    with col3:
        st.markdown("""
        ### RandomOverSampler
        
        **Description:** Randomly duplicates minority class samples
        
        **Class Distribution:**
        - Class 0: 400 samples (50%)
        - Class 1: 400 samples (50%)
        
        **Pros:**
        - ✓ Simple and fast
        - ✓ No information loss
        - ✓ Easy to understand
        - ✓ Good baseline
        
        **Cons:**
        - ✗ Creates duplicates
        - ✗ Risk of overfitting
        - ✗ Less robust than SMOTE
        - ✗ Increases data size
        
        **Best For:** Quick baseline solutions
        """)
    
    st.markdown("---")
    
    st.subheader("Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Performance Insights:**
        
        1. **Best Overall:** SMOTE + AdaBoost
           - F1-Score: 0.6721
           - Highest Recall: 75.93%
        
        2. **Best Accuracy:** No Balancing + AdaBoost
           - Accuracy: 75.97%
        
        3. **Best ROC-AUC:** RandomOverSampler + Random Forest
           - ROC-AUC: 0.8298
        """)
    
    with col2:
        st.warning("""
        **Recommendations:**
        
        1. **For High Recall:** Use SMOTE
           - Better at finding minority class
           - Reduces false negatives
        
        2. **For Balanced Performance:** Use RandomOverSampler
           - Good F1-Score
           - Simpler than SMOTE
        
        3. **Consider:** ROC-AUC metric
           - More reliable for imbalanced data
           - Not affected by threshold
        """)
    
    st.markdown("---")
    
    st.subheader("Data Distribution Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        labels = ['No Diabetes', 'Diabetes']
        sizes = [400, 214]
        colors = ['#66c2a5', '#fc8d62']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=sizes,
            marker=dict(colors=colors),
            title="Original Data"
        )])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        labels = ['No Diabetes', 'Diabetes']
        sizes = [400, 400]
        colors = ['#66c2a5', '#fc8d62']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=sizes,
            marker=dict(colors=colors),
            title="After SMOTE"
        )])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        labels = ['No Diabetes', 'Diabetes']
        sizes = [400, 400]
        colors = ['#66c2a5', '#fc8d62']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=sizes,
            marker=dict(colors=colors),
            title="After RandomOverSampler"
        )])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Class Imbalance Solutions | SMOTE | RandomOverSampler | Pima Dataset</div>", unsafe_allow_html=True)
