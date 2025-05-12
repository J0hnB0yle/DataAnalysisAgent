interface/web_app.py

import streamlit as st
import pandas as pd
import os
import tempfile
import sys
import base64
from PIL import Image
import io

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import load_data, analyze_data, create_visualization, recommend_visualizations, explain_analysis

st.set_page_config(page_title="Data Analysis Agent", layout="wide")

st.title("Data Analysis Agent")
st.write("Upload data files and generate insights and visualizations")

# File upload
uploaded_file = st.file_uploader("Upload data file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # Save uploaded file to temp directory
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"Uploaded: {uploaded_file.name}")
    
    # Load data
    data = load_data(file_path)
    
    if data is not None:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Analysis & Insights", "Visualizations", "Custom Analysis"])
        
        # Tab 1: Data Overview
        with tab1:
            st.header("Data Overview")
            st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
            
            # Display column info
            col_info = pd.DataFrame({
                'Data Type': data.dtypes,
                'Non-Null Count': data.count(),
                'Null Count': data.isnull().sum(),
                'Unique Values': [data[col].nunique() for col in data.columns]
            })
            st.subheader("Column Information")
            st.dataframe(col_info)
            
            # Display data sample
            st.subheader("Data Sample")
            st.dataframe(data.head(10))
        
        # Tab 2: Analysis & Insights
        with tab2:
            st.header("Analysis & Insights")
            
            if st.button("Generate Analysis"):
                with st.spinner("Analyzing data..."):
                    analysis_results = analyze_data(data)
                    
                    # Display insights
                    st.subheader("AI-Generated Insights")
                    st.write(analysis_results['insights'])
                    
                    # Display basic statistics for numeric columns
                    if 'numeric_stats' in analysis_results['statistics']:
                        st.subheader("Numeric Column Statistics")
                        for col, stats in analysis_results['statistics']['numeric_stats'].items():
                            st.write(f"**{col}**")
                            st.write(f"Mean: {stats['mean']:.2f}, Median: {stats['50%']:.2f}, Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
                            st.write(f"Standard Deviation: {stats['std']:.2f}")
                            st.write("---")
        
        # Tab 3: Visualizations
        with tab3:
            st.header("Visualizations")
            
            if st.button("Recommend Visualizations"):
                with st.spinner("Generating recommendations..."):
                    viz_recommendations = recommend_visualizations(data)
                    
                    st.subheader("Recommended Visualizations")
                    st.write(viz_recommendations['enhanced_suggestions'])
                    
                    # Display interactive visualization creator
                    st.subheader("Create Visualization")
                    
                    # Get column lists by type
                    numeric_cols = list(data.select_dtypes(include=['number']).columns)
                    categorical_cols = list(data.select_dtypes(include=['object', 'category']).columns)
                    
                    # Let user select visualization type
                    viz_type = st.selectbox(
                        "Select visualization type",
                        ["histogram", "scatter", "bar", "box", "correlation", "pie"]
                    )
                    
                    # Conditional inputs based on visualization type
                    if viz_type == "histogram":
                        col = st.selectbox("Select column", numeric_cols)
                        columns = [col]
                    elif viz_type == "scatter":
                        col1 = st.selectbox("Select X-axis column", numeric_cols)
                        col2 = st.selectbox("Select Y-axis column", numeric_cols)
                        columns = [col1, col2]
                    elif viz_type == "bar":
                        col = st.selectbox("Select column", categorical_cols)
                        columns = [col]
                    elif viz_type == "box":
                        columns = st.multiselect("Select columns", numeric_cols)
                    elif viz_type == "correlation":
                        columns = st.multiselect("Select columns for correlation matrix", numeric_cols)
                    elif viz_type == "pie":
                        col = st.selectbox("Select categorical column", categorical_cols)
                        columns = [col]
                    
                    title = st.text_input("Visualization title (optional)")
                    
                    if st.button("Generate Visualization"):
                        if not columns:
                            st.error("Please select at least one column")
                        else:
                            with st.spinner("Creating visualization..."):
                                img_str, error = create_visualization(data, viz_type, columns, title)
                                
                                if error:
                                    st.error(error)
                                else:
                                    st.image(f"data:image/png;base64,{img_str}")
                                    
                                    # Add explanation of the visualization
                                    explanation = explain_analysis(data, viz_type, {"columns": columns})
                                    st.subheader("Interpretation")
                                    st.write(explanation)
        
        # Tab 4: Custom Analysis
        with tab4:
            st.header("Custom Analysis")
            
            analysis_type = st.selectbox(
                "Select analysis type",
                ["Correlation Analysis", "Time Series Decomposition", "Cluster Analysis", 
                 "Outlier Detection", "Feature Importance"]
            )
            
            st.write("Configure analysis parameters:")
            
            parameters = {}
            if analysis_type == "Correlation Analysis":
                parameters['columns'] = st.multiselect("Select columns", data.columns)
                parameters['method'] = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
            elif analysis_type == "Time Series Decomposition":
                date_cols = st.selectbox("Select date column", data.columns)
                value_cols = st.selectbox("Select value column", numeric_cols)
                parameters['date_column'] = date_cols
                parameters['value_column'] = value_cols
                parameters['period'] = st.number_input("Period (e.g., 12 for monthly data)", min_value=2, value=12)
            elif analysis_type == "Cluster Analysis":
                parameters['columns'] = st.multiselect("Select columns for clustering", numeric_cols)
                parameters['n_clusters'] = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
            elif analysis_type == "Outlier Detection":
                parameters['columns'] = st.multiselect("Select columns to check for outliers", numeric_cols)
                parameters['method'] = st.selectbox("Detection method", ["Z-Score", "IQR", "Isolation Forest"])
            elif analysis_type == "Feature Importance":
                parameters['target'] = st.selectbox("Select target column", data.columns)
                parameters['features'] = st.multiselect("Select feature columns", [c for c in data.columns if c != parameters.get('target')])
            
            if st.button("Run Analysis"):
                with st.spinner(f"Running {analysis_type}..."):
                    explanation = explain_analysis(data, analysis_type, parameters)
                    st.subheader("Analysis Explanation")
                    st.write(explanation)
                    
                    # Here you would actually run the analysis
                    # This would be implemented in the main.py file
                    st.write("Analysis implementation would go here")
    else:
        st.error("Failed to load data. Please check your file format.")