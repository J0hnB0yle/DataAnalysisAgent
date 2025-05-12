import streamlit as st
import pandas as pd
import os
import tempfile
import sys
import base64
from PIL import Image
import io

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import load_data, analyze_data, create_visualization, predict_values

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
        tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Analysis & Insights", "Visualizations", "Predictions"])
        
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
            
            # Display interactive visualization creator
            st.subheader("Create Visualization")
            
            # Get column lists by type
            numeric_cols = list(data.select_dtypes(include=['number']).columns)
            categorical_cols = list(data.select_dtypes(include=['object', 'category']).columns)
            
            # Let user select visualization type
            viz_type = st.selectbox(
                "Select visualization type",
                ["histogram", "scatter", "bar", "box", "correlation", "line", "pie"]
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
                col = st.selectbox("Select column", categorical_cols if categorical_cols else numeric_cols)
                columns = [col]
            elif viz_type == "box":
                columns = st.multiselect("Select columns", numeric_cols)
            elif viz_type == "correlation":
                columns = st.multiselect("Select columns for correlation matrix", numeric_cols)
            elif viz_type == "line":
                columns = st.multiselect("Select columns for line chart", numeric_cols)
            elif viz_type == "pie":
                col = st.selectbox("Select categorical column", categorical_cols if categorical_cols else numeric_cols)
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
        
        # Tab 4: Predictions
        with tab4:
            st.header("Predictive Analytics")
            
            # Check if we have enough numeric columns for prediction
            numeric_cols = list(data.select_dtypes(include=['number']).columns)
            if len(numeric_cols) < 2:
                st.warning("Prediction requires at least 2 numeric columns (1 target + 1 feature).")
            else:
                st.subheader("Linear Regression Analysis")
                target_col = st.selectbox("Select target column to predict:", numeric_cols)
                
                available_features = [col for col in numeric_cols if col != target_col]
                feature_cols = st.multiselect("Select feature columns (predictors):", available_features)
                
                if st.button("Run Prediction") and feature_cols:
                    if len(feature_cols) < 1:
                        st.error("Please select at least one feature column.")
                    else:
                        with st.spinner("Running prediction model..."):
                            prediction_results = predict_values(data, target_col, feature_cols)
                            
                            st.subheader("Model Performance")
                            st.write(f"R² Score: {prediction_results['r2']:.4f} (Higher is better, 1.0 is perfect)")
                            st.write(f"Mean Squared Error: {prediction_results['mse']:.4f} (Lower is better)")
                            
                            st.subheader("Feature Importance")
                            feature_importance = pd.DataFrame(prediction_results['coefficients'])
                            st.dataframe(feature_importance)
                            
                            st.subheader("Model Insights")
                            st.write(prediction_results['model_insights'])
    else:
        st.error("Failed to load data. Please check your file format.")
