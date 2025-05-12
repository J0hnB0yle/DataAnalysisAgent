import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import anthropic
import io
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

def load_data(file_path):
    """Load data from CSV or Excel file"""
    print(f"Loading data from: {file_path}")
    try:
        if file_path.lower().endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.lower().endswith(('.xls', '.xlsx', '.xlsm')):
            data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")
        
        print(f"Successfully loaded data with {data.shape[0]} rows and {data.shape[1]} columns")
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def analyze_data(data):
    """Generate basic statistics and insights about the data"""
    if data is None or data.empty:
        return "No data to analyze"
    
    # Basic statistics
    stats = {}
    stats['shape'] = data.shape
    stats['columns'] = list(data.columns)
    stats['dtypes'] = data.dtypes.to_dict()
    stats['missing_values'] = data.isnull().sum().to_dict()
    
    # Numeric columns statistics
    numeric_cols = data.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        stats['numeric_stats'] = data[numeric_cols].describe().to_dict()
    
    # Categorical columns statistics
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        stats['categorical_counts'] = {col: data[col].value_counts().to_dict() for col in cat_cols if len(data[col].unique()) < 10}
    
    # Generate insights using Claude
    insights = generate_insights(data, stats)
    
    return {
        'statistics': stats,
        'insights': insights
    }

def generate_insights(data, stats):
    """Use Claude to generate insights from the data"""
    # Sample first 20 rows for context
    sample_data = data.head(20).to_string()
    
    # Prepare statistics for prompt
    stats_str = str(stats)
    
    # Create prompt for insights
    prompt = f"""
    I have a dataset with the following characteristics:
    - Shape: {stats['shape']}
    - Columns: {stats['columns']}
    - Data types: {stats['dtypes']}
    
    Here's a sample of the data:
    {sample_data}
    
    Based on this information, please provide:
    1. Key insights about this data
    2. Potential patterns or relationships to explore
    3. Recommended visualizations that would be informative
    4. Data quality issues that should be addressed
    5. Suggested analyses to extract more value from this data
    
    Be specific and concise in your recommendations.
    """
    
    # Generate insights using Claude
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1500,
        temperature=0.0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text

def create_visualization(data, viz_type, columns, title=None):
    """Create visualization based on specified type and columns"""
    if data is None or data.empty:
        return None, "No data to visualize"
    
    plt.figure(figsize=(10, 6))
    
    try:
        if viz_type == 'histogram':
            plt.hist(data[columns[0]], bins=30)
            plt.xlabel(columns[0])
            plt.ylabel('Frequency')
            title = title or f'Histogram of {columns[0]}'
            
        elif viz_type == 'scatter':
            plt.scatter(data[columns[0]], data[columns[1]])
            plt.xlabel(columns[0])
            plt.ylabel(columns[1])
            title = title or f'Scatter Plot of {columns[1]} vs {columns[0]}'
            
        elif viz_type == 'bar':
            data[columns[0]].value_counts().sort_values(ascending=False).head(15).plot(kind='bar')
            plt.xlabel(columns[0])
            plt.ylabel('Count')
            title = title or f'Bar Chart of {columns[0]}'
            
        elif viz_type == 'box':
            data[columns].boxplot()
            plt.ylabel('Value')
            title = title or f'Box Plot of {", ".join(columns)}'
            
        elif viz_type == 'correlation':
            corr_matrix = data[columns].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
            title = title or 'Correlation Matrix'
            
        elif viz_type == 'pie':
            if len(columns) != 1:
                return None, "Pie chart requires exactly one categorical column"
            value_counts = data[columns[0]].value_counts()
            plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
            title = title or f'Pie Chart of {columns[0]}'
            
        else:
            return None, f"Unsupported visualization type: {viz_type}"
        
        plt.title(title)
        plt.tight_layout()
        
        # Save visualization to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode the image to base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str, None
        
    except Exception as e:
        return None, f"Error creating visualization: {str(e)}"

def recommend_visualizations(data):
    """Recommend useful visualizations based on data characteristics"""
    if data is None or data.empty:
        return "No data to analyze for visualization recommendations"
    
    recommendations = []
    
    # Get column types
    numeric_cols = list(data.select_dtypes(include=['number']).columns)
    categorical_cols = list(data.select_dtypes(include=['object', 'category']).columns)
    datetime_cols = list(data.select_dtypes(include=['datetime']).columns)
    
    # Recommend histograms for numeric columns
    if numeric_cols:
        for col in numeric_cols[:3]:  # Limit to first 3 to avoid too many recommendations
            recommendations.append({
                'type': 'histogram',
                'columns': [col],
                'title': f'Distribution of {col}',
                'reason': f'Shows the distribution of values for {col}'
            })
    
    # Recommend scatter plots for pairs of numeric columns
    if len(numeric_cols) >= 2:
        for i in range(min(3, len(numeric_cols))):
            for j in range(i+1, min(4, len(numeric_cols))):
                recommendations.append({
                    'type': 'scatter',
                    'columns': [numeric_cols[i], numeric_cols[j]],
                    'title': f'{numeric_cols[j]} vs {numeric_cols[i]}',
                    'reason': f'Visualize relationship between {numeric_cols[i]} and {numeric_cols[j]}'
                })
    
    # Recommend bar charts for categorical columns
    if categorical_cols:
        for col in categorical_cols[:3]:
            if len(data[col].unique()) < 15:  # Only if not too many unique values
                recommendations.append({
                    'type': 'bar',
                    'columns': [col],
                    'title': f'Counts of {col}',
                    'reason': f'Shows the frequency of each category in {col}'
                })
    
    # Recommend correlation matrix for numeric data
    if len(numeric_cols) > 2:
        recommendations.append({
            'type': 'correlation',
            'columns': numeric_cols[:6],  # Limit to 6 columns for readability
            'title': 'Correlation Matrix',
            'reason': 'Shows relationships between all numeric variables'
        })
    
    # Use Claude to enhance recommendations
    enhanced_recommendations = enhance_recommendations(data, recommendations)
    
    return enhanced_recommendations

def enhance_recommendations(data, base_recommendations):
    """Use Claude to enhance visualization recommendations"""
    # Sample first 10 rows for context
    sample_data = data.head(10).to_string()
    
    # Prepare base recommendations for prompt
    base_recs_str = "\n".join([f"- {r['type']} for {', '.join(r['columns'])}: {r['reason']}" for r in base_recommendations])
    
    # Create prompt for enhanced recommendations
    prompt = f"""
    I have a dataset with the following characteristics:
    - Shape: {data.shape}
    - Columns: {list(data.columns)}
    - Data types: {data.dtypes.to_dict()}
    
    Here's a sample of the data:
    {sample_data}
    
    Based on an initial analysis, these visualizations were recommended:
    {base_recs_str}
    
    Please review these recommendations and suggest:
    1. Any additional visualizations that would be valuable
    2. Any modifications to the existing recommendations
    3. Which 2-3 visualizations would be MOST insightful for this dataset and why
    
    Be specific and practical in your suggestions.
    """
    
    # Generate enhanced recommendations using Claude
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.1,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Return both the original recommendations and Claude's enhancements
    return {
        'base_recommendations': base_recommendations,
        'enhanced_suggestions': response.content[0].text
    }

def explain_analysis(data, analysis_type, parameters=None):
    """Explain a statistical analysis or visualization in plain language"""
    if data is None or data.empty:
        return "No data to analyze"
    
    parameters = parameters or {}
    
    # Prepare data sample for context
    sample_data = data.head(10).to_string()
    
    # Create prompt for explanation
    prompt = f"""
    I'm analyzing a dataset with the following characteristics:
    - Shape: {data.shape}
    - Columns: {list(data.columns)}
    
    Here's a sample of the data:
    {sample_data}
    
    I want to perform a {analysis_type} analysis with these parameters:
    {parameters}
    
    Please explain:
    1. What this analysis does and what it's used for
    2. How to interpret the results
    3. Potential limitations or assumptions
    4. What insights I might gain from this analysis with this specific dataset
    
    Explain in clear, non-technical terms that a business user would understand.
    """
    
    # Generate explanation using Claude
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.1,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text

# Example usage
if __name__ == "__main__":
    # Simple test case
    test_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['X', 'Y', 'Z', 'X', 'Y']
    })
    
    print("Data Analysis Agent initialized successfully!")
    print(f"Test data shape: {test_data.shape}")