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
            
        elif viz_type == 'line':
            if len(columns) == 1:
                plt.plot(data.index, data[columns[0]])
                plt.ylabel(columns[0])
            else:
                for col in columns:
                    plt.plot(data.index, data[col], label=col)
                plt.legend()
            plt.xlabel('Index')
            title = title or f'Line Chart of {", ".join(columns)}'
            
        elif viz_type == 'pie':
            if len(columns) != 1:
                return None, "Pie chart requires exactly one categorical column"
            counts = data[columns[0]].value_counts()
            if len(counts) > 10:
                # Too many categories, use top 10
                counts = counts.head(10)
                title = title or f'Top 10 Categories in {columns[0]}'
            else:
                title = title or f'Pie Chart of {columns[0]}'
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
            
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

def predict_values(data, target_column, feature_columns):
    """Perform simple prediction using linear regression"""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Prepare data
    X = data[feature_columns]
    y = data[target_column]
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get feature importance
    coefficients = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    
    # Generate human-readable insights from the model
    model_insights = analyze_model(target_column, coefficients, mse, r2)
    
    return {
        'mse': mse,
        'r2': r2,
        'coefficients': coefficients.to_dict('records'),
        'model_insights': model_insights
    }

def analyze_model(target, coefficients, mse, r2):
    """Generate human-readable insights from model results using Claude"""
    coeff_str = "\n".join([f"- {row['Feature']}: {row['Coefficient']}" for row in coefficients.to_dict('records')])
    
    prompt = f"""
    I've built a linear regression model to predict {target} with the following results:
    
    Mean Squared Error: {mse}
    R² Score: {r2}
    
    Feature coefficients:
    {coeff_str}
    
    Please interpret these results in plain language, explaining:
    1. How well the model predicts {target} (based on the R² score)
    2. Which features are most important for predicting {target}
    3. The meaning of the key coefficients (whether they increase or decrease the predicted value)
    4. Suggestions for improving the model
    
    Keep your explanation concise and understandable for a non-technical audience.
    """
    
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.1,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text

# Example usage if run directly
if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['X', 'Y', 'Z', 'X', 'Y']
    })
    print("Data Analysis Agent initialized successfully!")
    print(f"Sample data shape: {sample_data.shape}")
