# Data Analysis Agent

An AI-powered data analysis tool leveraging Claude API to analyze datasets, generate insights, and create visualizations.

## Features

- **Data Loading**: Process CSV and Excel files
- **Statistical Analysis**: Generate comprehensive statistics and identify patterns
- **Data Visualization**: Create various chart types (histograms, scatter plots, etc.)
- **AI-Powered Insights**: Get Claude-generated interpretations of data trends
- **Predictive Analytics**: Simple linear regression for basic predictions

## Technical Architecture

This agent uses a modular architecture:
- Core data analysis engine in `app/main.py`
- Streamlit web interface in `interface/web_app.py`
- Claude API for advanced insight generation and interpretation

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YourUsername/DataAnalysisAgent.git
cd DataAnalysisAgent
