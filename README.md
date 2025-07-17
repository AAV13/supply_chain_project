# Predictive API for Supply Chain Optimization

## Overview

This project builds and deploys a high-performance API that serves as the predictive engine for an AI-powered supply chain agent. The API ingests historical sales data, forecasts future demand for product categories, and provides two layers of actionable recommendations.

This service is designed to be called by automation platforms like **n8n**, Zapier, or custom scripts to create a true, end-to-end AI agent that can make automated, data-driven decisions.

The API provides two types of insights:

1.  **Tactical Inventory Alerts**: Recommends when and how much inventory to reorder to prevent stockouts while minimizing costs.
2.  **Strategic Logistics Alerts**: Identifies high-risk or high-cost product categories and suggests strategic changes to improve performance.

The entire system is deployed as a robust FastAPI web service, making these insights available via a simple API call.

## Features

* **Automated Data Preprocessing**: A standalone script cleans, validates, and transforms raw transactional data into a model-ready format.
* **Multi-Model Demand Forecasting**: Trains a separate, specialized `Prophet` model for each product category. The model is configured to automatically detect yearly and weekly seasonality with a multiplicative effect.
* **Hyperparameter Tuning**: Uses `Optuna` to automatically find the best model settings for each category, maximizing forecast accuracy.
* **MLOps Experiment Tracking**: Integrates with `MLflow` to log all training parameters, performance metrics, and model artifacts for reproducibility.
* **Intelligent Inventory Optimization**:
    * Calculates a dynamic **Reorder Point (ROP)** to know *when* to order.
    * Calculates the **Economic Order Quantity (EOQ)** to know *how much* to order cost-effectively.
* **Strategic Logistics Insights**: Analyzes historical data to flag categories with high late delivery risk or excessive holding costs.
* **API Deployment**: The entire system is wrapped in a `FastAPI` application, ready for production deployment.

## Tech Stack

* **Backend**: Python, FastAPI
* **Forecasting**: Prophet (fbprophet)
* **Optimization**: Optuna, Scikit-learn
* **MLOps**: MLflow
* **Data Handling**: Pandas, NumPy
* **Web Server**: Uvicorn, Gunicorn
* **Deployment**: Render

## Project Structure

```
/
|-- saved_models/         # Directory where trained .pkl models are saved
|-- .gitignore            # Specifies files for Git to ignore
|-- main.py               # FastAPI server application
|-- model.py              # The core SupplyChainOptimizer class and logic
|-- preprocess.py         # The standalone data cleaning and feature engineering script
|-- preprocessed_supply_chain_data.csv  # The output of the preprocessing script
|-- requirements.txt      # Project dependencies
|-- README.md             # This file
```

## Setup and Installation

Follow these steps to set up and run the project locally.

**1. Clone the Repository:**

```
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

**2. Create and Activate a Virtual Environment:**

```
# Create the environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate
```

**3. Install Dependencies:**
Install all required packages from the `requirements.txt` file.

```
pip install -r requirements.txt
```

## Running the Application

The pipeline runs in three stages: preprocessing, model training, and serving the API.

**Step 1: Run the Preprocessing Script**
First, you need to generate the clean data file. (You only need to do this once).
