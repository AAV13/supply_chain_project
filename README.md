# Predictive API for Supply Chain Optimization

## Overview

This project provides a RESTful API for forecasting product demand across various categories. It uses historical supply chain data to train individual Random Forest models for each product category and serves predictions through a FastAPI backend.

This service is designed to be called by automation platforms like **n8n** to create a true, end-to-end AI agent that can make automated, data-driven decisions.

The API provides two types of insights:

1.  **Tactical Inventory Alerts**: Recommends when and how much inventory to reorder to prevent stockouts while minimizing costs.
2.  **Strategic Logistics Alerts**: Identifies high-risk or high-cost product categories and suggests strategic changes to improve performance.

The primary goal of this project is to predict future product demand based on historical data points like sales, price, and stock levels. By training a separate regression model for each product category, the API can deliver tailored forecasts, which is essential for optimizing inventory, logistics, and overall supply chain efficiency. The application is built with FastAPI, ensuring high performance and automatic interactive documentation.

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
* **Deployment**: Railway, and also tried Render

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
```
# Assuming your preprocessing code is in 'preprocess.py'
python preprocess.py
```
This will create the `preprocessed_supply_chain_data.csv` file.

**Step 2: Train the Models**
Next, run the main modeling script. This will train a model for each category and save the trained models to the `saved_models/` directory.
```
python model.py
```

**Step 3: Run the FastAPI Server**
Finally, start the web server to serve the recommendations via the API.
```
uvicorn main:app --reload
```

## API Usage

Once the server is running, you can access the interactive API documentation (Swagger UI) in your browser at:

[**http://127.0.0.1:8000/docs**](http://127.0.0.1:8000/docs)

From this page, you can use the `/recommendations/` endpoint to get live inventory and strategic alerts by providing your current stock levels in a JSON format. This API is designed to be called by an automation tool like n8n to complete the AI agent workflow.
