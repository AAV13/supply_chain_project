# AI-Powered Supply Chain Optimization & Automation

## Overview
This project is a comprehensive, AI-driven solution for intelligent inventory management. It combines a **FastAPI-powered recommendation engine** with a sophisticated **n8n automation workflow** to provide real-time analysis, strategic alerts, and automated reporting directly within a Google Sheet.

The system analyzes inventory levels, predicts reorder points, provides strategic alerts about logistical risks, and uses a Large Language Model (Google Gemini) to deliver human-readable summaries and action plans.

## ✨ Key Features

* **RESTful API:** A robust API built with **FastAPI** that provides inventory recommendations and strategic alerts based on historical data.
* **Automated Data Processing:** An **n8n workflow** runs on a schedule to automatically fetch data from a Google Sheet.
* **Intelligent Analysis:** The system uses **Google Gemini** to interpret the raw API data and generate concise, actionable summaries.
* **Proactive Email Alerts:** Automatically sends email notifications via **Gmail** for inventory items that require immediate action (e.g., reordering).
* **Automated Reporting:** Updates the Google Sheet with the AI-generated summary for each item, creating a self-updating dashboard.
* **Cloud Deployment:** The FastAPI application is deployed and accessible via **Railway**.

The project consists of two main components that work together: the FastAPI backend and the n8n automation workflow.

1.  **FastAPI Backend:**
    * Accepts a product category and its current stock level.
    * Analyzes the data to produce `inventory_recommendations` and `strategic_alerts`.
    * Hosted on Railway for reliable, 24/7 access.

2.  **n8n Workflow:**
    * **Trigger:** Runs automatically on a daily schedule.
    * **Read Data:** Fetches all product rows from a designated **Google Sheet**.
    * **Call API:** For each product, it calls the FastAPI endpoint to get a recommendation.
    * **AI Summarization:** Sends the API's JSON response to **Google Gemini** to generate a human-friendly summary.
    * **Conditional Alerting:** An `IF` node checks if the summary contains "Action Required:". If true, it sends a detailed alert via **Gmail**.
    * **Update Sheet:** Writes the AI summary back into the correct row in the Google Sheet, closing the loop.

### 2. The n8n Workflow

To replicate the automation, you would need to build the workflow as described in the architecture section. This involves:
1.  Setting up a Google Sheet with your inventory data (`category_name`, `current_stock`, etc.).
2.  Creating an n8n workflow with the required nodes (`Schedule Trigger`, `Google Sheets`, `Code`, `HTTP Request`, `Basic LLM Chain`, `IF`, `Gmail`).
3.  Configuring credentials for Google Sheets, Google Gemini, and Gmail within n8n.
4.  Using the expressions and logic developed in this project to connect the nodes and process the data.


## Github Repository 
This repository provides a RESTful API for forecasting product demand across various categories. It uses historical supply chain data to train individual Random Forest models for each product category and serves predictions through a FastAPI backend.

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
