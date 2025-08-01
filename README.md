# AI-Powered Supply Chain Optimization & Automation

## Overview
This project is a comprehensive, AI-driven solution for intelligent inventory management. It combines a **FastAPI-powered recommendation engine** with a sophisticated **n8n automation workflow** to provide real-time analysis, strategic alerts, and automated reporting directly within a Google Sheet.

The system analyzes inventory levels, predicts reorder points, provides strategic alerts about logistical risks, and uses a Large Language Model (Google Gemini) to deliver human-readable summaries and action plans.

## Key Features
### API & Forecasting Engine
* **Multi-Model Demand Forecasting:** Trains a separate, specialized `Prophet` model for each product category.
* **Hyperparameter Tuning:** Uses `Optuna` to automatically find the best model settings for each category, maximizing forecast accuracy.
* **MLOps Experiment Tracking:** Integrates with `MLflow` to log all training parameters, metrics, and model artifacts for reproducibility.
* **Intelligent Inventory Optimization:**
    * Calculates a dynamic **Reorder Point (ROP)** to know *when* to order.
    * Calculates the **Economic Order Quantity (EOQ)** to know *how much* to order cost-effectively.
* **Strategic Logistics Insights:** Analyzes historical data to flag categories with high late delivery risk or excessive holding costs.
* **API Deployment**: The entire system is wrapped in a `FastAPI` application, ready for production deployment.

### n8n Automation Workflow
* **Automated Data Processing:** Runs on a schedule to automatically process every item in a Google Sheet without manual intervention.
* **Intelligent Analysis:** Leverages **Google Gemini** to translate complex JSON data into clear, actionable business insights.
* **Proactive Email Alerts:** Automatically notifies stakeholders via **Gmail** about urgent inventory issues that require immediate attention.
* **Automated Reporting:** Creates a "living" report by constantly updating the Google Sheet with the latest AI-generated summaries.

The project consists of two main components that work together: the FastAPI backend and the n8n automation workflow.

1.  **FastAPI Backend:**
    * Accepts a product category and its current stock level.
    * Analyzes the data to produce `inventory_recommendations` and `strategic_alerts`.
    * Hosted on Railway for reliable, 24/7 access.
  
<img width="956" height="427" alt="webpage expanded" src="https://github.com/user-attachments/assets/3757f553-db97-4b61-b70b-8060ee62e02f" />   

2.  **n8n Workflow:**
    * **Trigger:** Runs automatically on a monthly schedule.
    * **Read Data:** Fetches all product rows from a designated Google Sheet with the inventory data.
    * **Call API:** For each product, it calls the FastAPI endpoint to get a recommendation.
    * **AI Summarization:** Sends the API's JSON response to and **LLM** (Google Gemini) to generate a human-friendly summary.
    * **Conditional Alerting:** An `IF` node checks if the summary contains "Action Required:". These are urgent so, it sends a detailed alert via email.
    * **Update Sheet:** Writes the AI summary back into the correct row in the Google Sheet, closing the loop.

<img width="788" height="239" alt="workflow" src="https://github.com/user-attachments/assets/82550ec4-9786-4053-92a3-94f7145f2718" />

## Tech Stack

* **Backend**: Python, FastAPI
* **Forecasting**: Prophet (fbprophet)
* **Hyperparameter Tuning**: Optuna, Scikit-learn
* **MLOps**: MLflow
* **Data Handling**: Pandas, NumPy
* **Web Server**: Uvicorn, Gunicorn
* **Deployment**: Railway, and also tried Render  

### Github Repository 
This repository provides a RESTful API for forecasting product demand across various categories. It uses historical supply chain data to train individual Random Forest models for each product category and serves predictions through a FastAPI backend.

This service is designed to be called by automation platforms like **n8n** to create a true, end-to-end AI agent that can make automated, data-driven decisions.

The API provides two types of insights:

1.  **Tactical Inventory Alerts**: Recommends when and how much inventory to reorder to prevent stockouts while minimizing costs.
2.  **Strategic Logistics Alerts**: Identifies high-risk or high-cost product categories and suggests strategic changes to improve performance.

The primary goal of this project is to predict future product demand based on historical data points like sales, price, and stock levels. By training a separate regression model for each product category, the API can deliver tailored forecasts, which is essential for optimizing inventory, logistics, and overall supply chain efficiency. The application is built with FastAPI, ensuring high performance and automatic interactive documentation.

The entire system is deployed as a robust FastAPI web service, making these insights available via a simple API call.

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
Once the server is running, you can access the interactive API documentation (Swagger UI) in your browser at: http://127.0.0.1:8000/docs

The production API is also live on Railway: https://www.google.com/search?q=https://supplychainproject-production.up.railway.app/docs

POST /recommendations/{category_name}
Provides inventory and strategic analysis for a given product category.

URL Params:

category_name (string, required): The URL-encoded name of the product category.

Request Body:

JSON

{
  "current_stock": {
    "Category Name": 1234
  }
}
Success Response (200):

JSON

{
  "inventory_recommendations": [
    "REORDER ALERT for 'Accessories': Current stock (45) is below reorder point (90). Recommended order quantity (EOQ): 247 units."
  ],
  "strategic_alerts": [
    "STRATEGIC ALERT for 'Accessories': High late delivery risk detected (57.0%)..."
  ],
  "probabilistic_forecast": null
}
