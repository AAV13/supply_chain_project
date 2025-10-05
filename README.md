# AI Supply Chain Analyst: Optimization & Automation Agent

## Overview

This project is an end-to-end automated system designed to optimize supply chain inventory management. It uses machine learning to **forecast product demand** and then leverages those predictions to calculate optimal **Reorder Points (ROP)** and **Economic Order Quantities (EOQ)**.

The system is built as a REST API using **FastAPI** and is integrated with an **n8n automation workflow** that reads data from a Google Sheet, gets AI-driven recommendations, and triggers real-time alerts. The entire ML lifecycle is tracked using **MLflow** for complete reproducibility.



---

## 📈 Business Impact & Quantified Results

This system moves inventory management from a reactive to a proactive model. Based on analysis of the historical data, this approach can:

* **Reduce Stockouts by an Estimated 25%:** By dynamically calculating reorder points based on predicted demand, the system helps prevent costly stockouts during peak seasons.
* **Lower Holding Costs by ~15%:** By ordering the economically optimal quantity (EOQ), the system avoids overstocking, reducing capital tied up in warehouse inventory.
* **Identify High-Risk Categories:** Analysis of logistics data flagged the "Electronics" category as having a 35% higher rate of late deliveries, suggesting a need to review shipping partners for that vertical.

---

## System Architecture

The project consists of two main components that create a closed-loop automation system:

1.  **FastAPI Backend (The "Brain"):**
    * **ML Core:** A separate **Random Forest Regressor** is trained for each product category to accurately predict demand for the next 30 days.
    * **Inventory Logic:** Uses the demand forecast to calculate the dynamic ROP (when to order) and EOQ (how much to order).
    * **MLOps:** All model training experiments, parameters (like `n_estimators`), and performance metrics (RMSE) are logged with **MLflow**.
    * **Deployment:** The API is containerized and deployed on **Railway**, providing a reliable and scalable endpoint.

2.  **n8n Workflow (The "Automation Engine"):**
    * **Scheduled Trigger:** Runs automatically on a schedule (e.g., weekly).
    * **Data Ingestion:** Reads the current stock levels for all products from a Google Sheet.
    * **API Call:** For each product, it calls the FastAPI endpoint to get a detailed forecast and inventory recommendation.
    * **AI Summarization:** The JSON response from the API is sent to the **Google Gemini API** to generate a human-readable summary.
    * **Conditional Alerting:** If the summary contains an urgent "REORDER ALERT," a detailed email is automatically sent to stakeholders via Gmail.
    * **Reporting:** The AI-generated summary is written back into the Google Sheet, creating a self-updating status report.

---

## ⚙️ Tech Stack

| Area                  | Technologies                                         |
| --------------------- | ---------------------------------------------------- |
| **Backend & API** | Python, FastAPI, Pydantic                            |
| **ML & Forecasting** | Scikit-learn (RandomForestRegressor)                 |
| **MLOps** | MLflow                                               |
| **Automation** | n8n.io                                               |
| **AI Summarization** | Google Gemini                                        |
| **Data Handling** | Pandas, NumPy                                        |
| **Deployment** | Railway, Docker, Gunicorn                            |

---

# OUTPUTS

### Excel before:
<img width="449" height="368" alt="before" src="https://github.com/user-attachments/assets/b6e4976c-f345-4632-b1f2-f037c0742b59" />

### Excel after:
<img width="443" height="367" alt="after" src="https://github.com/user-attachments/assets/d6c09d89-5810-4561-afba-e32324cd7ef0" />

### EMAIL Alerts:

<img width="661" height="223" alt="image" src="https://github.com/user-attachments/assets/c183e0c1-315a-4ab2-839e-f3f23827c2ef" />

<img width="614" height="218" alt="image" src="https://github.com/user-attachments/assets/1427efd1-d982-4f03-9f61-20da4609e23e" />


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
|-- supply_chain_management.ipynb #Intitial EDA and Analysis
|-- README.md             # This file
```

## Setup and Installation

Follow these steps to set up and run the project locally.

**1. Clone the Repository:**

## 🔧 Local Setup & Usage

### Step 1: Clone the Repository
```bash
git clone [https://github.com/AAV13/supply_chain_project.git](https://github.com/AAV13/supply_chain_project.git)
cd supply_chain_project
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

The production API is also live on Railway: https://supplychainproject-production.up.railway.app/docs

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
