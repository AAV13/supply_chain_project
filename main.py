import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List
from contextlib import asynccontextmanager
import logging
import uuid

# Import the SupplyChainOptimizer class from your existing model.py file
# Ensure that model.py is in the same directory as this main.py file.
from model import SupplyChainOptimizer

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 1. Define the Lifespan Context Manager for Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs on application startup to load the AI agent and models.
    """
    logger.info("Server is starting up. Loading AI agent...")
    
    # Initialize the optimizer and attach it to the app's state
    app.state.optimizer = SupplyChainOptimizer("preprocessed_supply_chain_data.csv")
    
    # Load the data needed for statistics and model loading
    if not app.state.optimizer.load_preprocessed_data():
        raise RuntimeError("Failed to load preprocessed data on startup.")
    
    # Load the pre-trained models from the 'saved_models' directory
    app.state.optimizer.load_models()
    
    if not app.state.optimizer.demand_models:
        raise RuntimeError("No models were loaded. Ensure models are saved in the 'saved_models' directory.")
        
    logger.info("AI Agent loaded successfully and is ready to serve requests.")
    
    yield
    
    logger.info("Server is shutting down.")


# --- 2. Initialize the FastAPI App with the Lifespan Manager ---
app = FastAPI(
    lifespan=lifespan,
    title="Supply Chain AI Agent API",
    description="An API to get demand forecasts and inventory recommendations.",
    version="1.0.0"
)

# --- 3. In-Memory Storage for Task Results ---
# In a production system, you would use a more robust solution like Redis or a database.
task_results: Dict[str, Dict] = {}


# --- 4. Define the Background Task Function ---
def run_optimization_task(task_id: str, current_stock: Dict[str, float]):
    """
    This function contains the heavy computation and is run in the background.
    """
    logger.info(f"Starting background task: {task_id}")
    try:
        # Access the globally loaded optimizer
        optimizer = app.state.optimizer
        
        # Perform the heavy calculations
        inventory_recs = optimizer.generate_inventory_recommendations(current_stock)
        logistics_alerts = optimizer.generate_logistics_alerts()
        
        # Store the results in our global dictionary
        task_results[task_id] = {
            "status": "completed",
            "inventory_recommendations": inventory_recs,
            "strategic_alerts": logistics_alerts
        }
        logger.info(f"Background task {task_id} completed successfully.")
        
    except Exception as e:
        logger.error(f"Background task {task_id} failed: {e}")
        task_results[task_id] = {"status": "failed", "error": str(e)}


# --- 5. Define the API Request and Response Models ---
class StockLevels(BaseModel):
    current_stock: Dict[str, float] = Field(
        ..., 
        example={
            'Sporting Goods': 15000.0,
            'Cleats': 25000.0,
            "Women's Apparel": 40000.0,
            "Men's Footwear": 10000.0,
            "Camping & Hiking": 500.0,
            "Accessories": 10.0,
            "Golf Gloves": 5.0,
            'Indoor/Outdoor Games': 8000.0,
            'Shop By Sport': 12000.0,
            'Fishing': 3000.0,
            'Cardio Equipment': 20.0,
            'Water Sports': 4000.0,
            'Electronics': 9000.0,
            'Girls\' Apparel': 18000.0,
            'Golf Balls': 5000.0
        }
    )

class TaskResponse(BaseModel):
    task_id: str
    status: str

class ApiResponse(BaseModel):
    status: str
    inventory_recommendations: List[str]
    strategic_alerts: List[str]


# --- 6. Create the API Endpoints ---
@app.post("/recommendations/", response_model=TaskResponse, status_code=202)
def start_recommendations_task(stock_levels: StockLevels, background_tasks: BackgroundTasks):
    """
    Accepts a request to generate recommendations and starts a background task.
    Responds immediately with a task ID.
    """

    # Generate a unique ID for this task
    task_id = str(uuid.uuid4())
    task_results[task_id] = {"status": "processing"}
    
    # Add the heavy computation to the background tasks
    background_tasks.add_task(run_optimization_task, task_id, stock_levels.current_stock)
    
    # Ensure the response contains the task_id and status.
    return {"task_id": task_id, "status": "processing"}


@app.get("/results/{task_id}", response_model=ApiResponse)
def get_task_results(task_id: str):
    """
    Fetches the results of a background task using its task ID.
    """
    result = task_results.get(task_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    
    if result["status"] == "processing":
        raise HTTPException(status_code=202, detail="Task is still processing. Please try again in a moment.")
        
    if result["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Task failed: {result.get('error')}")

    return result


@app.get("/")
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"status": "Supply Chain AI Agent is running."}
