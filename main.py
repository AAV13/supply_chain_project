import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List
from contextlib import asynccontextmanager
import logging
import uuid
import redis
import os
import json

# Import the SupplyChainOptimizer class from your existing model.py file
from model import SupplyChainOptimizer

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 1. Define the Lifespan Context Manager for Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs on application startup to load the AI agent, models,
    and connect to the Redis database.
    """
    logger.info("Server is starting up. Loading AI agent and connecting to Redis...")
    
    # --- Connect to Redis ---
    # Render or another provider sets the REDIS_URL environment variable
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise RuntimeError("REDIS_URL environment variable not set. Cannot connect to Redis.")
    
    try:
        app.state.redis = redis.from_url(redis_url, decode_responses=True)
        app.state.redis.ping() # Check the connection
        logger.info("Successfully connected to Redis.")
    except redis.exceptions.ConnectionError as e:
        raise RuntimeError(f"Failed to connect to Redis: {e}")

    # --- Load AI Agent ---
    app.state.optimizer = SupplyChainOptimizer("preprocessed_supply_chain_data.csv")
    if not app.state.optimizer.load_preprocessed_data():
        raise RuntimeError("Failed to load preprocessed data on startup.")
    app.state.optimizer.load_models()
    if not app.state.optimizer.demand_models:
        raise RuntimeError("No models were loaded.")
        
    logger.info("AI Agent loaded successfully.")
    
    yield
    
    # --- Cleanup on Shutdown ---
    app.state.redis.close()
    logger.info("Redis connection closed. Server is shutting down.")


# --- 2. Initialize the FastAPI App with the Lifespan Manager ---
app = FastAPI(
    lifespan=lifespan,
    title="Supply Chain AI Agent API",
    description="An API to get demand forecasts and inventory recommendations.",
    version="1.0.0"
)


# --- 3. Define the Background Task Function ---
def run_optimization_task(task_id: str, current_stock: Dict[str, float]):
    """
    This function contains the heavy computation and is run in the background.
    It now uses Redis to store the results.
    """
    logger.info(f"Starting background task: {task_id}")
    try:
        optimizer = app.state.optimizer
        redis_client = app.state.redis
        
        inventory_recs = optimizer.generate_inventory_recommendations(current_stock)
        logistics_alerts = optimizer.generate_logistics_alerts()
        
        result_data = {
            "status": "completed",
            "inventory_recommendations": inventory_recs,
            "strategic_alerts": logistics_alerts
        }
        # Store the JSON string in Redis and set it to expire after 1 hour
        redis_client.set(task_id, json.dumps(result_data), ex=3600)
        
        logger.info(f"Background task {task_id} completed successfully.")
        
    except Exception as e:
        logger.error(f"Background task {task_id} failed: {e}")
        error_data = {"status": "failed", "error": str(e)}
        app.state.redis.set(task_id, json.dumps(error_data), ex=3600)


# --- 4. Define the API Request and Response Models ---
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


# --- 5. Create the API Endpoints ---
@app.post("/recommendations/", response_model=TaskResponse, status_code=202)
def start_recommendations_task(stock_levels: StockLevels, background_tasks: BackgroundTasks):
    """
    Accepts a request and starts a background task. Responds immediately with a task ID.
    """
    task_id = str(uuid.uuid4())
    # Set initial status in Redis
    app.state.redis.set(task_id, json.dumps({"status": "processing"}), ex=3600)
    
    background_tasks.add_task(run_optimization_task, task_id, stock_levels.current_stock)
    
    return {"task_id": task_id, "status": "processing"}


@app.get("/results/{task_id}", response_model=ApiResponse)
def get_task_results(task_id: str):
    """
    Fetches the results of a background task from Redis using its task ID.
    """
    result_json = app.state.redis.get(task_id)
    
    if not result_json:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    
    result = json.loads(result_json)
    
    if result["status"] == "processing":
        raise HTTPException(status_code=202, detail="Task is still processing.")
        
    if result["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Task failed: {result.get('error')}")

    return result


@app.get("/")
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"status": "Supply Chain AI Agent is running."}
