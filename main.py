import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
from contextlib import asynccontextmanager
import logging

# Import the SupplyChainOptimizer class from your existing model.py file
# Ensure that model.py is in the same directory as this main.py file.
from model import SupplyChainOptimizer

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 1. Define the Lifespan Context Manager for Startup/Shutdown ---
# This is the new, recommended way to handle startup events.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs on application startup to load the AI agent and models.
    """
    logger.info("Server is starting up...")
    
    # Initialize the optimizer and attach it to the app's state
    # This makes it accessible in our API endpoints.
    app.state.optimizer = SupplyChainOptimizer("preprocessed_supply_chain_data.csv")
    
    # Load the data needed for statistics and model loading
    if not app.state.optimizer.load_preprocessed_data():
        raise RuntimeError("Failed to load preprocessed data on startup.")
    
    # Load the pre-trained models from the 'saved_models' directory
    app.state.optimizer.load_models()
    
    if not app.state.optimizer.demand_models:
        raise RuntimeError("No models were loaded. Ensure models are saved in the 'saved_models' directory.")
        
    logger.info("Loaded successfully and is ready to serve requests.")
    
    yield
    
    # Code below `yield` would run on shutdown (e.g., for cleanup)
    logger.info("Server is shutting down.")


# --- 2. Initialize the FastAPI App with the Lifespan Manager ---
app = FastAPI(
    lifespan=lifespan,
    title="Supply Chain API",
    description="An API to get demand forecasts and inventory recommendations.",
    version="1.0.0"
)


# --- 3. Define the API Request and Response Models ---
# Pydantic models ensure that the data sent to our API is in the correct format.
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

class ApiResponse(BaseModel):
    inventory_recommendations: List[str]
    strategic_alerts: List[str]


# --- 4. Create the API Endpoint ---
@app.post("/recommendations/", response_model=ApiResponse)
def get_recommendations(stock_levels: StockLevels):
    """
    This is the main API endpoint. It takes the current stock levels
    and returns a set of AI-driven recommendations.
    """
    # Access the optimizer from the application state
    if not hasattr(app.state, 'optimizer') or app.state.optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer is not ready. Please wait a moment and try again.")
    
    try:
        # Get the two types of recommendations from our agent
        inventory_recs = app.state.optimizer.generate_inventory_recommendations(stock_levels.current_stock)
        logistics_alerts = app.state.optimizer.generate_logistics_alerts()
        
        # Return them in a structured JSON response
        return {
            "inventory_recommendations": inventory_recs,
            "strategic_alerts": logistics_alerts
        }
    except Exception as e:
        # If anything goes wrong, return a server error
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


# --- 5. Add a Root Endpoint for Health Checks ---
@app.get("/")
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"status": "Supply Chain AI Agent is running."}

# --- How to Run This Server ---
# 1. Make sure you have the required libraries installed:
#    pip install fastapi "uvicorn[standard]"
#
# 2. Save this code as `main.py`.
#
# 3. In your terminal, run the following command from the same directory:
#    uvicorn main:app --reload
#
# 4. Open your web browser and go to http://127.0.0.1:8000/docs
#    This will open the interactive API documentation (Swagger UI).
#
# 5. From the documentation, you can test the `/recommendations/` endpoint
#    by providing a JSON object with your current stock levels.
