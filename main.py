import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
from contextlib import asynccontextmanager
import logging

# Import your SupplyChainOptimizer logic
from model import SupplyChainOptimizer

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the SupplyChainOptimizer and load models on startup."""
    logger.info("Server is starting up...")

    # The correct, permanent URL to your data file
    DATA_URL = "https://github.com/AAV13/supply_chain_project/releases/download/v1.0.0/preprocessed_supply_chain_data.csv"
    
    app.state.optimizer = SupplyChainOptimizer(preprocessed_data_url=DATA_URL)

    if not app.state.optimizer.load_preprocessed_data():
        raise RuntimeError("Failed to load preprocessed data.")

    app.state.optimizer.load_models()

    if not app.state.optimizer.demand_models:
        raise RuntimeError("No models were loaded.")

    logger.info("Models loaded and API is ready.")
    yield
    logger.info("Server is shutting down.")


# --- Initialize FastAPI App ---
app = FastAPI(
    lifespan=lifespan,
    title="Supply Chain API",
    description="An API to get demand forecasts and inventory recommendations for a specific category.",
    version="1.1.0"
)


# --- Request/Response Models ---
class StockLevels(BaseModel):
    current_stock: Dict[str, float] = Field(
        ..., 
        example={
            'Accessories': 10.0, 'Camping & Hiking': 500.0, 'Cardio Equipment': 20.0,
            'Cleats': 25000.0, 'Electronics': 9000.0, 'Fishing': 3000.0,
            "Girls' Apparel": 18000.0, 'Golf Balls': 5000.0, 'Golf Gloves': 5.0,
            'Indoor/Outdoor Games': 8000.0, "Men's Footwear": 10000.0, 'Shop By Sport': 12000.0,
            'Sporting Goods': 15000.0, 'Water Sports': 4000.0, "Women's Apparel": 40000.0
        }
    )

class ApiResponse(BaseModel):
    inventory_recommendations: List[str]
    strategic_alerts: List[str]


# --- Recommendation Endpoint (Now processes one category at a time) ---
@app.post("/recommendations/{category_name}", response_model=ApiResponse)
def get_recommendations(category_name: str, stock_levels: StockLevels):
    """
    Takes current stock levels and a single category name,
    then returns AI-driven recommendations.
    """
    if not hasattr(app.state, 'optimizer') or app.state.optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer is not ready.")

    try:
        # Pass the specific category_name to the functions
        inventory_recs = app.state.optimizer.generate_inventory_recommendations(
            stock_levels.current_stock,
            category_name
        )
        logistics_alerts = app.state.optimizer.generate_logistics_alerts_for_category(
            category_name
        )

        return {
            "inventory_recommendations": inventory_recs,
            "strategic_alerts": logistics_alerts
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Category '{category_name}' not found.")
    except Exception as e:
        logger.error(f"Error processing '{category_name}': {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


# --- Health Check Endpoint ---
@app.get("/")
def read_root():
    return {"status": "Supply Chain Optimization project is running."}


# --- Run Command ---
# Run with: uvicorn main:app --reload
