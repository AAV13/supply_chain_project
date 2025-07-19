import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
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
    title="Supply Chain API (Probabilistic)",
    description="An API to get demand forecasts (including uncertainty) and inventory recommendations.",
    version="2.1.0" # Version bump
)


# --- Request/Response Models ---
class StockLevels(BaseModel):
    current_stock: Dict[str, float] = Field(..., example={'Fishing': 3000.0, 'Cleats': 25000.0})

class ForecastData(BaseModel):
    dates: List[str]
    forecast_yhat: List[float]
    forecast_yhat_lower: List[float]
    forecast_yhat_upper: List[float]

class ApiResponse(BaseModel):
    inventory_recommendations: List[str]
    strategic_alerts: List[str]
    probabilistic_forecast: Optional[ForecastData] = None


# --- Recommendation Endpoint ---
@app.post("/recommendations/{category_name}", response_model=ApiResponse)
def get_recommendations(category_name: str, stock_levels: StockLevels):
    """
    Takes stock levels and a category, then returns recommendations
    and the probabilistic forecast data.
    """
    if not hasattr(app.state, 'optimizer') or app.state.optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer is not ready.")

    try:
        # Get the fully processed output from the optimizer
        recs, alerts, forecast_dict = app.state.optimizer.get_processed_recommendations(
            current_stock=stock_levels.current_stock,
            category_name=category_name,
            forecast_days=30
        )
        
        return ApiResponse(
            inventory_recommendations=recs,
            strategic_alerts=alerts,
            probabilistic_forecast=forecast_dict
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Category '{category_name}' not found.")
    except Exception as e:
        logger.error(f"Error processing '{category_name}': {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


# --- Health Check Endpoint ---
@app.get("/")
def read_root():
    return {"status": "Supply Chain Optimization project is running."}
