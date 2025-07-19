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


# --- Lifespan Manager (Runs on Server Startup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the SupplyChainOptimizer and loads all data and models on startup.
    """
    logger.info("Server is starting up...")

    # The public URL to your original, rich dataset in GitHub Releases
    DATA_URL = "https://github.com/AAV13/supply_chain_project/releases/download/v1.0.0/preprocessed_supply_chain_data.csv"
    
    # Initialize the optimizer with the URL. It will handle all data processing internally.
    app.state.optimizer = SupplyChainOptimizer(data_source_path_or_url=DATA_URL)

    # Load the raw data and the pre-trained models
    if not app.state.optimizer.load_preprocessed_data():
        raise RuntimeError("Fatal: Failed to load and process data on startup.")

    app.state.optimizer.load_models()

    if not app.state.optimizer.demand_models:
        raise RuntimeError("Fatal: No models were loaded on startup.")

    logger.info("Models and data loaded successfully. API is ready.")
    yield
    logger.info("Server is shutting down.")


# --- Initialize FastAPI App ---
app = FastAPI(
    lifespan=lifespan,
    title="Supply Chain API (Probabilistic)",
    description="An API to get demand forecasts, inventory recommendations, and logistics alerts.",
    version="3.0.0"
)


# --- API Request/Response Models ---
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


# --- API Endpoints ---
@app.post("/recommendations/{category_name}", response_model=ApiResponse)
def get_recommendations(category_name: str, stock_levels: StockLevels):
    """
    Takes current stock levels and a category, then returns recommendations
    and the probabilistic forecast data.
    """
    if not hasattr(app.state, 'optimizer') or app.state.optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer is not ready.")

    try:
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


@app.get("/")
def read_root():
    return {"status": "Supply Chain Optimization project is running."}
