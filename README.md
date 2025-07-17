# Supply Chain Optimization ML Pipeline

This project implements an automated supply chain optimization system that provides demand forecasting, inventory optimization, and actionable recommendations.

## Features

- Time-series demand forecasting using Facebook Prophet
- Inventory level optimization
- MLflow experiment tracking
- FastAPI REST API for predictions
- Automated recommendations generation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python supply_chain_model.py
```

3. Start the API server:
```bash
python api.py
```

The API will be available at http://localhost:8000

## API Endpoints

### POST /forecast
Get demand forecast and inventory recommendations

Request body:
```json
{
    "forecast_days": 30,
    "holding_cost_rate": 0.1
}
```

Response:
```json
{
    "inventory_levels": {
        "2023-11-01": {
            "predicted_demand": 100,
            "optimal_inventory": 120,
            "safety_stock": 20
        }
        // ... more dates
    },
    "recommendations": [
        "High demand alert for 2023-11-01: Consider increasing inventory to 120 units"
        // ... more recommendations
    ]
}
```

### GET /health
Health check endpoint

## Model Details

The system uses Facebook Prophet for time-series forecasting with the following features:
- Yearly, weekly seasonality
- Multiplicative seasonality mode
- Automatic changepoint detection
- Uncertainty intervals for safety stock calculation

## Monitoring

The system uses MLflow to track:
- Model parameters
- Performance metrics (MAPE, RMSE)
- Forecast plots
- Training runs

## Future Improvements

1. Implement A/B testing framework
2. Add route optimization
3. Enhance inventory optimization with more sophisticated algorithms
4. Add real-time monitoring dashboard
5. Implement automated model retraining 