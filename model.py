import pandas as pd
import numpy as np
from prophet import Prophet
import mlflow
import optuna
from typing import Dict, List, Any
import logging
import warnings
import re
from pathlib import Path
import pickle

# --- Setup ---
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sanitize_for_mlflow(name: str) -> str:
    sanitized_name = name.replace('/', '_')
    return re.sub(r"[^a-zA-Z0-9_.\s:-]", "", sanitized_name)

class SupplyChainOptimizer:
    def __init__(self, preprocessed_data_url: str, models_dir: str = "saved_models"):
        self.preprocessed_data_url = preprocessed_data_url
        self.models_dir = Path(models_dir)
        self.data: pd.DataFrame = None
        self.demand_models: Dict[str, Prophet] = {}
        self.category_demands: Dict[str, pd.DataFrame] = {}
        self.category_stats: pd.DataFrame = None
        self.shipping_mode_stats: pd.DataFrame = None
        self.final_metrics: Dict[str, Dict[str, Any]] = {}
        self.models_dir.mkdir(exist_ok=True)

    def load_preprocessed_data(self) -> bool:
        logger.info(f"Loading preprocessed data from URL...")
        try:
            self.data = pd.read_csv(self.preprocessed_data_url, parse_dates=['order_date'])
        except Exception as e:
            logger.error(f"Failed to load data from URL: {self.preprocessed_data_url}. Error: {e}")
            return False

        self.category_stats = self.data.groupby('category_name').agg(
            order_quantity=('order_quantity', 'sum'),
            avg_unit_price=('order_total', 'sum'),
            late_delivery_risk=('late_delivery_risk', 'mean') 
        )
        self.category_stats['avg_unit_price'] = self.category_stats.apply(
            lambda row: row['avg_unit_price'] / row['order_quantity'] if row['order_quantity'] > 0 else 0, axis=1
        )
        self.shipping_mode_stats = self.data.groupby(['category_name', 'shipping_mode']).agg(
            late_delivery_risk=('late_delivery_risk', 'mean')
        ).reset_index()

        for category in self.data['category_name'].unique():
            category_data = self.data[self.data['category_name'] == category]
            daily_agg = category_data.groupby('order_date').agg(
                y=('order_quantity', 'sum')
            ).reset_index().rename(columns={'order_date': 'ds'})
            self.category_demands[category] = daily_agg
        
        logger.info(f"Data loaded and prepared for {len(self.category_demands)} categories.")
        return True

    def _objective(self, trial: optuna.Trial, train_data: pd.DataFrame, val_data: pd.DataFrame) -> float:
        params = {
            'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
            'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True),
        }
        model = Prophet(**params, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False).fit(train_data)
        future_df = model.make_future_dataframe(periods=len(val_data))
        forecast = model.predict(future_df)
        val_predictions = forecast.iloc[-len(val_data):]['yhat'].values
        val_actual = val_data['y'].values
        return np.sqrt(np.mean((val_actual - val_predictions) ** 2))

    def train_all_forecast_models(self, validation_days: int = 30, n_trials: int = 5) -> None:
        logger.info("Training demand forecast models for all categories...")
        # ... (This function remains unchanged as it's for local training)
        pass # Pass to keep the code brief, logic is the same as your original

    def save_models(self):
        logger.info(f"Saving models to directory: {self.models_dir}")
        # ... (This function remains unchanged)
        pass # Pass to keep the code brief, logic is the same as your original

    def load_models(self):
        if not self.models_dir.exists():
            logger.error(f"Models directory not found: {self.models_dir}")
            return
        logger.info(f"Loading models from directory: {self.models_dir}")
        all_categories = list(self.category_demands.keys())
        for model_file in self.models_dir.glob("model_*.pkl"):
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            category_name_sanitized = model_file.stem.replace("model_", "")
            original_category = next((cat for cat in all_categories if sanitize_for_mlflow(cat) == category_name_sanitized), None)
            if original_category:
                self.demand_models[original_category] = model
                logger.info(f"Loaded model for '{original_category}' from {model_file}")

    def generate_inventory_recommendations(self, current_stock: Dict[str, float], category_name: str, lead_time_days: int = 5, service_level: float = 0.95, ordering_cost: float = 50, holding_cost_rate: float = 0.2) -> List[str]:
        """Generates inventory reorder recommendations for a single category."""
        logger.info(f"Generating inventory recommendations for '{category_name}'...")
        model = self.demand_models.get(category_name)
        if not model:
            logger.warning(f"No demand model found for '{category_name}'.")
            return [f"No model available for category '{category_name}'."]

        recommendations = []
        future = model.make_future_dataframe(periods=lead_time_days)
        forecast = model.predict(future)
        
        avg_daily_demand = forecast.iloc[-lead_time_days:]['yhat'].mean()
        std_dev_demand = forecast.iloc[-lead_time_days:]['yhat'].std()
        
        z_score = {0.95: 1.645}.get(service_level, 1.645)
        safety_stock = z_score * std_dev_demand * np.sqrt(lead_time_days)
        reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
        
        annual_demand = avg_daily_demand * 365
        avg_unit_price = self.category_stats.loc[category_name, 'avg_unit_price']
        holding_cost = avg_unit_price * holding_cost_rate
        
        economic_order_quantity = np.sqrt((2 * ordering_cost * annual_demand) / holding_cost) if holding_cost > 0 else 0
        
        if current_stock.get(category_name, 0) < reorder_point:
            rec_text = (
                f"REORDER ALERT for '{category_name}': "
                f"Current stock ({current_stock.get(category_name, 0):.0f}) is below reorder point ({reorder_point:.0f}). "
                f"Recommended order quantity (EOQ): {economic_order_quantity:.0f} units."
            )
            recommendations.append(rec_text)
            
        if not recommendations:
            recommendations.append(f"Stock level for '{category_name}' is sufficient.")
        return recommendations

    def generate_logistics_alerts_for_category(self, category_name: str, high_cost_threshold: float = 150.0) -> List[str]:
        """Generates strategic alerts for a single category."""
        logger.info(f"Generating strategic logistics alerts for '{category_name}'...")
        alerts = []
        stats = self.category_stats.loc[category_name]
        late_delivery_threshold = self.category_stats['late_delivery_risk'].quantile(0.75)
        
        if stats['late_delivery_risk'] > late_delivery_threshold:
            category_shipping_perf = self.shipping_mode_stats[self.shipping_mode_stats['category_name'] == category_name]
            if not category_shipping_perf.empty:
                best_mode = category_shipping_perf.loc[category_shipping_perf['late_delivery_risk'].idxmin()]
                alerts.append(
                    f"STRATEGIC ALERT for '{category_name}': High late delivery risk detected ({stats['late_delivery_risk']:.1%}). "
                    f"To reduce risk, consider switching to '{best_mode['shipping_mode']}', which has a historical risk of only {best_mode['late_delivery_risk']:.1%}."
                )
        
        if stats['avg_unit_price'] > high_cost_threshold:
            alerts.append(
                f"COST ALERT for '{category_name}': High average unit price (${stats['avg_unit_price']:.2f}) indicates high holding costs. "
                f"Consider a leaner inventory policy."
            )
        
        return alerts

# The main function and if __name__ == "__main__" block are for local execution
# and remain unchanged from your original file.
def main():
    pass

if __name__ == "__main__":
    pass
