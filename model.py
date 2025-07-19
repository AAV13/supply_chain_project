import pandas as pd
import numpy as np
from prophet import Prophet
import mlflow
import optuna
from typing import Dict, List, Any, Tuple, Optional
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
    """
    Removes characters from a string that are invalid for MLflow keys or filenames.
    """
    sanitized_name = name.replace('/', '_')
    return re.sub(r"[^a-zA-Z0-9_.\s:-]", "", sanitized_name)

class SupplyChainOptimizer:
    """
    An AI agent that forecasts demand and optimizes inventory for a supply chain.
    """
    def __init__(self, preprocessed_data_url: str, models_dir: str = "saved_models"):
        """Initializes the optimizer with the URL to the clean data."""
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
        """
        Loads the preprocessed data from a URL and prepares it for modeling.
        """
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
        """Objective function for Optuna hyperparameter optimization."""
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
        """
        Trains a Prophet model for each category with hyperparameter tuning.
        """
        logger.info("Training demand forecast models for all categories...")
        with mlflow.start_run(run_name="Category_Demand_Forecast_Training"):
            mlflow.log_param("validation_days", validation_days)
            mlflow.log_param("optuna_trials", n_trials)

            for category, daily_demand in self.category_demands.items():
                if len(daily_demand) < validation_days * 2:
                    logger.warning(f"Skipping category '{category}': not enough data for training and validation.")
                    continue
                
                logger.info(f"Training model for category: '{category}'")
                train_data, val_data = daily_demand.iloc[:-validation_days], daily_demand.iloc[-validation_days:]

                study = optuna.create_study(direction="minimize")
                study.optimize(lambda trial: self._objective(trial, train_data, val_data), n_trials=n_trials)
                
                best_params = study.best_params
                sanitized_category = sanitize_for_mlflow(category)
                mlflow.log_params({f"{sanitized_category}_best_{k}": v for k, v in best_params.items()})
                
                final_model = Prophet(**best_params).fit(daily_demand)
                self.demand_models[category] = final_model
                
                future = final_model.make_future_dataframe(periods=validation_days)
                forecast = final_model.predict(future)
                val_predictions = forecast.iloc[-validation_days:]['yhat'].values
                val_actual = val_data['y'].values
                final_rmse = np.sqrt(np.mean((val_actual - val_predictions) ** 2))
                
                self.final_metrics[category] = {"rmse": final_rmse, "best_params": best_params}
                mlflow.log_metric(f"{sanitized_category}_final_rmse", final_rmse)
                logger.info(f"Model for '{category}' trained. Final RMSE: {final_rmse:.2f}")

    def save_models(self):
        """Saves all trained Prophet models to pickle files."""
        logger.info(f"Saving models to directory: {self.models_dir}")
        for category, model in self.demand_models.items():
            sanitized_category = sanitize_for_mlflow(category)
            model_path = self.models_dir / f"model_{sanitized_category}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved model for '{category}' to {model_path}")

    def load_models(self):
        """Loads Prophet models from pickle files."""
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

    def get_processed_recommendations(
        self, current_stock: Dict[str, float], category_name: str, forecast_days: int
    ) -> Tuple[List[str], List[str], Optional[Dict]]:
        """
        Generates recommendations and returns a fully processed dictionary for the API.
        """
        logger.info(f"Generating probabilistic forecast for '{category_name}'...")
        model = self.demand_models.get(category_name)
        if not model:
            logger.warning(f"No demand model found for '{category_name}'.")
            return [f"No model available for category '{category_name}'."], [], None

        future_df = model.make_future_dataframe(periods=forecast_days)
        forecast_df = model.predict(future_df)

        inventory_recs = self._generate_inventory_text(
            current_stock, category_name, forecast_df
        )
        logistics_alerts = self.generate_logistics_alerts_for_category(category_name)

        formatted_forecast = {
            "dates": [d.strftime('%Y-%m-%d') for d in future_df['ds']],
            "forecast_yhat": forecast_df['yhat'].tolist(),
            "forecast_yhat_lower": forecast_df['yhat_lower'].tolist(),
            "forecast_yhat_upper": forecast_df['yhat_upper'].tolist()
        }
        return inventory_recs, logistics_alerts, formatted_forecast

    def _generate_inventory_text(
        self, current_stock: Dict[str, float], category_name: str, forecast_df: pd.DataFrame, 
        lead_time_days: int = 5, service_level: float = 0.95, ordering_cost: float = 50, holding_cost_rate: float = 0.2
    ) -> List[str]:
        """Generates inventory reorder text based on a pre-computed forecast."""
        lead_time_forecast = forecast_df.iloc[-lead_time_days:]
        avg_daily_demand = lead_time_forecast['yhat'].mean()
        std_dev_demand = (lead_time_forecast['yhat_upper'] - lead_time_forecast['yhat_lower']).mean()
        
        z_score = {0.95: 1.645}.get(service_level, 1.645)
        safety_stock = z_score * std_dev_demand * np.sqrt(lead_time_days)
        reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
        
        annual_demand = forecast_df['yhat'].iloc[-365:].mean() * 365
        avg_unit_price = self.category_stats.loc[category_name, 'avg_unit_price']
        holding_cost = avg_unit_price * holding_cost_rate
        
        economic_order_quantity = np.sqrt((2 * ordering_cost * annual_demand) / holding_cost) if holding_cost > 0 else 0
        
        recommendations = []
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
        try:
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
        except KeyError:
            logger.warning(f"Could not generate logistics alerts for '{category_name}'. Stats not found.")
        
        return alerts

def main() -> SupplyChainOptimizer:
    """
    Main function to run the full optimization pipeline.
    """
    optimizer = SupplyChainOptimizer("preprocessed_supply_chain_data.csv")
    if not optimizer.load_preprocessed_data():
        return None
    optimizer.train_all_forecast_models(n_trials=5) 
    
    current_stock = {
        'Sporting Goods': 15000.0, 'Cleats': 25000.0, "Women's Apparel": 40000.0,
        "Men's Footwear": 10000.0, "Camping & Hiking": 500.0, "Accessories": 10.0,
        "Golf Gloves": 5.0, 'Indoor/Outdoor Games': 8000.0, 'Shop By Sport': 12000.0,
        'Fishing': 3000.0, 'Cardio Equipment': 20.0, 'Water Sports': 4000.0,
        'Electronics': 9000.0, 'Girls\' Apparel': 18000.0, 'Golf Balls': 5000.0
    }
    
    inventory_recommendations, logistics_alerts, _ = optimizer.get_processed_recommendations(
        current_stock, 'Sporting Goods', 30
    )
    
    print("\n--- Inventory Recommendations ---")
    for rec in inventory_recommendations:
        print(f"- {rec}")
    print("\n--- Strategic Alerts ---")
    if logistics_alerts:
        for alert in logistics_alerts:
            print(f"- {alert}")
    else:
        print("- No strategic alerts at this time.")
    return optimizer

if __name__ == "__main__":
    trained_optimizer = main()
    if trained_optimizer:
        logger.info("\n--- Post-Training Task ---")
        trained_optimizer.save_models()
