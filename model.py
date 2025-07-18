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
# Suppress warnings from libraries for a cleaner, more readable output
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configure our own logger for clear, informative messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def sanitize_for_mlflow(name: str) -> str:
    """
    Removes characters from a string that are invalid for MLflow keys or filenames.
    """
    # **THE FIX IS HERE**: Explicitly replace the forward slash before removing other invalid characters.
    sanitized_name = name.replace('/', '_')
    return re.sub(r"[^a-zA-Z0-9_.\s:-]", "", sanitized_name)

class SupplyChainOptimizer:
    """
    An AI agent that forecasts demand and optimizes inventory for a supply chain.
    It assumes it is being fed a preprocessed data file.
    """
    def __init__(self, preprocessed_data_url: str, models_dir: str = "saved_models"):
        """Initializes the optimizer with the URL to the clean data."""
        self.preprocessed_data_url = preprocessed_data_url
        self.models_dir = Path(models_dir)
        self.data: pd.DataFrame = None
        self.demand_models: Dict[str, Prophet] = {}
        self.category_demands: Dict[str, pd.DataFrame] = {}
        self.category_stats: pd.DataFrame = None
        self.shipping_mode_stats: pd.DataFrame = None # For logistics recommendations
        self.final_metrics: Dict[str, Dict[str, Any]] = {} # To store final model performance
        
        # Create the directory for saving models if it doesn't exist
        self.models_dir.mkdir(exist_ok=True)


    def load_preprocessed_data(self) -> bool:
        """
        Loads the preprocessed data from a URL and prepares it for modeling.
        """
        logger.info(f"Loading preprocessed data from URL...")
        try:
            # Read the data directly from the public URL
            self.data = pd.read_csv(self.preprocessed_data_url, parse_dates=['order_date'])
        except Exception as e:
            logger.error(f"Failed to load data from URL: {self.preprocessed_data_url}. Error: {e}")
            return False

        # Calculate category-level statistics for use in optimization
        self.category_stats = self.data.groupby('category_name').agg(
            order_quantity=('order_quantity', 'sum'),
            avg_unit_price=('order_total', 'sum'),
            late_delivery_risk=('late_delivery_risk', 'mean') 
        )
        self.category_stats['avg_unit_price'] = self.category_stats.apply(
            lambda row: row['avg_unit_price'] / row['order_quantity'] if row['order_quantity'] > 0 else 0,
            axis=1
        )
        
        # Calculate performance of each shipping mode per category
        self.shipping_mode_stats = self.data.groupby(['category_name', 'shipping_mode']).agg(
            late_delivery_risk=('late_delivery_risk', 'mean')
        ).reset_index()


        # Create a separate daily demand dataframe for each category for Prophet
        for category in self.data['category_name'].unique():
            category_data = self.data[self.data['category_name'] == category]
            # Prophet requires columns 'ds' (datestamp) and 'y' (target value)
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
        rmse = np.sqrt(np.mean((val_actual - val_predictions) ** 2))
        return rmse

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
                
                # Train final model on all available data and store it
                final_model = Prophet(**best_params).fit(daily_demand)
                self.demand_models[category] = final_model
                
                # Evaluate the final model and store metrics
                future = final_model.make_future_dataframe(periods=validation_days)
                forecast = final_model.predict(future)
                val_predictions = forecast.iloc[-validation_days:]['yhat'].values
                val_actual = val_data['y'].values
                final_rmse = np.sqrt(np.mean((val_actual - val_predictions) ** 2))
                
                self.final_metrics[category] = {
                    "rmse": final_rmse,
                    "best_params": best_params
                }
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

    def generate_inventory_recommendations(self, current_stock: Dict[str, float], lead_time_days: int = 5, service_level: float = 0.95, ordering_cost: float = 50, holding_cost_rate: float = 0.2) -> List[str]:
        """Generates inventory reorder recommendations using ROP and EOQ."""
        logger.info("Generating inventory recommendations...")
        if not self.demand_models:
            logger.warning("No demand models are loaded or trained. Cannot generate recommendations.")
            return ["No models available to generate recommendations."]

        recommendations = []
        for category, model in self.demand_models.items():
            future = model.make_future_dataframe(periods=lead_time_days)
            forecast = model.predict(future)
            
            avg_daily_demand = forecast.iloc[-lead_time_days:]['yhat'].mean()
            std_dev_demand = forecast.iloc[-lead_time_days:]['yhat'].std()
            
            z_score = {0.95: 1.645}.get(service_level, 1.645)
            safety_stock = z_score * std_dev_demand * np.sqrt(lead_time_days)
            reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
            
            annual_demand = avg_daily_demand * 365
            avg_unit_price = self.category_stats.loc[category, 'avg_unit_price']
            holding_cost = avg_unit_price * holding_cost_rate
            
            economic_order_quantity = np.sqrt((2 * ordering_cost * annual_demand) / holding_cost) if holding_cost > 0 else 0
            
            if current_stock.get(category, 0) < reorder_point:
                rec_text = (
                    f"REORDER ALERT for '{category}': "
                    f"Current stock ({current_stock.get(category, 0):.0f}) is below reorder point ({reorder_point:.0f}). "
                    f"Recommended order quantity (EOQ): {economic_order_quantity:.0f} units."
                )
                recommendations.append(rec_text)
            
        if not recommendations:
            recommendations.append("All category stock levels are sufficient.")
        return recommendations

    def generate_logistics_alerts(self, high_cost_threshold: float = 150.0) -> List[str]:
        """Generates strategic alerts based on historical category performance."""
        logger.info("Generating strategic logistics alerts...")
        alerts = []
        
        # Use a dynamic, data-driven threshold instead of a hard-coded one.
        # Define "high risk" as any category in the top 25% (75th percentile) for late deliveries.
        late_delivery_threshold = self.category_stats['late_delivery_risk'].quantile(0.75)
        logger.info(f"Using dynamic late delivery risk threshold: {late_delivery_threshold:.1%}")
        
        for category, stats in self.category_stats.iterrows():
            # Check for high late delivery risk
            if stats['late_delivery_risk'] > late_delivery_threshold:
                # Find the best shipping mode for this high-risk category
                category_shipping_perf = self.shipping_mode_stats[self.shipping_mode_stats['category_name'] == category]
                if not category_shipping_perf.empty:
                    best_mode = category_shipping_perf.loc[category_shipping_perf['late_delivery_risk'].idxmin()]
                    alert_text = (
                        f"STRATEGIC ALERT for '{category}': "
                        f"High late delivery risk detected ({stats['late_delivery_risk']:.1%}). "
                        f"To reduce risk, consider switching to '{best_mode['shipping_mode']}', which has a historical risk of only {best_mode['late_delivery_risk']:.1%}."
                    )
                    alerts.append(alert_text)
            
            # Check for high holding costs (based on average unit price)
            if stats['avg_unit_price'] > high_cost_threshold:
                alert_text = (
                    f"COST ALERT for '{category}': "
                    f"High average unit price (${stats['avg_unit_price']:.2f}) indicates high holding costs. "
                    f"Consider a leaner inventory policy (e.g., higher service level, more frequent reviews)."
                )
                alerts.append(alert_text)
            
        return alerts


def main() -> SupplyChainOptimizer:
    """
    Main function to run the full optimization pipeline.
    
    Returns:
        The trained SupplyChainOptimizer instance.
    """
    # Initialize optimizer with the path to the preprocessed data
    optimizer = SupplyChainOptimizer("preprocessed_supply_chain_data.csv")
    
    if not optimizer.load_preprocessed_data():
        return None
    
    # Train the models
    optimizer.train_all_forecast_models(n_trials=5) 
    
    # Example current stock levels (this would come from a live inventory system)
    current_stock = {
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
    
    # --- Generate and Print Both Types of Recommendations ---
    inventory_recommendations = optimizer.generate_inventory_recommendations(current_stock)
    logistics_alerts = optimizer.generate_logistics_alerts()
    
    print("\n--- Inventory Recommendations ---")
    for rec in inventory_recommendations:
        print(f"- {rec}")
        
    print("\n--- Strategic Alerts ---")
    if logistics_alerts:
        for alert in logistics_alerts:
            print(f"- {alert}")
    else:
        print("- No strategic alerts at this time.")

        
    '''# Print the final performance summary
    print("\n--- Final Model Performance Summary ---")
    for category, metrics in optimizer.final_metrics.items():
        print(f"\nCategory: {category}")
        print(f"  - Final RMSE: {metrics['rmse']:.2f}")
        print(f"  - Best Params: {metrics['best_params']}")'''
        
    return optimizer

if __name__ == "__main__":
    # Run the main pipeline and capture the returned optimizer object
    trained_optimizer = main()
    
    # Now you can use the trained_optimizer object for other tasks, like saving the models
    if trained_optimizer:
        logger.info("\n--- Post-Training Task ---")
        trained_optimizer.save_models()
