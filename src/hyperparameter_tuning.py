import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from pathlib import Path
import time

class HyperparameterTuner:
    """
    Hyperparameter tuning for regression models
    """
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.cv_results = None
        
    def get_param_grid(self):
        """Define parameter grids for different models"""
        
        if self.model_type == 'xgboost':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2]
            }
        
        elif self.model_type == 'random_forest':
            return {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
    
    def get_random_param_dist(self):
        """Define parameter distributions for RandomizedSearch"""
        
        if self.model_type == 'xgboost':
            return {
                'n_estimators': [100, 150, 200, 250, 300],
                'max_depth': [3, 4, 5, 6, 7, 8, 9],
                'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'min_child_weight': [1, 2, 3, 4, 5],
                'gamma': [0, 0.05, 0.1, 0.15, 0.2, 0.25]
            }
        
        elif self.model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500],
                'max_depth': [5, 10, 15, 20, 25, 30, None],
                'min_samples_split': [2, 3, 5, 7, 10],
                'min_samples_leaf': [1, 2, 3, 4, 5],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
    
    def tune_with_grid_search(self, X_train, y_train, cv=5):
        """
        Perform Grid Search for hyperparameter tuning
        Best for smaller parameter spaces
        """
        print(f"\n{'='*70}")
        print(f"🔍 GRID SEARCH - {self.model_type.upper()}")
        print(f"{'='*70}")
        
        # Initialize base model
        if self.model_type == 'xgboost':
            base_model = XGBRegressor(random_state=42, n_jobs=-1)
        elif self.model_type == 'random_forest':
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Get parameter grid
        param_grid = self.get_param_grid()
        
        print(f"\n📊 Parameter Grid:")
        for param, values in param_grid.items():
            print(f"   {param}: {values}")
        
        # Perform Grid Search
        print(f"\n⏳ Starting Grid Search with {cv}-fold cross-validation...")
        print(f"   Total combinations to test: {np.prod([len(v) for v in param_grid.values()])}")
        
        start_time = time.time()
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=2,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        
        # Store results
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.cv_results = pd.DataFrame(grid_search.cv_results_)
        
        # Print results
        print(f"\n✅ Grid Search Complete!")
        print(f"   Time Elapsed: {elapsed_time:.2f} seconds")
        print(f"\n🏆 Best Parameters:")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        print(f"\n📊 Best CV R² Score: {self.best_score:.4f}")
        
        return self.best_model, self.best_params
    
    def tune_with_random_search(self, X_train, y_train, n_iter=50, cv=5):
        """
        Perform Randomized Search for hyperparameter tuning
        Best for larger parameter spaces
        """
        print(f"\n{'='*70}")
        print(f"🎲 RANDOMIZED SEARCH - {self.model_type.upper()}")
        print(f"{'='*70}")
        
        # Initialize base model
        if self.model_type == 'xgboost':
            base_model = XGBRegressor(random_state=42, n_jobs=-1)
        elif self.model_type == 'random_forest':
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Get parameter distributions
        param_dist = self.get_random_param_dist()
        
        print(f"\n📊 Parameter Distributions:")
        for param, values in param_dist.items():
            print(f"   {param}: {len(values)} options")
        
        # Perform Randomized Search
        print(f"\n⏳ Starting Randomized Search...")
        print(f"   Iterations: {n_iter}")
        print(f"   CV Folds: {cv}")
        
        start_time = time.time()
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=2,
            random_state=42,
            return_train_score=True
        )
        
        random_search.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        
        # Store results
        self.best_model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.cv_results = pd.DataFrame(random_search.cv_results_)
        
        # Print results
        print(f"\n✅ Randomized Search Complete!")
        print(f"   Time Elapsed: {elapsed_time:.2f} seconds")
        print(f"\n🏆 Best Parameters:")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        print(f"\n📊 Best CV R² Score: {self.best_score:.4f}")
        
        return self.best_model, self.best_params
    
    def evaluate_tuned_model(self, X_test, y_test):
        """Evaluate the tuned model on test set"""
        
        if self.best_model is None:
            raise ValueError("No tuned model found. Run tuning first.")
        
        # Predict
        y_pred = self.best_model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\n{'='*70}")
        print(f"📊 TUNED MODEL PERFORMANCE")
        print(f"{'='*70}")
        print(f"\n   Test R² Score: {r2:.4f}")
        print(f"   Test RMSE:     ${rmse:,.2f}")
        print(f"   Test MAE:      ${mae:,.2f}")
        
        return {
            'test_r2': r2,
            'test_rmse': rmse,
            'test_mae': mae
        }
    
    def save_tuned_model(self, models_dir, prefix='tuned'):
        """Save the tuned model and parameters"""
        
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = models_path / f'{prefix}_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save parameters
        params_file = models_path / f'{prefix}_params.pkl'
        with open(params_file, 'wb') as f:
            pickle.dump(self.best_params, f)
        
        print(f"\n💾 Saved tuned model to: {model_file}")
        print(f"💾 Saved parameters to: {params_file}")
    
    def compare_with_baseline(self, baseline_score):
        """Compare tuned model with baseline"""
        
        improvement = ((self.best_score - baseline_score) / baseline_score) * 100
        
        print(f"\n{'='*70}")
        print(f"📈 IMPROVEMENT ANALYSIS")
        print(f"{'='*70}")
        print(f"\n   Baseline R² Score:     {baseline_score:.4f}")
        print(f"   Tuned R² Score:        {self.best_score:.4f}")
        print(f"   Improvement:           {improvement:+.2f}%")
        
        if improvement > 0:
            print(f"\n✅ Hyperparameter tuning improved performance!")
        else:
            print(f"\n⚠️ Baseline model was already well-optimized.")