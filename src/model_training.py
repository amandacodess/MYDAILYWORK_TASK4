import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    """
    Train and evaluate multiple regression models
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare performance"""
        
        print("\n🤖 Training Models...")
        print("="*60)
        
        # Define models
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        # Train and evaluate each model
        for name, model in models_to_train.items():
            print(f"\n📌 Training {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluate
            metrics = self._evaluate_model(y_train, y_pred_train, y_test, y_pred_test)
            
            # Store
            self.models[name] = model
            self.results[name] = metrics
            
            # Print results
            print(f"\n   📊 {name} Performance:")
            print(f"      Train R² Score: {metrics['train_r2']:.4f}")
            print(f"      Test R² Score:  {metrics['test_r2']:.4f}")
            print(f"      Test RMSE:      {metrics['test_rmse']:.2f}")
            print(f"      Test MAE:       {metrics['test_mae']:.2f}")
        
        # Select best model
        self._select_best_model()
        
        return self.results
    
    def _evaluate_model(self, y_train, y_pred_train, y_test, y_pred_test):
        """Calculate evaluation metrics"""
        return {
            'train_r2': r2_score(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_test)
        }
    
    def _select_best_model(self):
        """Select best model based on test R² score"""
        best_r2 = -np.inf
        
        for name, metrics in self.results.items():
            if metrics['test_r2'] > best_r2:
                best_r2 = metrics['test_r2']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        print("\n" + "="*60)
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print(f"   Test R² Score: {self.results[self.best_model_name]['test_r2']:.4f}")
        print("="*60)
    
    def plot_model_comparison(self, save_path):
        """Visualize model comparison"""
        # Prepare data for plotting
        model_names = list(self.results.keys())
        test_r2 = [self.results[name]['test_r2'] for name in model_names]
        test_rmse = [self.results[name]['test_rmse'] for name in model_names]
        test_mae = [self.results[name]['test_mae'] for name in model_names]
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # R² Score
        axes[0].bar(model_names, test_r2, color=['#FF6B6B' if name == self.best_model_name else '#4ECDC4' for name in model_names])
        axes[0].set_ylabel('R² Score', fontweight='bold')
        axes[0].set_title('Model R² Score Comparison', fontweight='bold', fontsize=14)
        axes[0].set_ylim([min(test_r2) * 0.9, 1.0])
        axes[0].grid(alpha=0.3, axis='y')
        
        # RMSE
        axes[1].bar(model_names, test_rmse, color=['#FF6B6B' if name == self.best_model_name else '#95E1D3' for name in model_names])
        axes[1].set_ylabel('RMSE', fontweight='bold')
        axes[1].set_title('Model RMSE Comparison (Lower is Better)', fontweight='bold', fontsize=14)
        axes[1].grid(alpha=0.3, axis='y')
        
        # MAE
        axes[2].bar(model_names, test_mae, color=['#FF6B6B' if name == self.best_model_name else '#F38181' for name in model_names])
        axes[2].set_ylabel('MAE', fontweight='bold')
        axes[2].set_title('Model MAE Comparison (Lower is Better)', fontweight='bold', fontsize=14)
        axes[2].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Model comparison plot saved to {save_path}")
        plt.show()
    
    def plot_feature_importance(self, feature_names, save_path):
        """Plot feature importance for tree-based models"""
        if self.best_model_name in ['Random Forest', 'XGBoost']:
            importance = self.best_model.feature_importances_
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False).head(15)
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(importance_df)), importance_df['Importance'], color='steelblue')
            plt.yticks(range(len(importance_df)), importance_df['Feature'])
            plt.xlabel('Importance Score', fontweight='bold')
            plt.title(f'Top 15 Feature Importances ({self.best_model_name})', 
                     fontweight='bold', fontsize=14)
            plt.gca().invert_yaxis()
            plt.grid(alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Feature importance plot saved to {save_path}")
            plt.show()
    
    def save_best_model(self, models_dir):
        """Save the best performing model"""
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        
        with open(models_path / 'best_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'metrics': self.results[self.best_model_name]
        }
        
        with open(models_path / 'model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"💾 Best model ({self.best_model_name}) saved to {models_dir}")