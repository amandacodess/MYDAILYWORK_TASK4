import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path

class ModelExplainer:
    """
    SHAP-based model explainability for interpretable ML
    """
    
    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self):
        """Create SHAP explainer based on model type"""
        
        print("\n🔍 Creating SHAP Explainer...")
        
        model_name = type(self.model).__name__
        
        if 'XGB' in model_name or 'GradientBoosting' in model_name:
            # Tree explainer for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
            print("   ✅ Using TreeExplainer (fast, exact)")
        
        elif 'RandomForest' in model_name:
            # Tree explainer for Random Forest
            self.explainer = shap.TreeExplainer(self.model)
            print("   ✅ Using TreeExplainer (fast, exact)")
        
        else:
            # Kernel explainer for other models (slower but model-agnostic)
            background = shap.sample(self.X_train, 100)
            self.explainer = shap.KernelExplainer(self.model.predict, background)
            print("   ✅ Using KernelExplainer (model-agnostic)")
        
        return self.explainer
    
    def calculate_shap_values(self, X_test):
        """Calculate SHAP values for test set"""
        
        if self.explainer is None:
            self.create_explainer()
        
        print("\n⏳ Calculating SHAP values...")
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X_test)
        
        print("   ✅ SHAP values calculated!")
        
        return self.shap_values
    
    def plot_summary(self, X_test, save_path=None):
        """Create SHAP summary plot"""
        
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        print("\n📊 Creating SHAP Summary Plot...")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, 
            X_test, 
            feature_names=self.feature_names,
            show=False
        )
        plt.title('SHAP Summary Plot - Feature Impact on Predictions', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved to: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, X_test, save_path=None):
        """Create SHAP feature importance bar plot"""
        
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        print("\n📊 Creating SHAP Feature Importance Plot...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            X_test,
            feature_names=self.feature_names,
            plot_type='bar',
            show=False
        )
        plt.title('SHAP Feature Importance - Mean Impact on Model Output',
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved to: {save_path}")
        
        plt.show()
    
    def plot_waterfall(self, X_test, instance_idx=0, save_path=None):
        """Create SHAP waterfall plot for a single prediction"""
        
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        print(f"\n📊 Creating SHAP Waterfall Plot for instance {instance_idx}...")
        
        # Create explanation object
        shap_explanation = shap.Explanation(
            values=self.shap_values[instance_idx],
            base_values=self.explainer.expected_value,
            data=X_test.iloc[instance_idx],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap_explanation, show=False)
        plt.title(f'SHAP Waterfall Plot - Prediction Explanation (Instance {instance_idx})',
                 fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved to: {save_path}")
        
        plt.show()
    
    def plot_force(self, X_test, instance_idx=0, save_path=None):
        """Create SHAP force plot for a single prediction"""
        
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        print(f"\n📊 Creating SHAP Force Plot for instance {instance_idx}...")
        
        # Force plot
        shap.force_plot(
            self.explainer.expected_value,
            self.shap_values[instance_idx],
            X_test.iloc[instance_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved to: {save_path}")
        
        plt.show()
    
    def plot_dependence(self, X_test, feature_name, save_path=None):
        """Create SHAP dependence plot for a specific feature"""
        
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        print(f"\n📊 Creating SHAP Dependence Plot for '{feature_name}'...")
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_name,
            self.shap_values,
            X_test,
            feature_names=self.feature_names,
            show=False
        )
        plt.title(f'SHAP Dependence Plot - {feature_name}',
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved to: {save_path}")
        
        plt.show()
    
    def generate_full_report(self, X_test, output_dir='visualizations/shap'):
        """Generate complete SHAP analysis report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"📊 GENERATING COMPLETE SHAP REPORT")
        print(f"{'='*70}")
        
        # 1. Summary plot
        self.plot_summary(X_test, save_path=output_path / 'shap_summary.png')
        
        # 2. Feature importance
        self.plot_feature_importance(X_test, save_path=output_path / 'shap_importance.png')
        
        # 3. Waterfall for first prediction
        self.plot_waterfall(X_test, instance_idx=0, save_path=output_path / 'shap_waterfall.png')
        
        # 4. Dependence plots for top 3 features
        # Get top 3 features by mean absolute SHAP value
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        top_features_idx = np.argsort(mean_shap)[-3:][::-1]
        
        for idx in top_features_idx:
            feature = self.feature_names[idx]
            safe_name = feature.replace('/', '_').replace(' ', '_')
            self.plot_dependence(
                X_test, 
                feature, 
                save_path=output_path / f'shap_dependence_{safe_name}.png'
            )
        
        print(f"\n✅ Complete SHAP report generated in: {output_dir}")