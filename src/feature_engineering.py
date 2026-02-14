import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

class FeatureEngineer:
    """
    Feature engineering pipeline for car sales prediction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def engineer_features(self, df, target_col):
        """
        Main feature engineering pipeline
        """
        print("\n🔧 Engineering features...")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical variables
        X_encoded = self._encode_categorical(X)
        
        # Store feature names
        self.feature_names = X_encoded.columns.tolist()
        
        return X_encoded, y
    
    def _encode_categorical(self, X):
        """Encode categorical variables"""
        X_copy = X.copy()
        categorical_cols = X_copy.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Use label encoding for simplicity
            # In production, consider one-hot encoding for non-ordinal categories
            le = LabelEncoder()
            X_copy[col] = le.fit_transform(X_copy[col].astype(str))
            self.label_encoders[col] = le
            print(f"   ✅ Encoded {col}: {len(le.classes_)} categories")
        
        return X_copy
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        print("\n📏 Scaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        return X_train_scaled, X_test_scaled
    
    def save_artifacts(self, models_dir):
        """Save scaler and encoders"""
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        
        with open(models_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(models_path / 'label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        print(f"💾 Saved preprocessing artifacts to {models_dir}")