import pandas as pd
import numpy as np
from pathlib import Path

class CarSalesDataProcessor:
    """
    Professional data preprocessing pipeline for car sales prediction
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """Load data from kagglehub download path with encoding handling"""
        # Find the CSV file in the downloaded path
        csv_files = list(Path(self.data_path).rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {self.data_path}")
        
        csv_path = csv_files[0]
        print(f"📁 Found CSV file: {csv_path}")
        
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
        
        for encoding in encodings:
            try:
                print(f"   Trying encoding: {encoding}...")
                self.df = pd.read_csv(csv_path, encoding=encoding)
                print(f"✅ Successfully loaded with encoding: {encoding}")
                print(f"✅ Loaded {len(self.df)} records with {len(self.df.columns)} columns")
                return self.df
            except UnicodeDecodeError as e:
                print(f"   ❌ Failed with {encoding}")
                continue
            except Exception as e:
                print(f"   ❌ Error with {encoding}: {str(e)[:100]}")
                continue
        
        # If all encodings fail, try with error handling
        print("\n⚠️ All standard encodings failed. Trying with error handling...")
        try:
            self.df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')
            print(f"✅ Loaded with UTF-8 (replacing errors)")
            print(f"✅ Loaded {len(self.df)} records with {len(self.df.columns)} columns")
            return self.df
        except Exception as e:
            # Last resort: try reading as binary and detecting encoding
            print(f"\n❌ All methods failed. Attempting charset detection...")
            try:
                import chardet
                with open(csv_path, 'rb') as f:
                    result = chardet.detect(f.read(100000))
                detected_encoding = result['encoding']
                print(f"   Detected encoding: {detected_encoding}")
                self.df = pd.read_csv(csv_path, encoding=detected_encoding)
                print(f"✅ Loaded with detected encoding: {detected_encoding}")
                print(f"✅ Loaded {len(self.df)} records with {len(self.df.columns)} columns")
                return self.df
            except ImportError:
                print("   ⚠️ chardet not installed. Install with: pip install chardet")
                raise Exception(f"Failed to load CSV. Try installing chardet: pip install chardet")
            except Exception as e2:
                raise Exception(f"Failed to load CSV with any method. Final error: {e2}")
    
    def explore_data(self):
        """Initial data exploration"""
        print("\n📊 Dataset Overview:")
        print(self.df.info())
        print("\n📈 Statistical Summary:")
        print(self.df.describe())
        print("\n🔍 Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found!")
        print("\n🎯 Sample Records:")
        print(self.df.head())
        print(f"\n📋 Column Names:")
        print(self.df.columns.tolist())
        
    def clean_data(self):
        """Handle missing values and outliers"""
        print("\n🧹 Cleaning data...")
        
        # Store original count
        original_count = len(self.df)
        
        # Drop duplicates
        self.df = self.df.drop_duplicates()
        duplicates_removed = original_count - len(self.df)
        if duplicates_removed > 0:
            print(f"   Removed {duplicates_removed} duplicate rows")
        
        # Identify the price column (case-insensitive search)
        price_col = None
        for col in self.df.columns:
            if 'price' in col.lower():
                price_col = col
                break
        
        if price_col is None:
            print("⚠️ Warning: No 'price' column found. Using last column as target.")
            price_col = self.df.columns[-1]
        
        print(f"   Target column identified: '{price_col}'")
        
        # Drop rows where target is missing
        before_drop = len(self.df)
        self.df = self.df.dropna(subset=[price_col])
        target_na_removed = before_drop - len(self.df)
        if target_na_removed > 0:
            print(f"   Removed {target_na_removed} rows with missing target values")
        
        # Handle missing values in features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Impute numerical columns with median
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"   Imputed {col} with median: {median_val:.2f}")
        
        # Impute categorical columns with mode
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode()[0]
                self.df[col].fillna(mode_val, inplace=True)
                print(f"   Imputed {col} with mode: {mode_val}")
        
        print(f"\n✅ Cleaned: {original_count} → {len(self.df)} records")
        return self.df
    
    def save_processed_data(self, output_path):
        """Save cleaned data"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"💾 Processed data saved to {output_path}")