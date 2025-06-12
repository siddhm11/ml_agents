import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSVDataCleaner:
    """Clean CSV data by separating rows with NaN values in target column"""
    
    def __init__(self):
        self.original_data = None
        self.clean_data = None
        self.nan_data = None
        self.stats = {}
    
    def load_csv(self, csv_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """Load CSV with automatic encoding detection"""
        try:
            # Try different separators and encodings
            separators = [',', ';', '\t', '|']
            encodings = [encoding, 'utf-8', 'latin-1', 'cp1252']
            
            df = None
            for enc in encodings:
                for sep in separators:
                    try:
                        df = pd.read_csv(csv_path, encoding=enc, sep=sep)
                        if df.shape[1] > 1:  # Valid separation found
                            logger.info(f"Successfully loaded CSV with encoding: {enc}, separator: '{sep}'")
                            return df
                    except:
                        continue
            
            if df is None:
                raise ValueError("Could not load CSV with any encoding/separator combination")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise
    
    def analyze_target_column(self, df: pd.DataFrame, target_column: str) -> dict:
        """Analyze target column for missing values and data quality"""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        target_series = df[target_column]
        
        analysis = {
            'total_rows': len(df),
            'target_column': target_column,
            'missing_count': target_series.isnull().sum(),
            'missing_percentage': (target_series.isnull().sum() / len(df)) * 100,
            'valid_count': target_series.notna().sum(),
            'valid_percentage': (target_series.notna().sum() / len(df)) * 100,
            'data_type': str(target_series.dtype),
            'unique_values': target_series.nunique(),
            'min_value': target_series.min() if target_series.notna().any() else None,
            'max_value': target_series.max() if target_series.notna().any() else None,
            'mean_value': target_series.mean() if target_series.notna().any() else None
        }
        
        return analysis
    
    def separate_nan_data(self, df: pd.DataFrame, target_column: str) -> tuple:
        """Separate clean data from data with NaN values in target column"""
        logger.info(f"Separating data based on NaN values in '{target_column}'")
        
        # Identify rows with NaN values in target column
        nan_mask = df[target_column].isnull()
        
        # Separate clean and NaN data
        clean_data = df[~nan_mask].copy()  # Rows without NaN in target
        nan_data = df[nan_mask].copy()     # Rows with NaN in target
        
        logger.info(f"Original dataset: {len(df)} rows")
        logger.info(f"Clean dataset: {len(clean_data)} rows")
        logger.info(f"NaN dataset: {len(nan_data)} rows")
        
        return clean_data, nan_data
    
    def save_datasets(self, clean_data: pd.DataFrame, nan_data: pd.DataFrame, 
                     base_path: str, target_column: str) -> dict:
        """Save both clean and NaN datasets with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = Path(base_path)
        base_name = base_path.stem
        base_dir = base_path.parent
        
        # Create output paths
        clean_path = base_dir / f"{base_name}_CLEAN_{timestamp}.csv"
        nan_path = base_dir / f"{base_name}_NAN_REMOVED_{timestamp}.csv"
        stats_path = base_dir / f"{base_name}_CLEANING_STATS_{timestamp}.json"
        
        try:
            # Save clean dataset
            clean_data.to_csv(clean_path, index=False)
            logger.info(f"Clean dataset saved to: {clean_path}")
            
            # Save NaN dataset (if any rows exist)
            if len(nan_data) > 0:
                nan_data.to_csv(nan_path, index=False)
                logger.info(f"NaN dataset saved to: {nan_path}")
            else:
                logger.info("No NaN rows to save")
                nan_path = None
            
            # Save cleaning statistics
            cleaning_stats = {
                'timestamp': timestamp,
                'original_file': str(base_path),
                'target_column': target_column,
                'original_rows': len(clean_data) + len(nan_data),
                'clean_rows': len(clean_data),
                'nan_rows': len(nan_data),
                'cleaning_percentage': (len(nan_data) / (len(clean_data) + len(nan_data))) * 100,
                'clean_file': str(clean_path),
                'nan_file': str(nan_path) if nan_path else None,
                'columns': list(clean_data.columns),
                'clean_data_summary': {
                    'target_min': float(clean_data[target_column].min()) if len(clean_data) > 0 else None,
                    'target_max': float(clean_data[target_column].max()) if len(clean_data) > 0 else None,
                    'target_mean': float(clean_data[target_column].mean()) if len(clean_data) > 0 else None,
                    'target_std': float(clean_data[target_column].std()) if len(clean_data) > 0 else None
                }
            }
            
            with open(stats_path, 'w') as f:
                json.dump(cleaning_stats, f, indent=2, default=str)
            
            logger.info(f"Cleaning statistics saved to: {stats_path}")
            
            return {
                'clean_file': str(clean_path),
                'nan_file': str(nan_path) if nan_path else None,
                'stats_file': str(stats_path),
                'stats': cleaning_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to save datasets: {e}")
            raise
    
    def clean_csv(self, csv_path: str, target_column: str = "Purchase Amount (USD)", 
                  save_results: bool = True) -> dict:
        """Main function to clean CSV data"""
        logger.info(f"Starting CSV cleaning process for: {csv_path}")
        
        try:
            # Load data
            self.original_data = self.load_csv(csv_path)
            logger.info(f"Loaded dataset with shape: {self.original_data.shape}")
            
            # Analyze target column
            analysis = self.analyze_target_column(self.original_data, target_column)
            self.stats = analysis
            
            # Print analysis
            self.print_analysis(analysis)
            
            # Separate clean and NaN data
            self.clean_data, self.nan_data = self.separate_nan_data(
                self.original_data, target_column
            )
            
            # Save results if requested
            if save_results:
                save_info = self.save_datasets(
                    self.clean_data, self.nan_data, csv_path, target_column
                )
                
                return {
                    'status': 'success',
                    'original_rows': len(self.original_data),
                    'clean_rows': len(self.clean_data),
                    'nan_rows': len(self.nan_data),
                    'files': save_info,
                    'analysis': analysis
                }
            else:
                return {
                    'status': 'success',
                    'clean_data': self.clean_data,
                    'nan_data': self.nan_data,
                    'analysis': analysis
                }
                
        except Exception as e:
            logger.error(f"CSV cleaning failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def print_analysis(self, analysis: dict):
        """Print detailed analysis of the target column"""
        print("\n" + "="*60)
        print("📊 TARGET COLUMN ANALYSIS")
        print("="*60)
        print(f"📁 Target Column: {analysis['target_column']}")
        print(f"📈 Total Rows: {analysis['total_rows']:,}")
        print(f"✅ Valid Rows: {analysis['valid_count']:,} ({analysis['valid_percentage']:.2f}%)")
        print(f"❌ Missing Rows: {analysis['missing_count']:,} ({analysis['missing_percentage']:.2f}%)")
        print(f"🔢 Data Type: {analysis['data_type']}")
        print(f"🎯 Unique Values: {analysis['unique_values']:,}")
        
        if analysis['min_value'] is not None:
            print(f"📊 Value Range: {analysis['min_value']:,.2f} - {analysis['max_value']:,.2f}")
            print(f"📊 Mean Value: {analysis['mean_value']:,.2f}")
        
        print("="*60)
    
    def get_sample_data(self, dataset_type: str = 'clean', n_rows: int = 5) -> pd.DataFrame:
        """Get sample data from clean or nan dataset"""
        if dataset_type == 'clean' and self.clean_data is not None:
            return self.clean_data.head(n_rows)
        elif dataset_type == 'nan' and self.nan_data is not None:
            return self.nan_data.head(n_rows)
        else:
            return pd.DataFrame()

# Standalone execution function
def main():
    """Main execution function"""
    
    # Configuration
    CSV_FILE_PATH = "agents/Fashion_Retail_Sales-2.csv"  # Update with your file path
    TARGET_COLUMN = "Purchase Amount (USD)"
    
    # Initialize cleaner
    cleaner = CSVDataCleaner()
    
    try:
        # Clean the CSV
        results = cleaner.clean_csv(CSV_FILE_PATH, TARGET_COLUMN)
        
        if results['status'] == 'success':
            print("\n🎉 CSV CLEANING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"📁 Original File: {CSV_FILE_PATH}")
            print(f"📊 Original Rows: {results['original_rows']:,}")
            print(f"✅ Clean Rows: {results['clean_rows']:,}")
            print(f"❌ Removed Rows: {results['nan_rows']:,}")
            print(f"🧹 Cleaning Rate: {(results['nan_rows']/results['original_rows']*100):.2f}%")
            
            if 'files' in results:
                print(f"\n📁 FILES CREATED:")
                print(f"   ✅ Clean Dataset: {results['files']['clean_file']}")
                if results['files']['nan_file']:
                    print(f"   ❌ Removed Data: {results['files']['nan_file']}")
                print(f"   📊 Statistics: {results['files']['stats_file']}")
            
            # Show sample of clean data
            print(f"\n📋 SAMPLE CLEAN DATA:")
            print(cleaner.get_sample_data('clean', 3))
            
            if results['nan_rows'] > 0:
                print(f"\n📋 SAMPLE REMOVED DATA:")
                print(cleaner.get_sample_data('nan', 3))
        else:
            print(f"❌ Cleaning failed: {results['error']}")
            
    except Exception as e:
        print(f"❌ Error during cleaning: {e}")

# Usage example for integration with your ML agent
def integrate_with_ml_agent(csv_path: str, target_column: str = "Purchase Amount (USD)"):
    """Function to integrate with your existing ML agent"""
    
    cleaner = CSVDataCleaner()
    results = cleaner.clean_csv(csv_path, target_column, save_results=False)
    
    if results['status'] == 'success':
        # Return clean dataframe that can be used directly in your ML agent
        clean_df = results['clean_data']
        
        logger.info(f"Data cleaning complete: {len(clean_df)} clean rows available")
        return clean_df
    else:
        logger.error(f"Data cleaning failed: {results['error']}")
        return None

if __name__ == "__main__":
    main()
