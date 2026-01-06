"""
Baseline Models for Clarity Score Prediction

This script trains simple baseline models (logistic regression, random forest, etc.)
to establish baseline performance before moving to neural models.

Usage:
    python -m src.models.train_baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import json
import pickle
from datetime import datetime


class BaselineTrainer:
    """Train and evaluate baseline models for clarity prediction."""
    
    def __init__(self, project_root: str = "."):
        """Initialize trainer with project paths."""
        self.root = Path(project_root)
        self.splits_path = self.root / "data" / "splits"
        self.models_path = self.root / "models" / "baseline"
        self.results_path = self.root / "artifacts" / "results"
        
        # Create directories
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load train/val/test splits."""
        print("Loading data...")
        self.train_df = pd.read_csv(self.splits_path / "train.csv")
        self.val_df = pd.read_csv(self.splits_path / "val.csv")
        self.test_df = pd.read_csv(self.splits_path / "test.csv")
        
        print(f"  Train: {len(self.train_df)} samples")
        print(f"  Val:   {len(self.val_df)} samples")
        print(f"  Test:  {len(self.test_df)} samples")
    
    def prepare_features(self):
        """Extract feature columns and prepare data for modeling."""
        print("\nPreparing features...")
        
        # Define columns to exclude (metadata and target)
        exclude_cols = ['id', 'text', 'language', 'pair_id', 'clarity_score']
        
        # Get all column names
        all_cols = self.train_df.columns.tolist()
        
        # Filter to only numeric columns, excluding the ones we want to exclude
        self.feature_cols = []
        for col in all_cols:
            if col not in exclude_cols:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(self.train_df[col]):
                    self.feature_cols.append(col)
                else:
                    print(f"  ⚠️  Skipping non-numeric column: {col}")
        
        print(f"  Using {len(self.feature_cols)} features")
        print(f"  Features: {', '.join(self.feature_cols[:5])}...")
        
        # Extract features and target
        self.X_train = self.train_df[self.feature_cols].values
        self.y_train = self.train_df['clarity_score'].values
        
        self.X_val = self.val_df[self.feature_cols].values
        self.y_val = self.val_df['clarity_score'].values
        
        self.X_test = self.test_df[self.feature_cols].values
        self.y_test = self.test_df['clarity_score'].values
        
        # Check for NaN values
        if np.isnan(self.X_train).any():
            print("  ⚠️  Warning: NaN values found in training features, filling with 0")
            self.X_train = np.nan_to_num(self.X_train, 0)
            self.X_val = np.nan_to_num(self.X_val, 0)
            self.X_test = np.nan_to_num(self.X_test, 0)
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("  ✓ Features standardized")
    
    def train_model(self, model_name: str, model):
        """Train a single model and evaluate on val set."""
        print(f"\nTraining {model_name}...")
        
        # Train
        model.fit(self.X_train, self.y_train)
        
        # Predict on all sets
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)
        test_pred = model.predict(self.X_test)
        
        # Calculate metrics
        results = {
            'model': model_name,
            'train': self._calculate_metrics(self.y_train, train_pred),
            'val': self._calculate_metrics(self.y_val, val_pred),
            'test': self._calculate_metrics(self.y_test, test_pred)
        }
        
        # Print results
        print(f"  Train - MSE: {results['train']['mse']:.4f}, MAE: {results['train']['mae']:.4f}, R²: {results['train']['r2']:.4f}")
        print(f"  Val   - MSE: {results['val']['mse']:.4f}, MAE: {results['val']['mae']:.4f}, R²: {results['val']['r2']:.4f}")
        print(f"  Test  - MSE: {results['test']['mse']:.4f}, MAE: {results['test']['mae']:.4f}, R²: {results['test']['r2']:.4f}")
        
        # Save model
        model_path = self.models_path / f"{model_name.lower().replace(' ', '_')}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': self.scaler,
                'feature_cols': self.feature_cols,
                'results': results
            }, f)
        print(f"  ✓ Model saved to {model_path}")
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics."""
        return {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred))
        }
    
    def train_all_models(self):
        """Train all baseline models."""
        print("\n" + "="*60)
        print("Training Baseline Models")
        print("="*60)
        
        # Define models to train
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        }
        
        # Train each model
        all_results = {}
        for model_name, model in models.items():
            results = self.train_model(model_name, model)
            all_results[model_name] = results
        
        # Save all results
        self._save_results(all_results)
        
        # Print comparison
        self._print_comparison(all_results)
        
        return all_results
    
    def _save_results(self, all_results):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_path / f"baseline_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_file}")
    
    def _print_comparison(self, all_results):
        """Print comparison table of all models."""
        print("\n" + "="*60)
        print("Model Comparison (Validation Set)")
        print("="*60)
        print(f"{'Model':<25} {'MSE':<10} {'MAE':<10} {'R²':<10}")
        print("-"*60)
        
        # Sort by validation MAE
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['val']['mae'])
        
        for model_name, results in sorted_results:
            val_metrics = results['val']
            print(f"{model_name:<25} {val_metrics['mse']:<10.4f} {val_metrics['mae']:<10.4f} {val_metrics['r2']:<10.4f}")
        
        print("="*60)
        
        # Identify best model
        best_model = sorted_results[0][0]
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   Val MAE: {sorted_results[0][1]['val']['mae']:.4f}")
        print(f"   Val R²:  {sorted_results[0][1]['val']['r2']:.4f}")


def main():
    """Main training function."""
    # Initialize trainer
    trainer = BaselineTrainer()
    
    # Load data
    trainer.load_data()
    
    # Prepare features
    trainer.prepare_features()
    
    # Train all models
    results = trainer.train_all_models()
    
    print("\n" + "="*60)
    print("Baseline Training Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Check results in artifacts/results/")
    print("  2. Review model performance by language")
    print("  3. Move on to neural models (BERT, XLM-R)")


if __name__ == "__main__":
    main()
