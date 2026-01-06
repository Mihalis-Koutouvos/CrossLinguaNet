"""
Per-Language Model Evaluation

Evaluates trained models on each language separately to understand
cross-lingual performance and identify language-specific patterns.

Usage:
    python -m src.models.evaluate_per_language
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json


class PerLanguageEvaluator:
    """Evaluate models separately for each language."""
    
    def __init__(self, project_root: str = "."):
        """Initialize evaluator."""
        self.root = Path(project_root)
        self.splits_path = self.root / "data" / "splits"
        self.models_path = self.root / "models" / "baseline"
        self.results_path = self.root / "artifacts" / "results"
        
        self.results_path.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load test data."""
        print("Loading test data...")
        self.test_df = pd.read_csv(self.splits_path / "test.csv")
        print(f"  Loaded {len(self.test_df)} test samples")
        
        # Get languages
        self.languages = sorted(self.test_df['language'].unique())
        print(f"  Languages: {', '.join(self.languages)}")
    
    def load_model(self, model_path: Path):
        """Load a trained model."""
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        return saved_data
    
    def evaluate_model_per_language(self, model_name: str, model_data: dict):
        """Evaluate a model on each language separately."""
        print(f"\nEvaluating {model_name}...")
        
        model = model_data['model']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
        
        results = {'model': model_name, 'languages': {}}
        
        for lang in self.languages:
            # Get language-specific data
            lang_df = self.test_df[self.test_df['language'] == lang]
            
            # Extract features
            X_lang = lang_df[feature_cols].values
            y_lang = lang_df['clarity_score'].values
            
            # Standardize
            X_lang = scaler.transform(X_lang)
            
            # Predict
            y_pred = model.predict(X_lang)
            
            # Calculate metrics
            metrics = {
                'n_samples': len(lang_df),
                'mse': float(mean_squared_error(y_lang, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_lang, y_pred))),
                'mae': float(mean_absolute_error(y_lang, y_pred)),
                'r2': float(r2_score(y_lang, y_pred))
            }
            
            results['languages'][lang] = metrics
            
            print(f"  {lang.upper()}: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}, n={metrics['n_samples']}")
        
        return results
    
    def evaluate_all_models(self):
        """Evaluate all trained baseline models."""
        print("\n" + "="*60)
        print("Per-Language Model Evaluation")
        print("="*60)
        
        # Find all model files
        model_files = list(self.models_path.glob("*.pkl"))
        
        if not model_files:
            print("No trained models found!")
            print(f"Expected location: {self.models_path}")
            return
        
        print(f"\nFound {len(model_files)} trained models")
        
        all_results = {}
        
        for model_path in model_files:
            model_name = model_path.stem.replace('_', ' ').title()
            model_data = self.load_model(model_path)
            
            results = self.evaluate_model_per_language(model_name, model_data)
            all_results[model_name] = results
        
        # Save results
        self._save_results(all_results)
        
        # Print comparison
        self._print_comparison(all_results)
        
        return all_results
    
    def _save_results(self, all_results):
        """Save per-language results."""
        results_file = self.results_path / "per_language_evaluation.json"
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_file}")
    
    def _print_comparison(self, all_results):
        """Print comparison across models and languages."""
        print("\n" + "="*60)
        print("Cross-Language Performance Comparison")
        print("="*60)
        
        # Create comparison table
        print(f"\n{'Model':<25} {'Language':<10} {'MAE':<10} {'R²':<10}")
        print("-"*60)
        
        for model_name, results in sorted(all_results.items()):
            for lang, metrics in sorted(results['languages'].items()):
                print(f"{model_name:<25} {lang.upper():<10} {metrics['mae']:<10.4f} {metrics['r2']:<10.4f}")
        
        print("="*60)
        
        # Language-specific analysis
        print("\n" + "="*60)
        print("Language-Specific Performance (Average across models)")
        print("="*60)
        
        # Calculate average metrics per language
        lang_metrics = {lang: [] for lang in self.languages}
        
        for results in all_results.values():
            for lang, metrics in results['languages'].items():
                lang_metrics[lang].append(metrics['mae'])
        
        for lang in sorted(self.languages):
            avg_mae = np.mean(lang_metrics[lang])
            std_mae = np.std(lang_metrics[lang])
            print(f"  {lang.upper()}: MAE = {avg_mae:.4f} ± {std_mae:.4f}")


def main():
    """Main evaluation function."""
    evaluator = PerLanguageEvaluator()
    evaluator.load_data()
    evaluator.evaluate_all_models()
    
    print("\n" + "="*60)
    print("Per-Language Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
