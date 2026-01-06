#!/usr/bin/env python3
"""
View and analyze baseline training results.

Usage:
    python view_results.py
"""

import json
import pandas as pd
from pathlib import Path


def view_results():
    """View the latest baseline training results."""
    
    print("="*60)
    print("Baseline Model Results Summary")
    print("="*60)
    
    # Find the latest results file
    results_path = Path("artifacts/results")
    result_files = list(results_path.glob("baseline_results_*.json"))
    
    if not result_files:
        print("❌ No results found!")
        print(f"Expected location: {results_path}")
        return
    
    # Get the most recent file
    latest_file = sorted(result_files)[-1]
    print(f"\nReading: {latest_file.name}")
    print("-"*60)
    
    # Load results
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, model_results in results.items():
        comparison_data.append({
            'Model': model_name,
            'Train MAE': model_results['train']['mae'],
            'Train R²': model_results['train']['r2'],
            'Val MAE': model_results['val']['mae'],
            'Val R²': model_results['val']['r2'],
            'Test MAE': model_results['test']['mae'],
            'Test R²': model_results['test']['r2']
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Val MAE')  # Sort by validation MAE
    
    # Print results
    print("\n📊 MODEL PERFORMANCE COMPARISON")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    # Highlight best model
    best_model = df.iloc[0]
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   Validation MAE: {best_model['Val MAE']:.4f}")
    print(f"   Validation R²:  {best_model['Val R²']:.4f}")
    print(f"   Test MAE:       {best_model['Test MAE']:.4f}")
    print(f"   Test R²:        {best_model['Test R²']:.4f}")
    
    # Check for overfitting
    print("\n📈 OVERFITTING CHECK")
    print("="*60)
    for _, row in df.iterrows():
        train_mae = row['Train MAE']
        val_mae = row['Val MAE']
        diff = val_mae - train_mae
        pct_diff = (diff / train_mae) * 100
        
        status = "✓ Good" if pct_diff < 20 else "⚠️ Warning" if pct_diff < 50 else "❌ Overfitting"
        print(f"{row['Model']:<25} Train: {train_mae:.4f} → Val: {val_mae:.4f} ({pct_diff:+.1f}%) {status}")
    
    # Model interpretation
    print("\n💡 INTERPRETATION")
    print("="*60)
    print("MAE (Mean Absolute Error): Lower is better")
    print("  - How far off predictions are on average")
    print("  - In your case: average error in clarity score")
    print("\nR² (R-squared): Higher is better (0 to 1)")
    print("  - How much variance the model explains")
    print("  - 0.5 = explains 50% of variance")
    print("  - 0.8+ = very good fit")
    
    # Check trained models
    print("\n📦 TRAINED MODELS")
    print("="*60)
    models_path = Path("models/baseline")
    model_files = list(models_path.glob("*.pkl"))
    print(f"Location: {models_path}/")
    for model_file in sorted(model_files):
        size_kb = model_file.stat().st_size / 1024
        print(f"  ✓ {model_file.name:<30} ({size_kb:.1f} KB)")
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("  1. Run per-language evaluation:")
    print("     python -m src.models.evaluate_per_language")
    print("  2. Analyze feature importance")
    print("  3. Move to neural models (BERT, XLM-R)")
    print("="*60)


if __name__ == "__main__":
    view_results()
