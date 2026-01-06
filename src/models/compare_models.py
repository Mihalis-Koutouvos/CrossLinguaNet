"""
Compare Baseline vs Transformer Models

Compares the performance of baseline models and transformer models side-by-side.

Usage:
    python src/models/compare_models.py
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np


def load_baseline_results():
    """Load baseline model results."""
    results_path = Path("artifacts/results")
    
    # Find the most recent baseline results
    baseline_files = list(results_path.glob("baseline_results_*.json"))
    if not baseline_files:
        print("⚠ No baseline results found")
        return None
    
    latest_file = max(baseline_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def load_transformer_results():
    """Load transformer model results."""
    results_path = Path("artifacts/results")
    
    # Find the most recent transformer results
    transformer_files = list(results_path.glob("transformer_results_*.json"))
    if not transformer_files:
        print("⚠ No transformer results found")
        return None
    
    latest_file = max(transformer_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def load_transformer_per_language():
    """Load per-language transformer results."""
    results_path = Path("artifacts/results")
    
    # Find per-language results
    per_lang_files = list(results_path.glob("transformer_per_language_*.json"))
    if not per_lang_files:
        return None
    
    latest_file = max(per_lang_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def main():
    print("="*70)
    print("MODEL COMPARISON: Baseline vs Transformer")
    print("="*70)
    
    # Load results
    baseline_results = load_baseline_results()
    transformer_results = load_transformer_results()
    transformer_per_lang = load_transformer_per_language()
    
    if baseline_results is None or transformer_results is None:
        print("\n❌ Could not load all results. Make sure you've trained both models.")
        return
    
    # Compare overall test performance
    print("\n📊 OVERALL TEST PERFORMANCE")
    print("-"*70)
    
    print(f"\n{'Model':<30} {'MAE':<12} {'R²':<12} {'Type':<15}")
    print("-"*70)
    
    # Best baseline model (lowest test MAE)
    best_baseline = min(
        baseline_results['test_results'].items(),
        key=lambda x: x[1]['mae']
    )
    best_baseline_name, best_baseline_metrics = best_baseline
    
    print(f"{best_baseline_name:<30} {best_baseline_metrics['mae']:<12.4f} {best_baseline_metrics['r2']:<12.4f} {'Baseline':<15}")
    
    # Transformer model
    transformer_mae = transformer_results['test_metrics']['mae']
    transformer_r2 = transformer_results['test_metrics']['r2']
    model_name = transformer_results['model_name']
    
    print(f"{model_name:<30} {transformer_mae:<12.4f} {transformer_r2:<12.4f} {'Transformer':<15}")
    
    # Calculate improvement
    mae_improvement = ((best_baseline_metrics['mae'] - transformer_mae) / best_baseline_metrics['mae']) * 100
    r2_improvement = ((transformer_r2 - best_baseline_metrics['r2']) / abs(best_baseline_metrics['r2'])) * 100
    
    print("\n" + "-"*70)
    if mae_improvement > 0:
        print(f"✓ Transformer improved MAE by {mae_improvement:.1f}%")
    else:
        print(f"⚠ Baseline was better by {abs(mae_improvement):.1f}% (MAE)")
    
    if r2_improvement > 0:
        print(f"✓ Transformer improved R² by {r2_improvement:.1f}%")
    else:
        print(f"⚠ Baseline was better by {abs(r2_improvement):.1f}% (R²)")
    
    # Per-language comparison
    if transformer_per_lang:
        print("\n\n🌍 PER-LANGUAGE PERFORMANCE")
        print("-"*70)
        
        # Load baseline per-language results
        baseline_per_lang_file = Path("artifacts/results/per_language_evaluation.json")
        if baseline_per_lang_file.exists():
            with open(baseline_per_lang_file, 'r') as f:
                baseline_per_lang = json.load(f)
            
            print(f"\n{'Language':<12} {'Baseline MAE':<15} {'Transformer MAE':<18} {'Improvement':<15}")
            print("-"*70)
            
            for lang, metrics in transformer_per_lang['per_language'].items():
                # Find best baseline for this language
                baseline_lang_mae = None
                for model_results in baseline_per_lang.values():
                    if lang in model_results:
                        if baseline_lang_mae is None or model_results[lang]['mae'] < baseline_lang_mae:
                            baseline_lang_mae = model_results[lang]['mae']
                
                if baseline_lang_mae:
                    transformer_lang_mae = metrics['mae']
                    improvement = ((baseline_lang_mae - transformer_lang_mae) / baseline_lang_mae) * 100
                    
                    arrow = "✓" if improvement > 0 else "→"
                    print(f"{lang.upper():<12} {baseline_lang_mae:<15.4f} {transformer_lang_mae:<18.4f} {arrow} {abs(improvement):<.1f}%")
    
    # Training details
    print("\n\n⚙️ TRAINING DETAILS")
    print("-"*70)
    
    print(f"\nTransformer Model: {transformer_results['model_name']}")
    print(f"  Epochs:          {transformer_results['hyperparameters']['num_epochs']}")
    print(f"  Batch Size:      {transformer_results['hyperparameters']['batch_size']}")
    print(f"  Learning Rate:   {transformer_results['hyperparameters']['learning_rate']}")
    print(f"  Max Length:      {transformer_results['hyperparameters']['max_length']}")
    
    # Training history
    if 'history' in transformer_results:
        history = transformer_results['history']
        
        print("\n\n📈 TRAINING PROGRESS")
        print("-"*70)
        print(f"\n{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Val MAE':<15} {'Val R²':<15}")
        print("-"*70)
        
        for i in range(len(history['train_loss'])):
            print(f"{i+1:<8} {history['train_loss'][i]:<15.4f} {history['val_loss'][i]:<15.4f} "
                  f"{history['val_mae'][i]:<15.4f} {history['val_r2'][i]:<15.4f}")
        
        # Check for overfitting
        final_train = history['train_loss'][-1]
        final_val = history['val_loss'][-1]
        overfit_ratio = (final_val / final_train - 1) * 100
        
        print("\n" + "-"*70)
        if overfit_ratio < 5:
            print(f"✓ No overfitting detected (Val/Train ratio: {overfit_ratio:.1f}%)")
        elif overfit_ratio < 20:
            print(f"⚠ Slight overfitting (Val/Train ratio: {overfit_ratio:.1f}%)")
        else:
            print(f"❌ Significant overfitting (Val/Train ratio: {overfit_ratio:.1f}%)")
    
    # Summary
    print("\n\n📝 SUMMARY")
    print("="*70)
    
    if mae_improvement > 5 or r2_improvement > 5:
        print("\n✓ Transformer model shows clear improvement over baselines")
        print("  → Consider using transformer for production")
    elif abs(mae_improvement) < 5 and abs(r2_improvement) < 5:
        print("\n→ Transformer and baseline perform similarly")
        print("  → Consider baseline for faster inference")
    else:
        print("\n⚠ Baseline models performed better")
        print("  → May need more transformer training or hyperparameter tuning")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()