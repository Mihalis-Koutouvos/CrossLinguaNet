"""
Evaluate Transformer Model Per Language

Evaluates the trained transformer model on each language separately.

Usage:
    python src/models/evaluate_transformer.py
    python src/models/evaluate_transformer.py --model best_model
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
import json
import argparse
from train_transformer import TransformerClarityModel, ClarityDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def load_model(model_path, model_name, device):
    """Load a saved transformer model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = TransformerClarityModel(model_name).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['max_length']


def evaluate_language(model, data_loader, device):
    """Evaluate model on a specific language."""
    all_predictions = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            predictions = model(input_ids, attention_mask)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    mae = mean_absolute_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)
    
    return {
        'mae': mae,
        'r2': r2,
        'n_samples': len(all_labels)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate transformer model per language')
    parser.add_argument('--model', type=str, default='best_model',
                        help='Model checkpoint to load (default: best_model)')
    parser.add_argument('--model-name', type=str, default='xlm-roberta-base',
                        help='Transformer model name (default: xlm-roberta-base)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation (default: 32)')
    
    args = parser.parse_args()
    
    # Paths
    root = Path('.')
    splits_path = root / "data" / "splits"
    models_path = root / "models" / "transformer"
    results_path = root / "artifacts" / "results"
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(splits_path / "test.csv")
    print(f"  Loaded {len(test_df)} test samples")
    
    languages = sorted(test_df['language'].unique())
    print(f"  Languages: {', '.join(languages)}")
    
    # Load model
    model_path = models_path / f"{args.model}.pt"
    print(f"\nLoading model from {model_path}...")
    
    model, max_length = load_model(model_path, args.model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    print("\n" + "="*60)
    print("Per-Language Evaluation")
    print("="*60)
    
    # Evaluate each language
    results = {}
    
    for lang in languages:
        print(f"\nEvaluating {lang.upper()}...")
        
        # Filter data for this language
        lang_df = test_df[test_df['language'] == lang]
        
        # Create dataset and loader
        lang_dataset = ClarityDataset(
            lang_df['text'].tolist(),
            lang_df['clarity_score'].tolist(),
            tokenizer,
            max_length
        )
        
        lang_loader = DataLoader(
            lang_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Evaluate
        metrics = evaluate_language(model, lang_loader, device)
        results[lang] = metrics
        
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²:  {metrics['r2']:.4f}")
        print(f"  N:   {metrics['n_samples']}")
    
    # Print summary
    print("\n" + "="*60)
    print("Cross-Language Performance Summary")
    print("="*60)
    
    print(f"\n{'Language':<12} {'MAE':<10} {'R²':<10} {'N Samples':<10}")
    print("-" * 60)
    
    for lang in languages:
        metrics = results[lang]
        print(f"{lang.upper():<12} {metrics['mae']:<10.4f} {metrics['r2']:<10.4f} {metrics['n_samples']:<10}")
    
    # Calculate average metrics
    avg_mae = np.mean([results[lang]['mae'] for lang in languages])
    avg_r2 = np.mean([results[lang]['r2'] for lang in languages])
    
    print("-" * 60)
    print(f"{'AVERAGE':<12} {avg_mae:<10.4f} {avg_r2:<10.4f}")
    
    # Check for performance disparities
    mae_values = [results[lang]['mae'] for lang in languages]
    mae_std = np.std(mae_values)
    
    print("\n" + "="*60)
    print("Performance Analysis")
    print("="*60)
    print(f"\nMAE Standard Deviation: {mae_std:.4f}")
    
    if mae_std < 0.01:
        print("✓ Consistent performance across languages")
    elif mae_std < 0.05:
        print("⚠ Slight variation in performance across languages")
    else:
        print("❌ Significant performance disparities across languages")
    
    # Find best and worst performing languages
    best_lang = min(languages, key=lambda l: results[l]['mae'])
    worst_lang = max(languages, key=lambda l: results[l]['mae'])
    
    print(f"\nBest Performance:  {best_lang.upper()} (MAE: {results[best_lang]['mae']:.4f})")
    print(f"Worst Performance: {worst_lang.upper()} (MAE: {results[worst_lang]['mae']:.4f})")
    
    # Save results
    results_dict = {
        'model_checkpoint': args.model,
        'model_name': args.model_name,
        'per_language': results,
        'summary': {
            'average_mae': float(avg_mae),
            'average_r2': float(avg_r2),
            'mae_std': float(mae_std),
            'best_language': best_lang,
            'worst_language': worst_lang
        }
    }
    
    output_path = results_path / f"transformer_per_language_{args.model}.json"
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()