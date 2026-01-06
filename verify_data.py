#!/usr/bin/env python3
"""
Quick verification script to check your processed data.
Run this to make sure everything looks good before moving to training.

Usage:
    python verify_data.py
"""

import pandas as pd
from pathlib import Path

def verify_data():
    """Verify the processed data looks correct."""
    
    print("="*60)
    print("CrossLinguaNet Data Verification")
    print("="*60)
    
    # Check if files exist
    print("\n[1/5] Checking files exist...")
    required_files = {
        'train': 'data/splits/train.csv',
        'val': 'data/splits/val.csv',
        'test': 'data/splits/test.csv',
        'labeled': 'data/features/labeled_data.csv',
        'summary': 'artifacts/reports/dataset_summary.txt'
    }
    
    for name, filepath in required_files.items():
        path = Path(filepath)
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {name}: {filepath}")
    
    # Load data
    print("\n[2/5] Loading splits...")
    train_df = pd.read_csv('data/splits/train.csv')
    val_df = pd.read_csv('data/splits/val.csv')
    test_df = pd.read_csv('data/splits/test.csv')
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    print(f"  Total: {len(train_df) + len(val_df) + len(test_df):,} samples")
    
    # Check columns
    print("\n[3/5] Checking required columns...")
    required_columns = ['text', 'language', 'clarity_score']
    for col in required_columns:
        status = "✓" if col in train_df.columns else "✗"
        print(f"  {status} {col}")
    
    # Check data quality
    print("\n[4/5] Checking data quality...")
    
    # No missing texts
    missing_texts = train_df['text'].isna().sum()
    print(f"  Missing texts: {missing_texts} {'✓' if missing_texts == 0 else '✗'}")
    
    # No missing languages
    missing_langs = train_df['language'].isna().sum()
    print(f"  Missing languages: {missing_langs} {'✓' if missing_langs == 0 else '✗'}")
    
    # No missing clarity scores
    missing_scores = train_df['clarity_score'].isna().sum()
    print(f"  Missing clarity scores: {missing_scores} {'✓' if missing_scores == 0 else '✗'}")
    
    # Check language distribution
    print("\n[5/5] Language distribution in train set:")
    lang_counts = train_df['language'].value_counts()
    for lang, count in lang_counts.items():
        pct = count / len(train_df) * 100
        print(f"  {lang.upper()}: {count:,} ({pct:.1f}%)")
    
    # Clarity score statistics
    print("\n[BONUS] Clarity score statistics:")
    print(f"  Mean:   {train_df['clarity_score'].mean():.3f}")
    print(f"  Std:    {train_df['clarity_score'].std():.3f}")
    print(f"  Min:    {train_df['clarity_score'].min():.3f}")
    print(f"  Max:    {train_df['clarity_score'].max():.3f}")
    print(f"  Median: {train_df['clarity_score'].median():.3f}")
    
    # Sample data
    print("\n[SAMPLE] First training example:")
    print("-" * 60)
    sample = train_df.iloc[0]
    print(f"Language: {sample['language']}")
    print(f"Clarity:  {sample['clarity_score']:.3f}")
    print(f"Text:     {sample['text'][:100]}...")
    print("-" * 60)
    
    print("\n✓ Data verification complete!")
    print("="*60)
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


if __name__ == "__main__":
    data = verify_data()
