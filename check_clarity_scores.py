#!/usr/bin/env python3
"""
Check the clarity scores in the original labeled data.

Usage:
    python check_clarity_scores.py
"""

import pandas as pd
import numpy as np


def check_clarity_scores():
    """Check clarity scores in the labeled data."""
    
    print("="*60)
    print("Clarity Score Analysis")
    print("="*60)
    
    # Load the labeled data (before splitting)
    print("\n[1/3] Loading labeled data...")
    labeled_df = pd.read_csv('data/features/labeled_data.csv')
    print(f"  Shape: {labeled_df.shape}")
    
    # Check clarity_score_raw
    print("\n[2/3] Analyzing clarity_score_raw...")
    if 'clarity_score_raw' in labeled_df.columns:
        raw_scores = labeled_df['clarity_score_raw']
        print(f"  Min:    {raw_scores.min():.6f}")
        print(f"  Max:    {raw_scores.max():.6f}")
        print(f"  Mean:   {raw_scores.mean():.6f}")
        print(f"  Std:    {raw_scores.std():.6f}")
        print(f"  Unique: {raw_scores.nunique()}")
        print(f"  Sample: {raw_scores.head(10).tolist()}")
    else:
        print("  ❌ clarity_score_raw not found!")
    
    # Check clarity_score (normalized)
    print("\n[3/3] Analyzing clarity_score (normalized)...")
    if 'clarity_score' in labeled_df.columns:
        norm_scores = labeled_df['clarity_score']
        print(f"  Min:    {norm_scores.min():.6f}")
        print(f"  Max:    {norm_scores.max():.6f}")
        print(f"  Mean:   {norm_scores.mean():.6f}")
        print(f"  Std:    {norm_scores.std():.6f}")
        print(f"  Unique: {norm_scores.nunique()}")
        print(f"  Sample: {norm_scores.head(10).tolist()}")
        
        # Check if it's constant
        if norm_scores.nunique() == 1:
            print("\n  🚨 PROBLEM: All values are the same!")
            print("  The normalization step collapsed everything to a constant.")
        elif norm_scores.std() < 0.01:
            print("\n  ⚠️  WARNING: Very low variance")
            print("  The normalization may have squashed the values too much.")
        else:
            print("\n  ✓ Looks good!")
    else:
        print("  ❌ clarity_score not found!")
    
    # Check by language
    print("\n[BONUS] By Language:")
    print("-"*60)
    for lang in labeled_df['language'].unique():
        lang_df = labeled_df[labeled_df['language'] == lang]
        if 'clarity_score' in lang_df.columns:
            scores = lang_df['clarity_score']
            print(f"  {lang.upper()}: mean={scores.mean():.6f}, std={scores.std():.6f}, unique={scores.nunique()}")
    
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)
    
    if 'clarity_score' in labeled_df.columns:
        if labeled_df['clarity_score'].nunique() == 1:
            print("\n❌ The clarity_score normalization is broken!")
            print("\nLikely cause: The normalization step in clarity_proxy.py")
            print("is using the same min and max for all values, or")
            print("there's a bug in the scaling logic.")
            print("\nSOLUTION: We need to fix the create_labels() method")
            print("in src/labeling/clarity_proxy.py")
        else:
            print("\n✓ Clarity scores look fine in the original data.")
            print("The problem may be in the splitting step.")


if __name__ == "__main__":
    check_clarity_scores()