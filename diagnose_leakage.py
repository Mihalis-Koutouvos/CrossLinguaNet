"""
Diagnose data leakage issues in the training data.

Usage:
    python diagnose_leakage.py
"""

import pandas as pd
import numpy as np


def diagnose_leakage():
    """Check for data leakage in the training data."""
    
    print("="*60)
    print("Data Leakage Diagnostic")
    print("="*60)
    
    # Load data
    print("\n[1/5] Loading data...")
    train_df = pd.read_csv('data/splits/train.csv')
    print(f"  Train shape: {train_df.shape}")
    print(f"  Columns: {len(train_df.columns)}")
    
    # Check columns
    print("\n[2/5] Checking all columns...")
    print(f"{'Column Name':<30} {'Type':<15} {'Unique Values':<15} {'Sample Value'}")
    print("-"*80)
    
    for col in train_df.columns:
        dtype = str(train_df[col].dtype)
        n_unique = train_df[col].nunique()
        sample = str(train_df[col].iloc[0])[:30]
        print(f"{col:<30} {dtype:<15} {n_unique:<15} {sample}")
    
    # Check target variable
    print("\n[3/5] Analyzing target variable (clarity_score)...")
    clarity = train_df['clarity_score']
    print(f"  Min:    {clarity.min()}")
    print(f"  Max:    {clarity.max()}")
    print(f"  Mean:   {clarity.mean():.6f}")
    print(f"  Std:    {clarity.std():.6f}")
    print(f"  Unique: {clarity.nunique()}")
    print(f"  Sample values: {clarity.head(10).tolist()}")
    
    # Check for constant values
    print("\n[4/5] Checking for constant/near-constant values...")
    constant_cols = []
    for col in train_df.columns:
        if col not in ['id', 'text', 'language', 'pair_id']:
            if pd.api.types.is_numeric_dtype(train_df[col]):
                n_unique = train_df[col].nunique()
                if n_unique <= 1:
                    constant_cols.append(col)
                    print(f"  ⚠️  {col}: only {n_unique} unique value(s)")
    
    # Check correlation with target
    print("\n[5/5] Checking correlation with clarity_score...")
    print(f"{'Feature':<30} {'Correlation':<15} {'Status'}")
    print("-"*60)
    
    exclude_cols = ['id', 'text', 'language', 'pair_id', 'clarity_score']
    
    suspicious_features = []
    for col in train_df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[col]):
            corr = train_df[col].corr(train_df['clarity_score'])
            if abs(corr) > 0.99:
                status = "🚨 PERFECT CORRELATION!"
                suspicious_features.append(col)
            elif abs(corr) > 0.8:
                status = "⚠️  Very High"
            elif abs(corr) > 0.5:
                status = "✓ Moderate"
            else:
                status = "✓ Normal"
            
            print(f"{col:<30} {corr:>10.6f}     {status}")
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    if suspicious_features:
        print("\n🚨 DATA LEAKAGE DETECTED!")
        print("\nThe following features have perfect/near-perfect correlation")
        print("with the target variable (clarity_score):")
        for feat in suspicious_features:
            print(f"  - {feat}")
        
        print("\nPOSSIBLE CAUSES:")
        print("  1. The feature IS the target (or derived directly from it)")
        print("  2. The feature was calculated using future information")
        print("  3. The feature contains the answer")
        
        print("\nSOLUTION:")
        print("  Remove these features from training!")
        print("  They give the model the answer, not useful patterns.")
    else:
        print("\n✓ No obvious data leakage detected")
    
    if constant_cols:
        print(f"\n⚠️  Found {len(constant_cols)} constant features")
        print("  (These won't help the model but won't hurt either)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    diagnose_leakage()