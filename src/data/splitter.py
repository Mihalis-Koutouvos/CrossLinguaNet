#Creates train/val/test splits with language groupings

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import stats
from sklearn.model_selection import train_test_split
from pathlib import Path


class DataSplitter:
    """
    Creates stratified train/val/test splits for multilingual data.
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ):
        """
        Initialize splitter with split ratios.
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            random_state: Random seed for reproducibility
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
    
    def split_stratified(
        self,
        df: pd.DataFrame,
        stratify_column: str = 'language',
        pair_id_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified splits ensuring language distribution is preserved.
        
        Args:
            df: Input dataframe
            stratify_column: Column to stratify by (typically 'language')
            pair_id_column: If provided, keeps pairs together in same split
        
        Returns:
            (train_df, val_df, test_df)
        """
        if pair_id_column and pair_id_column in df.columns:
            # Handle paired data specially
            return self._split_paired_data(df, stratify_column, pair_id_column)
        
        # Standard stratified split
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            train_size=self.train_ratio,
            stratify=df[stratify_column],
            random_state=self.random_state
        )
        
        # Second split: val vs test
        # Adjust ratio for remaining data
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
        
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio_adjusted,
            stratify=temp_df[stratify_column],
            random_state=self.random_state
        )
        
        return train_df, val_df, test_df
    
    def _split_paired_data(
        self,
        df: pd.DataFrame,
        stratify_column: str,
        pair_id_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data while keeping translation pairs together.
        
        This ensures that all translations of the same content
        end up in the same split (train/val/test).
        """
        # Get unique pair IDs
        paired_df = df[df[pair_id_column].notna()].copy()
        unpaired_df = df[df[pair_id_column].isna()].copy()
        
        if len(paired_df) == 0:
            # No pairs, use standard split
            return self.split_stratified(df, stratify_column, pair_id_column=None)
        
        # Get one representative per pair for splitting
        pair_representatives = paired_df.groupby(pair_id_column).first().reset_index()
        
        # Split pair IDs
        train_pairs, temp_pairs = train_test_split(
            pair_representatives[pair_id_column],
            train_size=self.train_ratio,
            random_state=self.random_state
        )
        
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_pairs, test_pairs = train_test_split(
            temp_pairs,
            train_size=val_ratio_adjusted,
            random_state=self.random_state
        )
        
        # Assign all translations of each pair to the same split
        train_df_paired = paired_df[paired_df[pair_id_column].isin(train_pairs)]
        val_df_paired = paired_df[paired_df[pair_id_column].isin(val_pairs)]
        test_df_paired = paired_df[paired_df[pair_id_column].isin(test_pairs)]
        
        # Split unpaired data normally if any exists
        if len(unpaired_df) > 0:
            train_unpaired, val_unpaired, test_unpaired = self.split_stratified(
                unpaired_df, stratify_column, pair_id_column=None
            )
            
            # Combine paired and unpaired
            train_df = pd.concat([train_df_paired, train_unpaired], ignore_index=True)
            val_df = pd.concat([val_df_paired, val_unpaired], ignore_index=True)
            test_df = pd.concat([test_df_paired, test_unpaired], ignore_index=True)
        else:
            train_df = train_df_paired
            val_df = val_df_paired
            test_df = test_df_paired
        
        return train_df, val_df, test_df
    
    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str
    ):
        """
        Save splits to files.
        
        Args:
            train_df, val_df, test_df: Split dataframes
            output_dir: Directory to save splits
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(output_path / 'train.csv', index=False)
        val_df.to_csv(output_path / 'val.csv', index=False)
        test_df.to_csv(output_path / 'test.csv', index=False)
        
        print(f"Saved splits to {output_dir}:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")
        print(f"  Test:  {len(test_df)} samples")
    
    def get_split_statistics(self, train_df, val_df, test_df, stratify_column: str = "language"):
        """
        Get statistics about the data splits.
    
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            stratify_column: Column used for stratification (default: "language")
    
        Returns:
            Dictionary with split statistics
        """
        import pandas as pd
    
        stats = {
            "train": {"total": len(train_df)},
            "val": {"total": len(val_df)},
            "test": {"total": len(test_df)}
        }
    
        # Get statistics for each split
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if stratify_column in split_df.columns:
                # Extract the column safely
                col = split_df[stratify_column]
            
                # Handle case where it's a DataFrame instead of Series
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
            
                # Now get value counts
                lang_counts = col.value_counts()
            
                stats[split_name]["by_language"] = lang_counts.to_dict()
                stats[split_name]["languages"] = list(lang_counts.index)
            else:
                stats[split_name]["by_language"] = {}
                stats[split_name]["languages"] = []
    
        return stats


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Sample data
    sample_data = pd.DataFrame([
        {'id': 'inst_1', 'language': 'en', 'text': 'Take daily', 'pair_id': 'pair_1'},
        {'id': 'inst_2', 'language': 'es', 'text': 'Tomar diario', 'pair_id': 'pair_1'},
        {'id': 'inst_3', 'language': 'en', 'text': 'Avoid alcohol', 'pair_id': 'pair_2'},
        {'id': 'inst_4', 'language': 'es', 'text': 'Evitar alcohol', 'pair_id': 'pair_2'},
        {'id': 'inst_5', 'language': 'en', 'text': 'Consult doctor', 'pair_id': None},
    ] * 20)  # Duplicate for more samples
    
    splitter = DataSplitter()
    train_df, val_df, test_df = splitter.split_stratified(
        sample_data,
        stratify_column='language',
        pair_id_column='pair_id'
    )
    
    print("Split Statistics:")
    print(splitter.get_split_statistics(train_df, val_df, test_df))