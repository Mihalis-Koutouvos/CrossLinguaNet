#Orchestrates data loading, feature extraction, labeling, and splitting.

import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.loader import DataLoader
from data.schema import MedicalInstruction
from features.linguistic import FeaturePipeline
from labeling.clarity_proxy import ClarityProxyLabeler, ClarityAnalyzer
from data.splitter import DataSplitter


class CrossLinguaNetPipeline:
    """
    End-to-end pipeline for CrossLinguaNet data processing.
    """
    
    def __init__(self, project_root: str = "../.."):
        """
        Initialize pipeline.
        
        Args:
            project_root: Root directory of the project (from src/data/)
        """
        self.root = Path(project_root)
        self.raw_data_path = self.root / "data" / "raw"
        self.processed_path = self.root / "data" / "processed"
        self.features_path = self.root / "data" / "features"
        self.splits_path = self.root / "data" / "splits"
        self.reports_path = self.root / "artifacts" / "reports"
        
        # Create directories if they don't exist
        for path in [self.processed_path, self.features_path, self.splits_path, self.reports_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.loader = DataLoader(str(self.raw_data_path))
        self.feature_pipeline = FeaturePipeline()
        self.labeler = ClarityProxyLabeler()
        self.splitter = DataSplitter()
        self.analyzer = ClarityAnalyzer()
    
    def run_full_pipeline(
        self,
        raw_filename: str,
        text_column: str,
        language_column: str,
        id_column: str = None,
        pair_id_column: str = None
    ):
        """
        Run the complete pipeline from raw data to splits.
        
        Args:
            raw_filename: Name of raw CSV file in data/raw/
            text_column: Column containing instruction text
            language_column: Column containing language codes
            id_column: Column containing IDs (optional)
            pair_id_column: Column containing pair IDs (optional)
        """
        print("="*60)
        print("CrossLinguaNet Pipeline")
        print("="*60)
        
        # Step 1: Load and convert to schema
        print("\n[1/6] Loading raw data...")
        instructions = self.loader.load_from_csv(
            filename=raw_filename,
            text_column=text_column,
            language_column=language_column,
            id_column=id_column,
            pair_id_column=pair_id_column
        )
        
        # Convert to dataframe for processing
        data_df = pd.DataFrame([inst.to_dict() for inst in instructions])
        print(f"Loaded {len(data_df)} instructions")
        print(f"Languages: {', '.join(data_df['language'].unique())}")
        
        # Step 2: Extract features
        print("\n[2/6] Extracting linguistic features...")
        features_list = self.feature_pipeline.extract_features_batch(
            texts=data_df['text'].tolist(),
            languages=data_df['language'].tolist()
        )
        features_df = pd.DataFrame(features_list)
        
        # Combine with original data
        combined_df = pd.concat([data_df.reset_index(drop=True), features_df], axis=1)
        print(f"Extracted {len(features_df.columns)} features")
        
        # Save features
        features_path = self.features_path / "linguistic_features.csv"
        combined_df.to_csv(features_path, index=False)
        print(f"Saved features to {features_path}")
        
        # Step 3: Create clarity labels
        print("\n[3/6] Creating clarity proxy labels...")
        labeled_df = self.labeler.create_labels(combined_df)
        
        # Show score distribution
        score_stats = self.labeler.get_score_distribution(labeled_df)
        print("Clarity score distribution by language:")
        for lang, stats in score_stats.items():
            print(f"  {lang.upper()}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, n={stats['count']}")
        
        # Save labeled data
        labeled_path = self.features_path / "labeled_data.csv"
        labeled_df.to_csv(labeled_path, index=False)
        print(f"Saved labeled data to {labeled_path}")
        
        # Step 4: Analyze parallel pairs (if available)
        if pair_id_column and labeled_df['pair_id'].notna().any():
            print("\n[4/6] Analyzing parallel translations...")
            comparisons = self.analyzer.compare_parallel_pairs(labeled_df)
            
            if len(comparisons) > 0:
                summary = self.analyzer.aggregate_language_comparisons(comparisons)
                print(f"Found {len(comparisons)} pairwise comparisons")
                print("\nCross-language clarity deltas:")
                print(summary[['lang_pair', 'mean_clarity_delta', 'num_pairs']].head(10).to_string(index=False))
                
                # Save analysis
                comparisons_path = self.features_path / "parallel_comparisons.csv"
                comparisons.to_csv(comparisons_path, index=False)
                summary_path = self.features_path / "language_pair_summary.csv"
                summary.to_csv(summary_path, index=False)
                print(f"\nSaved analysis to {self.features_path}")
        else:
            print("\n[4/6] No parallel pairs found, skipping comparison analysis")
        
        # Step 5: Create train/val/test splits
        print("\n[5/6] Creating train/val/test splits...")
        train_df, val_df, test_df = self.splitter.split_stratified(
            labeled_df,
            stratify_column='language',
            pair_id_column=pair_id_column if pair_id_column else None
        )
        
        # Save splits
        self.splitter.save_splits(train_df, val_df, test_df, str(self.splits_path))
        
        # Show split statistics
        split_stats = self.splitter.get_split_statistics(train_df, val_df, test_df)
        print("\nLanguage distribution across splits:")
        print(split_stats)
        
        # Step 6: Generate summary report
        print("\n[6/6] Generating summary report...")
        self._generate_summary_report(labeled_df, train_df, val_df, test_df)
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)
        print("\nGenerated files:")
        print(f"  Features: {self.features_path}/")
        print(f"  Splits:   {self.splits_path}/")
        print(f"  Reports:  {self.reports_path}/")
    
    def _generate_summary_report(
        self,
        full_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):
        """Generate a summary report of the dataset."""
        report = []
        report.append("CrossLinguaNet Dataset Summary")
        report.append("=" * 60)
        report.append(f"\nTotal Instructions: {len(full_df)}")
        report.append(f"Languages: {', '.join(sorted(full_df['language'].unique()))}")
        report.append(f"\nClarity Score Range: [{full_df['clarity_score'].min():.3f}, {full_df['clarity_score'].max():.3f}]")
        report.append(f"Mean Clarity Score: {full_df['clarity_score'].mean():.3f}")
        report.append(f"Std Clarity Score: {full_df['clarity_score'].std():.3f}")
        
        report.append("\n\nLanguage Distribution:")
        for lang, count in full_df['language'].value_counts().items():
            pct = count / len(full_df) * 100
            report.append(f"  {lang.upper()}: {count:,} ({pct:.1f}%)")
        
        report.append("\n\nSplit Sizes:")
        report.append(f"  Train: {len(train_df):,} ({len(train_df)/len(full_df)*100:.1f}%)")
        report.append(f"  Val:   {len(val_df):,} ({len(val_df)/len(full_df)*100:.1f}%)")
        report.append(f"  Test:  {len(test_df):,} ({len(test_df)/len(full_df)*100:.1f}%)")
        
        if 'pair_id' in full_df.columns:
            num_pairs = full_df['pair_id'].nunique()
            report.append(f"\n\nParallel Translation Pairs: {num_pairs:,}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # Save report
        report_path = self.reports_path / "dataset_summary.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to {report_path}")


def main():
    """
    Run the pipeline with medical EN/ES/RU dataset.
    """
    # Initialize pipeline
    pipeline = CrossLinguaNetPipeline()
    
    # Run with the medical dataset created by setup_crosslinguanet.py
    pipeline.run_full_pipeline(
        raw_filename="medical_en_es_ru.csv",
        text_column="text",
        language_column="language",
        id_column="id",
        pair_id_column="pair_id"
    )


if __name__ == "__main__":
    main()