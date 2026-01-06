#Constructs weak supervision target for clarity prediction

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy import stats
from scipy.stats import zscore


class ClarityProxyLabeler:
    """
    Creates clarity proxy scores using linguistic features.
    """
    
    def __init__(self):
        """
        Initialize labeler with feature weights.
        
        Weights are designed so that:
        - Higher score = clearer/simpler
        - Lower score = more complex/unclear
        """
        # Features that INCREASE clarity (positive contribution)
        self.positive_weights = {
            'has_numbered_steps': 1.0,
            'has_bullets': 0.8,
            'type_token_ratio': 0.5,  # More diverse vocab can be clearer
        }
        
        # Features that DECREASE clarity (negative contribution)
        self.negative_weights = {
            'mean_sentence_length': -0.3,
            'std_sentence_length': -0.2,  # High variance = inconsistent structure
            'ambiguous_term_density': -1.5,  # Vague terms reduce clarity
            'long_word_ratio': -0.4,  # Complex vocabulary
            'modal_verb_count': -0.3,  # Uncertainty
            'negation_count': -0.2,  # Negations add complexity
            'max_sentence_length': -0.2,  # Very long sentences
        }
    
    def compute_raw_score(self, features: Dict[str, float]) -> float:
        """
        Compute raw clarity score from features.
        
        Args:
            features: Dict of linguistic features
        
        Returns:
            Raw clarity score (before normalization)
        """
        score = 0.0
        
        # Add positive contributions
        for feature, weight in self.positive_weights.items():
            if feature in features:
                score += features[feature] * weight
        
        # Add negative contributions
        for feature, weight in self.negative_weights.items():
            if feature in features:
                score += features[feature] * weight
        
        return score
    
    def normalize_scores_global(
        self,
        scores: List[float]
    ) -> np.ndarray:
        """
        Min-max normalize scores globally across all languages.
        
        This preserves variation while scaling to [0, 1] range.
        
        Args:
            scores: List of raw clarity scores
        
        Returns:
            Normalized scores as numpy array in [0, 1] range
        """
        scores_array = np.array(scores)
        
        # Get global min and max
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        # Avoid division by zero
        if max_score - min_score < 1e-10:
            return np.full_like(scores_array, 0.5)
        
        # Min-max normalization to [0, 1]
        normalized = (scores_array - min_score) / (max_score - min_score)
        
        return normalized
    
    def create_labels(
        self,
        features_df: pd.DataFrame,
        language_column: str = 'language'
    ) -> pd.DataFrame:
        """
        Create clarity proxy labels for a dataset.
        
        Args:
            features_df: DataFrame with linguistic features and language
            language_column: Name of language column
        
        Returns:
            DataFrame with added 'clarity_score' column
        """
        # Compute raw scores
        raw_scores = []
        for idx, row in features_df.iterrows():
            features = row.to_dict()
            score = self.compute_raw_score(features)
            raw_scores.append(score)
        
        # OPTION 1: Use raw scores directly (no normalization)
        # This is often best for regression tasks
        normalized_scores = raw_scores
        
        # OPTION 2: Use global min-max normalization (uncomment to use)
        # normalized_scores = self.normalize_scores_global(raw_scores)
        
        # Add to dataframe
        result_df = features_df.copy()
        result_df['clarity_score_raw'] = raw_scores
        result_df['clarity_score'] = normalized_scores
        
        return result_df
    
    def get_score_distribution(self, labeled_data, language_column: str = "language"):
        """
        Get distribution statistics for clarity scores.
        """
        import pandas as pd
        
        stats = {}
        
        # Handle dictionary input (the expected format from create_labels when used differently)
        if isinstance(labeled_data, dict):
            for lang, df in labeled_data.items():
                if "clarity_score" in df.columns:
                    scores = df["clarity_score"].dropna()
                    if len(scores) > 0:
                        stats[lang] = {
                            "count": len(scores),
                            "mean": float(scores.mean()),
                            "std": float(scores.std()),
                            "min": float(scores.min()),
                            "max": float(scores.max()),
                            "median": float(scores.median()),
                        }
        # Handle DataFrame input
        else:
            # Ensure language_column is a string
            if isinstance(language_column, list):
                language_column = language_column[0] if language_column else "language"
            
            if language_column in labeled_data.columns:
                # Extract the column safely
                lang_col = labeled_data[language_column]
                
                # If it's a DataFrame, extract the first column
                if isinstance(lang_col, pd.DataFrame):
                    lang_col = lang_col.iloc[:, 0]
                
                # Get unique values
                unique_langs = lang_col.unique()
                
                for lang in unique_langs:
                    lang_df = labeled_data[labeled_data[language_column] == lang]
                    if "clarity_score" in lang_df.columns:
                        scores = lang_df["clarity_score"].dropna()
                        if len(scores) > 0:
                            stats[lang] = {
                                "count": len(scores),
                                "mean": float(scores.mean()),
                                "std": float(scores.std()),
                                "min": float(scores.min()),
                                "max": float(scores.max()),
                                "median": float(scores.median()),
                            }
        
        return stats


class ClarityAnalyzer:
    """
    Analyzes clarity differences across languages.
    """
    
    def __init__(self):
        pass
    
    def compare_parallel_pairs(
        self,
        labeled_df: pd.DataFrame,
        pair_id_column: str = 'pair_id',
        language_column: str = 'language',
        score_column: str = 'clarity_score'
    ) -> pd.DataFrame:
        """
        Compare clarity scores for parallel translations.
        
        Args:
            labeled_df: DataFrame with clarity scores and pair IDs
            pair_id_column: Column containing pair IDs
            language_column: Column containing language codes
            score_column: Column containing clarity scores
        
        Returns:
            DataFrame with pairwise comparisons
        """
        # Filter to only records with pair IDs
        paired_df = labeled_df[labeled_df[pair_id_column].notna()].copy()
        
        if len(paired_df) == 0:
            print("No paired translations found")
            return pd.DataFrame()
        
        # Group by pair_id
        comparisons = []
        
        for pair_id, group in paired_df.groupby(pair_id_column):
            if len(group) < 2:
                continue  # Need at least 2 languages to compare
            
            # Create pairwise comparisons
            languages = group[language_column].values
            scores = group[score_column].values
            
            for i in range(len(languages)):
                for j in range(i + 1, len(languages)):
                    comparisons.append({
                        'pair_id': pair_id,
                        'lang_1': languages[i],
                        'lang_2': languages[j],
                        'score_1': scores[i],
                        'score_2': scores[j],
                        'score_diff': scores[i] - scores[j],
                        'clarity_delta': abs(scores[i] - scores[j])
                    })
        
        return pd.DataFrame(comparisons)
    
    def aggregate_language_comparisons(
        self,
        comparisons_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate clarity deltas across language pairs.
        
        Returns:
            Summary statistics for each language pair
        """
        if len(comparisons_df) == 0:
            return pd.DataFrame()
        
        # Create language pair identifier
        comparisons_df['lang_pair'] = comparisons_df.apply(
            lambda row: f"{row['lang_1']}-{row['lang_2']}", axis=1
        )
        
        # Aggregate by language pair
        summary = comparisons_df.groupby('lang_pair').agg({
            'score_diff': ['mean', 'std', 'median'],
            'clarity_delta': ['mean', 'std', 'median'],
            'pair_id': 'count'
        }).reset_index()
        
        summary.columns = ['lang_pair', 'mean_score_diff', 'std_score_diff', 
                          'median_score_diff', 'mean_clarity_delta', 
                          'std_clarity_delta', 'median_clarity_delta', 'num_pairs']
        
        return summary


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Sample features
    sample_features = pd.DataFrame([
        {
            'language': 'en',
            'mean_sentence_length': 12.0,
            'ambiguous_term_density': 0.05,
            'has_numbered_steps': 1.0,
            'long_word_ratio': 0.15
        },
        {
            'language': 'es',
            'mean_sentence_length': 15.0,
            'ambiguous_term_density': 0.08,
            'has_numbered_steps': 0.0,
            'long_word_ratio': 0.20
        }
    ])
    
    labeler = ClarityProxyLabeler()
    labeled_df = labeler.create_labels(sample_features)
    
    print("Labeled Data:")
    print(labeled_df[['language', 'clarity_score']])
    
    print("\nScore Distribution:")
    print(labeler.get_score_distribution(labeled_df))
