#Extracts clarity-related features from medical instructions

import re
import numpy as np
from typing import Dict, List
from collections import Counter
import spacy
from pathlib import Path

class LinguisticFeatureExtractor:
    """
    Extracts linguistic features for clarity assessment.
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        # Vague/ambiguous terms by language
        # Focused on EN, ES, RU
        self.ambiguous_terms = {
            'en': ['as needed', 'regularly', 'often', 'sometimes', 'avoid', 
                   'consult', 'may', 'might', 'should', 'approximately',
                   'usually', 'generally', 'possibly', 'if necessary'],
            'es': ['según sea necesario', 'regularmente', 'a menudo', 'evitar',
                   'consultar', 'puede', 'debería', 'aproximadamente',
                   'generalmente', 'usualmente', 'posiblemente', 'a veces',
                   'si es necesario'],
            'ru': ['по мере необходимости', 'регулярно', 'часто', 'избегать',
                   'проконсультироваться', 'может', 'следует', 'примерно',
                   'обычно', 'иногда', 'возможно', 'при необходимости'],
        }
        
        # Step/instruction indicators
        self.step_patterns = [
            r'\d+\.',  # "1.", "2."
            r'step \d+',
            r'\d+\)',  # "1)", "2)"
            r'first|second|third|finally|then|next',
        ]
    
    def extract_structural_features(self, text: str) -> Dict[str, float]:
        """
        Extract structure-based features.
        
        Returns:
            Dict of structural features
        """
        sentences = self._split_sentences(text)
        
        features = {}
        
        # Sentence statistics
        sent_lengths = [len(s.split()) for s in sentences]
        features['num_sentences'] = len(sentences)
        features['mean_sentence_length'] = np.mean(sent_lengths) if sent_lengths else 0
        features['std_sentence_length'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
        features['max_sentence_length'] = max(sent_lengths) if sent_lengths else 0
        
        # Structure indicators
        features['has_numbered_steps'] = float(bool(re.search(r'\d+\.', text)))
        features['has_bullets'] = float(bool(re.search(r'[•\-\*]', text)))
        features['has_parentheses'] = float(bool(re.search(r'\([^)]+\)', text)))
        
        # Punctuation density
        features['punctuation_density'] = len(re.findall(r'[.,;:!?]', text)) / max(len(text.split()), 1)
        
        # Text length
        features['num_tokens'] = len(text.split())
        features['num_characters'] = len(text)
        
        return features
    
    def extract_lexical_features(self, text: str, language: str = 'en') -> Dict[str, float]:
        """
        Extract lexical/vocabulary features.
        
        Args:
            text: Input text
            language: Language code
        
        Returns:
            Dict of lexical features
        """
        tokens = text.lower().split()
        features = {}
        
        # Basic lexical stats
        features['num_unique_tokens'] = len(set(tokens))
        features['type_token_ratio'] = len(set(tokens)) / max(len(tokens), 1)
        
        # Word length stats
        word_lengths = [len(word) for word in tokens]
        features['mean_word_length'] = np.mean(word_lengths) if word_lengths else 0
        features['max_word_length'] = max(word_lengths) if word_lengths else 0
        
        # Long word ratio (>7 characters)
        features['long_word_ratio'] = sum(1 for w in tokens if len(w) > 7) / max(len(tokens), 1)
        
        # Capitalization (medical terms often capitalized)
        features['capitalized_ratio'] = sum(1 for w in text.split() if w and w[0].isupper()) / max(len(tokens), 1)
        
        return features
    
    def extract_ambiguity_features(self, text: str, language: str = 'en') -> Dict[str, float]:
        """
        Extract features related to linguistic ambiguity.
        
        Args:
            text: Input text
            language: Language code (en, es, ru)
        
        Returns:
            Dict of ambiguity features
        """
        text_lower = text.lower()
        tokens = text_lower.split()
        
        features = {}
        
        # Ambiguous term count
        ambiguous_terms = self.ambiguous_terms.get(language, [])
        ambiguous_count = sum(1 for term in ambiguous_terms if term in text_lower)
        features['ambiguous_term_count'] = ambiguous_count
        features['ambiguous_term_density'] = ambiguous_count / max(len(tokens), 1)
        
        # Modal verbs by language
        if language == 'en':
            modals = ['may', 'might', 'could', 'should', 'would', 'must']
        elif language == 'es':
            modals = ['puede', 'podría', 'debería', 'debe', 'deberá']
        elif language == 'ru':
            modals = ['может', 'должен', 'следует', 'надо', 'нужно']
        else:
            modals = []
        
        features['modal_verb_count'] = sum(1 for word in tokens if word in modals)
        
        # Negation by language
        if language == 'en':
            negations = ['not', 'no', 'never', 'without', 'none', "don't", "doesn't"]
        elif language == 'es':
            negations = ['no', 'nunca', 'sin', 'ningún', 'ninguno', 'jamás']
        elif language == 'ru':
            negations = ['не', 'нет', 'никогда', 'без', 'ни']
        else:
            negations = []
        
        features['negation_count'] = sum(1 for word in tokens if word in negations)
        
        return features
    
    def extract_all_features(self, text: str, language: str = 'en') -> Dict[str, float]:
        """
        Extract all linguistic features.
        
        Args:
            text: Input text
            language: Language code
        
        Returns:
            Dict of all features combined
        """
        features = {}
        
        # Combine all feature groups
        features.update(self.extract_structural_features(text))
        features.update(self.extract_lexical_features(text, language))
        features.update(self.extract_ambiguity_features(text, language))
        
        return features
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitter.
        """
        # Split on common sentence endings
        sentences = re.split(r'[.!?]+', text)
        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences


class FeaturePipeline:
    """
    Pipeline for extracting features from multiple instructions.
    """
    
    def __init__(self):
        self.extractor = LinguisticFeatureExtractor()
    
    def extract_features_batch(
        self,
        texts: List[str],
        languages: List[str]
    ) -> List[Dict[str, float]]:
        """
        Extract features from a batch of texts.
        
        Args:
            texts: List of instruction texts
            languages: Corresponding language codes
        
        Returns:
            List of feature dictionaries
        """
        features_list = []
        
        for text, lang in zip(texts, languages):
            try:
                features = self.extractor.extract_all_features(text, lang)
                features['language'] = lang  # Keep track of language
                features_list.append(features)
            except Exception as e:
                print(f"Error extracting features: {e}")
                features_list.append({})
        
        return features_list
    
    def features_to_dataframe(self, features_list: List[Dict[str, float]]):
        """
        Convert feature dictionaries to pandas DataFrame.
        """
        import pandas as pd
        return pd.DataFrame(features_list)


if __name__ == "__main__":
    # Example usage
    extractor = LinguisticFeatureExtractor()
    
    sample_text = """
    Take one tablet daily with food. Do not exceed the recommended dose. 
    If you experience side effects, consult your doctor immediately.
    Store in a cool, dry place.
    """
    
    features = extractor.extract_all_features(sample_text, language='en')
    
    print("Extracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.3f}")