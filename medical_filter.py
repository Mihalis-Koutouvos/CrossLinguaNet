#This file is for creating our dataset

import csv 
import pathlib

#Medical keywords to look out for across English, Spanish, and Russian. This will be used 
#for filtering across this project
MEDICAL_KEYWORDS = {
    'en': ['patient', 'doctor', 'hospital', 'medicine', 'medication', 'treatment', 
           'therapy', 'diagnosis', 'symptom', 'disease', 'prescription', 'dose',
           'tablet', 'pill', 'injection', 'take', 'administer', 'consult'],
    'es': ['paciente', 'médico', 'doctor', 'hospital', 'medicina', 'medicamento',
           'tratamiento', 'terapia', 'diagnóstico', 'síntoma', 'enfermedad',
           'receta', 'dosis', 'tableta', 'píldora', 'tomar', 'consultar'],
    'ru': ['пациент', 'врач', 'больница', 'лекарство', 'препарат', 'лечение',
           'терапия', 'диагноз', 'симптом', 'заболевание', 'рецепт', 'доза',
           'таблетка', 'принимать', 'применять']
}

def is_valid_medical_sentence(text, lang):
    """Check if a sentence contains any medical keywords for the given language."""
    text_lower = text.lower()
    keywords = MEDICAL_KEYWORDS.get(lang, [])
    return any(keyword in text_lower for keyword in keywords)

def filter_parallel_file(filepath, lang1, lang2, max_pairs=10000):
    """
    Read parallel corpus file and keep only medical sentences.
    
    Args:
        filepath: Path to .txt file (tab-separated)
        lang1, lang2: Language codes ('en', 'es', 'ru')
        max_pairs: How many pairs to collect
    
    Returns:
        List of (text1, text2) tuples
    """
    print(f'Reading {filepath.name}...') 
    medical_pairs = []   


def main():
    return

if __name__ == "__main__":
    main()