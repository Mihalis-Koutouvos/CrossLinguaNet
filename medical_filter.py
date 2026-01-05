"""
Simple Medical Filter - Bare essentials only.
Downloads spa.txt and rus.txt → Creates medical_en_es_ru.csv
"""

import csv
from pathlib import Path

# Medical keywords for filtering
MEDICAL_KEYWORDS = {
    'en': ['patient', 'doctor', 'medicine', 'treatment', 'dose', 'tablet', 
           'hospital', 'prescription', 'medication', 'symptom'],
    'es': ['paciente', 'médico', 'medicina', 'tratamiento', 'dosis', 'tableta',
           'hospital', 'receta', 'medicamento', 'síntoma'],
    'ru': ['пациент', 'врач', 'лекарство', 'лечение', 'доза', 'таблетка',
           'больница', 'рецепт', 'препарат', 'симптом']
}

def has_medical_keyword(text, lang):
    """Check if text contains any medical keyword."""
    text_lower = text.lower()
    keywords = MEDICAL_KEYWORDS[lang]
    return any(kw in text_lower for kw in keywords)

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
    print(f"Reading {filepath.name}...")
    medical_pairs = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Stop when we have enough
            if len(medical_pairs) >= max_pairs:
                break
            
            # Progress indicator
            if line_num % 100000 == 0:
                print(f"  Scanned {line_num:,} lines, found {len(medical_pairs)} medical...")
            
            # Parse tab-separated line
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            
            text1, text2 = parts[0], parts[1]
            
            # Keep if either language has medical keywords
            if has_medical_keyword(text1, lang1) or has_medical_keyword(text2, lang2):
                medical_pairs.append((text1, text2))
    
    print(f"  ✓ Found {len(medical_pairs)} medical pairs\n")
    return medical_pairs

def create_csv(en_es_pairs, en_ru_pairs, output_path):
    """
    Write medical pairs to CSV with proper schema.
    
    Schema: id, language, text, pair_id, medical_score
    """
    print(f"Creating {output_path}...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['id', 'language', 'text', 'pair_id', 'medical_score'])
        
        # EN-ES pairs
        for i, (en_text, es_text) in enumerate(en_es_pairs):
            pair_id = f"pair_{i}"
            writer.writerow([f"{pair_id}_en", 'en', en_text, pair_id, 0.5])
            writer.writerow([f"{pair_id}_es", 'es', es_text, pair_id, 0.5])
        
        # EN-RU pairs  
        offset = len(en_es_pairs)
        for i, (en_text, ru_text) in enumerate(en_ru_pairs):
            pair_id = f"pair_{offset + i}"
            writer.writerow([f"{pair_id}_en", 'en', en_text, pair_id, 0.5])
            writer.writerow([f"{pair_id}_ru", 'ru', ru_text, pair_id, 0.5])
    
    total_rows = len(en_es_pairs) * 2 + len(en_ru_pairs) * 2
    print(f"✓ Done! Created {total_rows:,} rows\n")

def main():
    """Main execution."""
    print("="*60)
    print("CrossLinguaNet: Medical Data Filter")
    print("="*60)
    print()
    
    # File paths
    kaggle_dir = Path("data/raw/kaggle_parallel")
    spa_file = kaggle_dir / "spa.txt"
    rus_file = kaggle_dir / "rus.txt"
    output_file = Path("data/raw/medical_en_es_ru.csv")
    
    # Check files exist
    if not spa_file.exists():
        print(f"ERROR: {spa_file} not found!")
        print("\nDownload from:")
        print("https://www.kaggle.com/datasets/hgultekin/paralel-translation-corpus-in-22-languages")
        print(f"\nExtract spa.txt to: {kaggle_dir}/")
        return
    
    if not rus_file.exists():
        print(f"ERROR: {rus_file} not found!")
        return
    
    # Filter for medical content
    print("[1/3] Filtering English-Spanish pairs...")
    en_es = filter_parallel_file(spa_file, 'en', 'es', max_pairs=10000)
    
    print("[2/3] Filtering English-Russian pairs...")
    en_ru = filter_parallel_file(rus_file, 'en', 'ru', max_pairs=10000)
    
    print("[3/3] Writing CSV...")
    create_csv(en_es, en_ru, output_file)
    
    # Summary
    print("="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"Created: {output_file}")
    print(f"  EN-ES pairs: {len(en_es):,}")
    print(f"  EN-RU pairs: {len(en_ru):,}")
    print(f"  Total rows: {(len(en_es) + len(en_ru)) * 2:,}")
    print()
    print("Next step:")
    print("  cd src/data")
    print("  python pipeline.py")

if __name__ == "__main__":
    main()