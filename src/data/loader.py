#Data Loader
#Loads raw datasets and converts them to unified schema

import pandas as pd
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from .schema import MedicalInstruction, validate_instruction

class DataLoader:
    """
    Loads and processes raw medical instruction datasets
    """

    def __init__(self, raw_data_path: str):
        """
        Initialize DataLoader with path containing raw data files.

        Args:
            raw_data_path: Directory path containing raw data files
        """

        self.raw_data_path = Path(raw_data_path)
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data path {raw_data_path} does not exist.")
        
    def load_from_csv(self, filename: str, text_column: str, language_column: str,
                 id_column: Optional[str] = None, pair_id_column: Optional[str] = None) -> List[MedicalInstruction]:
        """
        Load data from CSV and convert to unified schema.
        
        Args:
            filename: Name of CSV file in raw_data_path
            text_column: Column containing medical instruction text
            language_column: Column containing language code
            id_column: Column containing unique ID (auto-generated if None)
            pair_id_column: Column containing pair ID for translations
        
        Returns:
            List of MedicalInstruction objects
        """

        filepath = self.raw_data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found in {self.raw_data_path}")

        print(f'Loading data from {filepath}...')   
        df = pd.read_csv(filepath)

        #Validate columns:
        required = [text_column, language_column]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        instructions = []
        errors = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting to schema"):
            # Generate or extract ID
            if id_column and id_column in df.columns:
                inst_id = str(row[id_column])
            else:
                inst_id = f"inst_{idx}"
            
            # Extract pair_id if available
            pair_id = None
            if pair_id_column and pair_id_column in df.columns:
                pair_id = str(row[pair_id_column]) if pd.notna(row[pair_id_column]) else None
            
            # Create instruction
            instruction = MedicalInstruction(
                id=inst_id,
                language=str(row[language_column]).lower().strip(),
                text=str(row[text_column]).strip(),
                pair_id=pair_id
            )
            
            # Validate
            is_valid, error = validate_instruction(instruction)
            if is_valid:
                instructions.append(instruction)
            else:
                errors.append((idx, error))
        
        if errors:
            print(f"Warning: {len(errors)} rows failed validation")
            print(f"First 5 errors: {errors[:5]}")
        
        print(f"Successfully loaded {len(instructions)} instructions")
        return instructions
    
    def load_multilingual_parallel(
        self,
        filename: str,
        language_columns: dict
    ) -> List[MedicalInstruction]:
        """
        Load parallel translations from a wide-format CSV.
        
        Args:
            filename: Name of CSV file
            language_columns: Dict mapping language codes to column names
                             e.g., {'en': 'english_text', 'es': 'spanish_text'}
        
        Returns:
            List of MedicalInstruction objects
        """
        filepath = self.raw_data_path / filename
        df = pd.read_csv(filepath)
        
        instructions = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing parallel data"):
            pair_id = f"pair_{idx}"
            
            for lang_code, col_name in language_columns.items():
                if col_name not in df.columns:
                    print(f"Warning: Column {col_name} not found, skipping {lang_code}")
                    continue
                
                text = str(row[col_name]).strip()
                if pd.isna(row[col_name]) or not text:
                    continue
                
                instruction = MedicalInstruction(
                    id=f"{pair_id}_{lang_code}",
                    language=lang_code.lower(),
                    text=text,
                    pair_id=pair_id
                )
                
                is_valid, _ = validate_instruction(instruction)
                if is_valid:
                    instructions.append(instruction)
        
        print(f"Loaded {len(instructions)} parallel instructions")
        return instructions
    
    def save_processed(
        self,
        instructions: List[MedicalInstruction],
        output_path: str,
        format: str = 'jsonl'
    ):
        """
        Save processed instructions to file.
        
        Args:
            instructions: List of MedicalInstruction objects
            output_path: Path to save file
            format: 'jsonl' or 'csv'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for inst in instructions:
                    f.write(inst.to_json() + '\n')
            print(f"Saved {len(instructions)} instructions to {output_path}")
        
        elif format == 'csv':
            df = pd.DataFrame([inst.to_dict() for inst in instructions])
            df.to_csv(output_path, index=False)
            print(f"Saved {len(instructions)} instructions to {output_path}")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_language_distribution(
        self,
        instructions: List[MedicalInstruction]
    ) -> dict:
        """
        Get distribution of languages in the dataset.
        
        Returns:
            Dict mapping language codes to counts
        """
        lang_counts = {}
        for inst in instructions:
            lang_counts[inst.language] = lang_counts.get(inst.language, 0) + 1
        return dict(sorted(lang_counts.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    # Example usage
    loader = DataLoader("data/raw")
    
    # Example 1: Load from simple CSV
    # instructions = loader.load_from_csv(
    #     filename="medical_data.csv",
    #     text_column="instruction_text",
    #     language_column="lang"
    # )
    
    # Example 2: Load parallel translations
    # instructions = loader.load_multilingual_parallel(
    #     filename="parallel_medical.csv",
    #     language_columns={
    #         'en': 'english',
    #         'es': 'spanish',
    #         'hi': 'hindi'
    #     }
    # )
    
    # loader.save_processed(instructions, "data/processed/instructions.jsonl")
    # print(loader.get_language_distribution(instructions))
    
    pass

       
