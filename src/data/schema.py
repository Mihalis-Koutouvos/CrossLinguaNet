#Data Schema
#Defines the unified schema for multilingual medical instructions.

from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class MedicalInstruction:
    """
    Unified schema for medical instructional text.
    
    Attributes:
        id: Unique identifier for this instruction
        language: ISO-like language code (e.g., 'en', 'es', 'hi', 'pa')
        text: The medical instruction text
        pair_id: Optional shared ID for parallel translations
    """

    id: str
    language: str
    text: str
    pair_id: Optional[str] = None

    def to_dict(self): 
        """Convert to dictionary representation."""

        return {
            "id": self.id,
            "language": self.language,
            "text": self.text,
            "pair_id": self.pair_id
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create instance from dictionary."""

        return cls(
            id=data['id'],
            language=data['language'],
            text=data['text'],
            pair_id=data.get('pair_id')
        )
    
    def to_json(self) -> str:
        """Conver to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str):
        """Create instance from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    def __repr__(self):
        text_preview = self.text[:50] + '...' if len(self.text) > 50 else self.text
        return f"MedicalInstruction(id={self.id}, lang={self.language}, text='{text_preview}')"

def validate_language_code(lang: str) -> bool:
    """
    Validate language code format.
    Should be 2-3 lowercase letters.
    """
    return isinstance(lang, str) and len(lang) in [2, 3] and lang.islower()

def validate_instruction(instruction: MedicalInstruction) -> tuple[bool, str]:
    """
    Validate a MedicalInstruction instance.
    
    Returns:
        (is_valid, error_message)
    """
    if not instruction.id or not isinstance(instruction.id, str):
        return False, "Invalid or missing ID"
    
    if not validate_language_code(instruction.language):
        return False, f"Invalid language code: {instruction.language}"
    
    if not instruction.text or not isinstance(instruction.text, str):
        return False, "Invalid or missing text"
    
    if len(instruction.text.strip()) == 0:
        return False, "Text cannot be empty"
    
    return True, ""