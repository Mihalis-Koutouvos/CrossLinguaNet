"""
Create synthetic medical instructions in EN/ES/RU.
No downloads needed - generates realistic medical text.
"""

import csv
import random
from pathlib import Path

# Medical instruction templates
TEMPLATES = {
    'en': [
        "Take {dose} {medication} {frequency} {timing}.",
        "Apply {medication} to the affected area {frequency}.",
        "Avoid {food} while taking this medication.",
        "Consult your doctor if {symptom} persists.",
        "Store {medication} in a cool, dry place.",
        "Do not exceed {dose} per day.",
        "Take with {food} to reduce stomach upset.",
        "Discontinue use if {symptom} occurs.",
        "Swallow {medication} whole, do not crush or chew.",
        "Take {dose} before bedtime.",
        "Mix {medication} with water before taking.",
        "Keep out of reach of children.",
        "Use exactly as prescribed by your doctor.",
        "May cause {symptom}. Contact doctor if severe.",
        "Take on an empty stomach, {timing} before meals.",
    ],
    'es': [
        "Tome {dose} {medication} {frequency} {timing}.",
        "Aplique {medication} en el área afectada {frequency}.",
        "Evite {food} mientras toma este medicamento.",
        "Consulte a su médico si {symptom} persiste.",
        "Guarde {medication} en un lugar fresco y seco.",
        "No exceda {dose} por día.",
        "Tome con {food} para reducir el malestar estomacal.",
        "Suspenda el uso si ocurre {symptom}.",
        "Trague {medication} entero, no triture ni mastique.",
        "Tome {dose} antes de acostarse.",
        "Mezcle {medication} con agua antes de tomar.",
        "Mantenga fuera del alcance de los niños.",
        "Use exactamente como lo prescribió su médico.",
        "Puede causar {symptom}. Contacte al médico si es grave.",
        "Tome con el estómago vacío, {timing} antes de las comidas.",
    ],
    'ru': [
        "Принимайте {dose} {medication} {frequency} {timing}.",
        "Нанесите {medication} на пораженную область {frequency}.",
        "Избегайте {food} при приеме этого лекарства.",
        "Проконсультируйтесь с врачом, если {symptom} сохраняется.",
        "Храните {medication} в прохладном, сухом месте.",
        "Не превышайте {dose} в день.",
        "Принимайте с {food}, чтобы уменьшить расстройство желудка.",
        "Прекратите использование, если возникает {symptom}.",
        "Глотайте {medication} целиком, не измельчайте и не разжевывайте.",
        "Принимайте {dose} перед сном.",
        "Смешайте {medication} с водой перед приемом.",
        "Храните в недоступном для детей месте.",
        "Используйте точно так, как предписано врачом.",
        "Может вызвать {symptom}. Обратитесь к врачу, если серьезно.",
        "Принимайте натощак, {timing} перед едой.",
    ]
}

# Vocabulary for filling templates
VOCABULARY = {
    'dose': {
        'en': ['one tablet', 'two tablets', 'one capsule', '5 ml', '10 mg', 'one dose'],
        'es': ['una tableta', 'dos tabletas', 'una cápsula', '5 ml', '10 mg', 'una dosis'],
        'ru': ['одну таблетку', 'две таблетки', 'одну капсулу', '5 мл', '10 мг', 'одну дозу']
    },
    'medication': {
        'en': ['this medication', 'the medicine', 'the prescription', 'the antibiotic', 'the cream'],
        'es': ['este medicamento', 'la medicina', 'la receta', 'el antibiótico', 'la crema'],
        'ru': ['это лекарство', 'лекарство', 'рецепт', 'антибиотик', 'крем']
    },
    'frequency': {
        'en': ['daily', 'twice daily', 'three times daily', 'every 6 hours', 'as needed'],
        'es': ['diariamente', 'dos veces al día', 'tres veces al día', 'cada 6 horas', 'según sea necesario'],
        'ru': ['ежедневно', 'два раза в день', 'три раза в день', 'каждые 6 часов', 'по мере необходимости']
    },
    'timing': {
        'en': ['with meals', 'after meals', 'before meals', 'at bedtime', 'in the morning'],
        'es': ['con las comidas', 'después de las comidas', 'antes de las comidas', 'antes de acostarse', 'por la mañana'],
        'ru': ['во время еды', 'после еды', 'перед едой', 'перед сном', 'утром']
    },
    'food': {
        'en': ['dairy products', 'alcohol', 'grapefruit', 'fatty foods', 'caffeine'],
        'es': ['productos lácteos', 'alcohol', 'toronja', 'alimentos grasos', 'cafeína'],
        'ru': ['молочные продукты', 'алкоголь', 'грейпфрут', 'жирную пищу', 'кофеин']
    },
    'symptom': {
        'en': ['dizziness', 'nausea', 'rash', 'swelling', 'difficulty breathing', 'pain'],
        'es': ['mareo', 'náuseas', 'sarpullido', 'hinchazón', 'dificultad para respirar', 'dolor'],
        'ru': ['головокружение', 'тошнота', 'сыпь', 'отек', 'затрудненное дыхание', 'боль']
    }
}

def generate_instruction(lang, template_idx):
    """Generate one instruction by filling in a template."""
    template = TEMPLATES[lang][template_idx]
    
    # Find all placeholders in template
    import re
    placeholders = re.findall(r'\{(\w+)\}', template)
    
    # Fill each placeholder
    filled = template
    for placeholder in placeholders:
        if placeholder in VOCABULARY:
            value = random.choice(VOCABULARY[placeholder][lang])
            filled = filled.replace(f'{{{placeholder}}}', value, 1)
    
    return filled

def create_parallel_set(template_idx, pair_id):
    """Create EN/ES/RU translations of the same instruction."""
    return {
        'en': generate_instruction('en', template_idx),
        'es': generate_instruction('es', template_idx),
        'ru': generate_instruction('ru', template_idx)
    }

def create_synthetic_dataset(num_pairs=10000, output_file="data/raw/medical_en_es_ru.csv"):
    """
    Create synthetic medical instruction dataset.
    
    Args:
        num_pairs: Number of instruction sets to generate
        output_file: Where to save the CSV
    """
    print("="*60)
    print("Creating Synthetic Medical Dataset")
    print("="*60)
    print(f"\nGenerating {num_pairs} parallel instruction sets...")
    print("Languages: EN, ES, RU")
    print()
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'language', 'text', 'pair_id', 'medical_score'])
        
        for pair_num in range(num_pairs):
            if pair_num % 1000 == 0 and pair_num > 0:
                print(f"  Generated {pair_num:,} pairs...")
            
            # Pick a random template
            template_idx = random.randint(0, len(TEMPLATES['en']) - 1)
            
            # Generate parallel instructions
            instructions = create_parallel_set(template_idx, f"pair_{pair_num}")
            
            # Write EN-ES pair
            pair_id = f"pair_{pair_num}"
            writer.writerow([
                f"{pair_id}_en", 'en', instructions['en'], pair_id, 0.5
            ])
            writer.writerow([
                f"{pair_id}_es", 'es', instructions['es'], pair_id, 0.5
            ])
            
            # For some pairs, add Russian too (to create trilingual sets)
            if pair_num < num_pairs // 2:
                writer.writerow([
                    f"{pair_id}_ru", 'ru', instructions['ru'], pair_id, 0.5
                ])
    
    # Calculate statistics
    total_rows = num_pairs * 2 + (num_pairs // 2)
    
    print(f"\n✓ Created {output_path}")
    print(f"  Total instructions: {total_rows:,}")
    print(f"  EN-ES pairs: {num_pairs:,}")
    print(f"  EN-RU pairs: {num_pairs // 2:,}")
    print()
    
    # Show samples
    print("Sample instructions:")
    print("-" * 60)
    sample = create_parallel_set(0, "sample")
    print(f"EN: {sample['en']}")
    print(f"ES: {sample['es']}")
    print(f"RU: {sample['ru']}")
    print()
    
    print("="*60)
    print("SUCCESS! Dataset created.")
    print("="*60)
    print("\nNext step:")
    print("  cd src/data")
    print("  python pipeline.py")

if __name__ == "__main__":
    # Generate 10K instruction sets = ~25K total rows
    create_synthetic_dataset(num_pairs=10000)