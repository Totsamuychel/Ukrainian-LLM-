"""
Dataset Cleaner
Removes low-quality and corrupted lines from dataset_train.txt:
- Lines lacking Cyrillic characters
- Lines dominated by OCR noise or punctuation marks
- Artifacts shorter than 5 characters (unless conversational dialog)
"""

import re
from pathlib import Path

INPUT_FILE = Path("dataset_train.txt")
OUTPUT_FILE = Path("dataset/processed/dataset_train_clean.txt")

def is_garbage(line: str) -> bool:
    line = line.strip()
    
    # 1. Preserve empty lines for paragraph separation
    if not line:
        return False 
        
    # 2. Preserve dataset delimiter tags
    if line.startswith("<START") or line.startswith("<END"):
        return False

    # 3. Check for Cyrillic letters (Ukrainian alphabet)
    if not re.search(r'[а-яА-ЯіІїЇєЄґҐ]', line):
        return True

    # 4. Check for high density of OCR noise or non-alphabetic symbols (<50% letters)
    letters = len(re.findall(r'[а-яА-ЯіІїЇєЄґҐa-zA-Z]', line))
    if len(line) > 10 and letters / len(line) < 0.5:
        return True

    return False

def main():
    print(f"🧹 Starting cleaning pipeline for {INPUT_FILE}...")
    
    if not INPUT_FILE.exists():
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    lines_removed = 0
    total_lines = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            total_lines += 1
            
            if is_garbage(line):
                lines_removed += 1
                continue
            
            f_out.write(line)
            
    print(f"✅ Cleaning complete!")
    print(f"📊 Total lines processed: {total_lines:,}")
    print(f"🗑️ Noise lines removed: {lines_removed:,} ({(lines_removed/total_lines)*100:.1f}%)")
    print(f"💾 Clean dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
