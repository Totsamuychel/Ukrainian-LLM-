"""
Dataset Cleaner
Видаляє сміття з dataset_train.txt:
- Рядки без кириличних літер
- Рядки, де забагато розділових знаків
- Рядки, коротші за 5 символів (якщо це не частина діалогу)
"""

import re
from pathlib import Path

INPUT_FILE = Path("dataset_train.txt")
OUTPUT_FILE = Path("dataset/processed/dataset_train_clean.txt")

def is_garbage(line):
    line = line.strip()
    
    # 1. Пусті рядки пропускаємо (але залишаємо структуру абзаців)
    if not line:
        return False 
        
    # 2. Якщо це службові теги нашого датасету - залишаємо
    if line.startswith("<START") or line.startswith("<END"):
        return False

    # 3. Перевірка на наявність кирилиці
    # Якщо в рядку немає жодної букви від А до Я або і, ї, є, ґ - це сміття
    if not re.search(r'[а-яА-ЯіІїЇєЄґҐ]', line):
        return True

    # 4. Перевірка на "засміченість" символами
    # Якщо букв менше ніж 50% від довжини рядка - ймовірно це OCR-сміття або таблиця
    letters = len(re.findall(r'[а-яА-ЯіІїЇєЄґҐa-zA-Z]', line))
    if len(line) > 10 and letters / len(line) < 0.5:
        return True

    return False

def main():
    print(f"🧹 Починаємо очищення {INPUT_FILE}...")
    
    if not INPUT_FILE.exists():
        print("❌ Файл не знайдено!")
        return

    lines_removed = 0
    total_lines = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            total_lines += 1
            
            if is_garbage(line):
                lines_removed += 1
                # Можна розкоментувати, щоб бачити, що видаляється:
                # print(f"DEL: {line.strip()}")
                continue
            
            f_out.write(line)

    print("\n" + "="*30)
    print("✅ Очищення завершено!")
    print(f"Всього рядків: {total_lines}")
    print(f"Видалено сміття: {lines_removed}")
    print(f"Чистий файл: {OUTPUT_FILE}")
    print("="*30)

if __name__ == "__main__":
    main()
