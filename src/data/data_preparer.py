"""
Dataset Preparer & Converter
1. Читає всі файли з dataset/raw/books
2. Конвертує PDF та EPUB у текст
3. Очищує текст від сміття
4. Зберігає результат у dataset/processed/dataset_train.txt
"""

import os
import re
from pathlib import Path
import fitz  # PyMuPDF
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from tqdm import tqdm

# --- НАЛАШТУВАННЯ ---
BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw" / "books"
PROCESSED_DIR = BASE_DIR / "processed"
OUTPUT_FILE = PROCESSED_DIR / "dataset_train.txt"

# Створюємо папку для результату
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    """
    Очищення тексту від артефактів верстки.
    """
    if not text:
        return ""
    
    # 1. Заміна нерозривних пробілів та інших дивних символів
    text = text.replace('\xa0', ' ').replace('\x0c', '')
    
    # 2. Видалення номерів сторінок (одинокі цифри на рядку)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # 3. Видалення колонтитулів (надто короткі рядки з великих літер)
    # Наприклад: "ТАРАС ШЕВЧЕНКО" зверху кожної сторінки
    # text = re.sub(r'^[А-ЯІЇЄҐ\s]{5,}$', '', text, flags=re.MULTILINE)

    # 4. Нормалізація пробілів (перетворення багаторазових пробілів в один)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 5. Об'єднання розірваних рядків (переносів слів)
    # Якщо рядок закінчується на дефіс, пробуємо склеїти (це ризиковано для віршів, тому обережно)
    # Для безпеки просто лишаємо як є або видаляємо зайві порожні рядки
    text = re.sub(r'\n{3,}', '\n\n', text) # Максимум 2 пустих рядка підряд
    
    return text.strip()

def read_pdf(path):
    """Витягує текст з PDF. Працює тільки якщо там є текстовий шар."""
    try:
        doc = fitz.open(path)
        text = []
        for page in doc:
            t = page.get_text()
            if t:
                text.append(t)
        return "\n".join(text)
    except Exception as e:
        print(f"❌ Помилка PDF {path.name}: {e}")
        return ""

def read_epub(path):
    """Витягує текст з EPUB."""
    try:
        book = epub.read_epub(path)
        chapters = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                chapters.append(soup.get_text())
        return "\n".join(chapters)
    except Exception as e:
        print(f"❌ Помилка EPUB {path.name}: {e}")
        return ""

def read_txt(path):
    """Читає звичайний TXT файл."""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Помилка TXT {path.name}: {e}")
        return ""

def process_files():
    print(f"📂 Сканування папки: {RAW_DIR}")
    
    files = [f for f in RAW_DIR.iterdir() if f.is_file()]
    if not files:
        print("⚠️ Папка пуста! Спочатку запустіть scraper.")
        return

    print(f"знайдено {len(files)} файлів. Починаємо обробку...")
    
    total_chars = 0
    processed_count = 0
    skipped_count = 0
    
    # Відкриваємо файл для запису (режим 'w' перезапише старий файл)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        
        # tqdm показує гарну смужку прогресу
        for file_path in tqdm(files, unit="file"):
            ext = file_path.suffix.lower()
            content = ""
            
            # Вибір методу читання
            if ext == '.txt':
                content = read_txt(file_path)
            elif ext == '.epub':
                content = read_epub(file_path)
            elif ext == '.pdf':
                # Пропускаємо PDF, які занадто великі (ймовірно скани > 50MB)
                if file_path.stat().st_size > 50 * 1024 * 1024:
                    skipped_count += 1
                    continue
                content = read_pdf(file_path)
            elif ext in ['.djvu', '.zip', '.rar']:
                # DJVU пропускаємо свідомо
                skipped_count += 1
                continue
            else:
                skipped_count += 1
                continue
            
            # Якщо вдалося витягти текст
            if content and len(content) > 500: # Мінімум 500 символів, щоб не писати сміття
                cleaned = clean_text(content)
                
                # Записуємо у файл з розділювачем
                out_f.write(f"\n\n<START_TEXT_FILE id=\"{file_path.name}\">\n")
                out_f.write(cleaned)
                out_f.write(f"\n<END_TEXT_FILE>\n")
                
                total_chars += len(cleaned)
                processed_count += 1
            else:
                skipped_count += 1

    print("\n" + "="*40)
    print(f"✅ ГОТОВО!")
    print(f"📄 Оброблено файлів: {processed_count}")
    print(f"⏩ Пропущено файлів: {skipped_count}")
    print(f"📊 Загальний розмір датасету: {total_chars / 1024 / 1024:.2f} MB")
    print(f"💾 Збережено в: {OUTPUT_FILE}")
    print("="*40)

if __name__ == "__main__":
    process_files()
