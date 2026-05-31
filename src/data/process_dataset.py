import os
from pathlib import Path
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import fitz  # PyMuPDF для PDF
import re

# Налаштування
SOURCE_DIR = Path("dataset/raw/books")  # Де лежать скачані файли
OUTPUT_FILE = Path("dataset/processed/dataset_train.txt")
OUTPUT_DIR = Path("dataset/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    """Базове очищення тексту"""
    if not text: return ""
    # Видаляємо зайві пробіли
    text = re.sub(r'\s+', ' ', text)
    # Видаляємо номери сторінок (приблизно)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    return text.strip()

def read_txt(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Помилка читання TXT {path.name}: {e}")
        return ""

def read_epub(path):
    try:
        book = epub.read_epub(path)
        text = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text.append(soup.get_text())
        return "\n".join(text)
    except Exception as e:
        print(f"Помилка читання EPUB {path.name}: {e}")
        return ""

def read_pdf(path):
    """Спроба витягти текст з PDF (якщо є текстовий шар)"""
    try:
        doc = fitz.open(path)
        text = []
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text)
    except Exception as e:
        print(f"Помилка читання PDF {path.name}: {e}")
        return ""

def main():
    print(f"Починаємо обробку файлів з {SOURCE_DIR}...")
    
    total_text = ""
    files_processed = 0
    
    # Проходимо по всіх файлах
    for file_path in SOURCE_DIR.iterdir():
        if not file_path.is_file(): continue
        
        content = ""
        ext = file_path.suffix.lower()
        
        print(f"Обробка: {file_path.name} ... ", end="")
        
        if ext == '.txt':
            content = read_txt(file_path)
        elif ext == '.epub':
            content = read_epub(file_path)
        elif ext == '.pdf':
            # PDF обробляємо вибірково, бо вони великі
            # Якщо файл > 50MB, це скоріше за все скан (картинки), пропускаємо для швидкості
            if file_path.stat().st_size > 50 * 1024 * 1024:
                print("SKIP (завеликий PDF, ймовірно скан)")
                continue
            content = read_pdf(file_path)
        else:
            print("SKIP (непідтримуваний формат)")
            continue
            
        # Якщо текст витягнувся
        if len(content) > 1000: # Хоча б 1000 символів
            cleaned = clean_text(content)
            total_text += cleaned + "\n\n" + "="*20 + "\n\n"
            print(f"OK ({len(cleaned)} символів)")
            files_processed += 1
        else:
            print("EMPTY (текст не знайдено або файл пустий)")

    # Збереження
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(total_text)
        
    print(f"\nГотово! Оброблено файлів: {files_processed}")
    print(f"Результат збережено в: {OUTPUT_FILE}")
    print(f"Загальний розмір чистого тексту: {len(total_text)/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()
