import os
from pathlib import Path
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import fitz  # PyMuPDF for PDF extraction
import re

# Dataset configuration
SOURCE_DIR = Path("dataset/raw/books")  # Path to raw downloaded literature files
OUTPUT_FILE = Path("dataset/processed/dataset_train.txt")
OUTPUT_DIR = Path("dataset/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    """Performs baseline text sanitization and normalization."""
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove standalone page numbers
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    return text.strip()

def read_txt(path: Path) -> str:
    """Reads raw text file."""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT {path.name}: {e}")
        return ""

def read_epub(path: Path) -> str:
    """Extracts text paragraphs from EPUB documents."""
    try:
        book = epub.read_epub(path)
        text = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text.append(soup.get_text())
        return "\n".join(text)
    except Exception as e:
        print(f"Error reading EPUB {path.name}: {e}")
        return ""

def read_pdf(path: Path) -> str:
    """Extracts text layer from digital PDF documents."""
    try:
        doc = fitz.open(path)
        text = []
        for page in doc:
            text.append(page.get_text())
        return "\n".join(text)
    except Exception as e:
        print(f"Error reading PDF {path.name}: {e}")
        return ""

def main():
    print(f"Starting corpus extraction from {SOURCE_DIR}...")
    
    total_text = []
    files_processed = 0
    
    if not SOURCE_DIR.exists():
        print(f"Directory {SOURCE_DIR} does not exist.")
        return
        
    for file_path in SOURCE_DIR.glob("**/*"):
        if not file_path.is_file():
            continue
            
        print(f"Processing: {file_path.name} ... ", end="")
        content = ""
        ext = file_path.suffix.lower()
        
        if ext == '.txt':
            content = read_txt(file_path)
        elif ext == '.epub':
            content = read_epub(file_path)
        elif ext == '.pdf':
            # Skip scanned PDFs (>50MB) to avoid OCR stalls
            if file_path.stat().st_size > 50 * 1024 * 1024:
                print("SKIP (large PDF, likely scanned image)")
                continue
            content = read_pdf(file_path)
        else:
            print("SKIP (unsupported format)")
            continue
            
        if len(content) > 1000:
            cleaned = clean_text(content)
            total_text.append(f"\n\n--- DOCUMENT: {file_path.stem} ---\n\n")
            total_text.append(cleaned)
            files_processed += 1
            print(f"OK ({len(cleaned):,} characters)")
        else:
            print("EMPTY (insufficient text extracted)")
            
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("".join(total_text))
        
    print(f"\nFinished! Processed {files_processed} files.")
    print(f"Combined corpus saved to: {OUTPUT_FILE}")
    print(f"Total extracted text size: {len(''.join(total_text)) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
