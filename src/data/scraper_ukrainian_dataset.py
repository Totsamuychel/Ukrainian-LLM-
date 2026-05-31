"""
Ukrainian Historical Texts Dataset Scraper (1780-1880)
Автоматичне завантаження творів для навчання LLM
"""

import requests
import os
import time
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging
from pathlib import Path
import re

# ============================================================================
# НАЛАШТУВАННЯ ЛОГУВАННЯ
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# КОНФІГ ДАТАСЕТУ
# ============================================================================

class DatasetConfig:
    """Конфіг для збору датасету"""
    
    # Період та регіон
    PERIOD_START = 1780
    PERIOD_END = 1880
    REGION = "Ukraine"
    
    # Директорії
    BASE_DIR = Path("dataset")
    RAW_DIR = BASE_DIR / "raw"
    BOOKS_DIR = RAW_DIR / "books"
    NEWSPAPERS_DIR = RAW_DIR / "newspapers"
    DOCUMENTS_DIR = RAW_DIR / "documents"
    PROCESSED_DIR = BASE_DIR / "processed"
    
    # Основні автори 18-19 ст.
    MAIN_AUTHORS = {
        "Іван Котляревський": "https://chtyvo.org.ua/authors/Ivan_Kotliarevskyi/",
        "Тарас Шевченко": "https://chtyvo.org.ua/authors/Taras_Shevchenko/",
        "Панас Мирний": "https://chtyvo.org.ua/authors/Panas_Mirnyi/",
        "Іван Франко": "https://chtyvo.org.ua/authors/Ivan_Franko/",
        "Марко Вовчок": "https://chtyvo.org.ua/authors/Marko_Vovchok/",
        "Костянтин Костомаров": "https://chtyvo.org.ua/authors/Kostiantyn_Kostomarov/",
        "Юрій Федькович": "https://chtyvo.org.ua/authors/Yurii_Fedkovych/",
        "Василь Стефаник": "https://chtyvo.org.ua/authors/Vasyl_Stefanyk/",
        "Остап Вишня": "https://chtyvo.org.ua/authors/Ostap_Vyshnia/",
        "Орест Кобилянська": "https://chtyvo.org.ua/authors/Orest_Kobylianskaya/",
    }
    
    # Налаштування для скрейпінгу
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    TIMEOUT = 10
    RETRY_COUNT = 3
    DELAY = 1  # сек між запитами (не спамити!)

# ============================================================================
# ІНІЦІАЛІЗАЦІЯ ДИРЕКТОРІЙ
# ============================================================================

def init_directories():
    """Створення необхідних директорій"""
    for directory in [
        DatasetConfig.BOOKS_DIR,
        DatasetConfig.NEWSPAPERS_DIR,
        DatasetConfig.DOCUMENTS_DIR,
        DatasetConfig.PROCESSED_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Директорія готова: {directory}")

# ============================================================================
# ОСНОВНИЙ СКРЕЙПЕР ДЛЯ CHTYVO.ORG.UA
# ============================================================================

class ChtyvoScraper:
    """Скрейпер для завантаження творів з Chtyvo"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(DatasetConfig.HEADERS)
        self.downloaded_works = []
    
    def download_text(self, url, retries=0):
        """Завантаження тексту з URL"""
        if retries > DatasetConfig.RETRY_COUNT:
            logger.error(f"✗ Не вдалось завантажити після {DatasetConfig.RETRY_COUNT} спроб: {url}")
            return None
        
        try:
            time.sleep(DatasetConfig.DELAY)  # Затримка для вежливості до сервера
            response = self.session.get(url, timeout=DatasetConfig.TIMEOUT)
            response.encoding = 'utf-8'
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.warning(f"⚠ Помилка при завантаженні {url}: {e}")
            return self.download_text(url, retries + 1)
    
    def extract_work_info(self, author_page_html):
        """Екстрагування інформації про твори зі сторінки автора"""
        soup = BeautifulSoup(author_page_html, 'html.parser')
        works = []
        
        # Пошук посилань на твори
        work_links = soup.find_all('a', {'class': 'l-work'})
        
        for link in work_links:
            work_url = link.get('href')
            work_title = link.text.strip()
            
            if work_url and work_title:
                # Перевірка, чи URL повний
                if not work_url.startswith('http'):
                    work_url = urljoin('https://chtyvo.org.ua/', work_url)
                
                works.append({
                    'title': work_title,
                    'url': work_url
                })
        
        return works
    
    def extract_text_from_work(self, work_html, title):
        """Екстрагування тексту твору зі сторінки"""
        soup = BeautifulSoup(work_html, 'html.parser')
        
        # Пошук основного тексту твору
        text_div = soup.find('div', {'class': ['l-content', 'content', 'text-content']})
        if not text_div:
            text_div = soup.find('article')
        if not text_div:
            text_div = soup.find('main')
        
        if not text_div:
            logger.warning(f"⚠ Не знайдено текст у творі: {title}")
            return None
        
        # Видалення скриптів та стилів
        for tag in text_div.find_all(['script', 'style']):
            tag.decompose()
        
        # Отримання тексту
        text = text_div.get_text(separator='\n', strip=True)
        
        # Очищення від надлишків
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Видалення порожніх рядків
        text = text.strip()
        
        return text if text else None
    
    def download_author_works(self, author_name, author_url):
        """Завантаження всіх творів автора"""
        logger.info(f"\n📚 Завантаження творів автора: {author_name}")
        
        # Завантаження сторінки автора
        author_page = self.download_text(author_url)
        if not author_page:
            return
        
        # Екстрагування інформації про твори
        works = self.extract_work_info(author_page)
        logger.info(f"   Знайдено творів: {len(works)}")
        
        # Завантаження кожного твору
        for i, work in enumerate(works, 1):
            logger.info(f"   [{i}/{len(works)}] {work['title']}")
            
            # Завантаження текст твору
            work_html = self.download_text(work['url'])
            if not work_html:
                continue
            
            # Екстрагування тексту
            text = self.extract_text_from_work(work_html, work['title'])
            if not text:
                continue
            
            # Збереження в файл
            self.save_work(author_name, work['title'], text)
    
    def save_work(self, author_name, work_title, text):
        """Збереження твору у файл"""
        # Очищення назви файлу від неприпустимих символів
        safe_filename = re.sub(r'[^\w\s-]', '', work_title)[:50]
        safe_filename = re.sub(r'[-\s]+', '_', safe_filename)
        
        filepath = DatasetConfig.BOOKS_DIR / f"{safe_filename}.txt"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Заголовок файлу
                f.write(f"# {work_title}\n")
                f.write(f"# Автор: {author_name}\n")
                f.write(f"# Джерело: https://chtyvo.org.ua/\n\n")
                f.write(text)
            
            filesize = os.path.getsize(filepath) / 1024  # KB
            logger.info(f"   ✓ Збережено: {filepath} ({filesize:.1f} KB)")
            self.downloaded_works.append({
                'author': author_name,
                'title': work_title,
                'file': str(filepath),
                'size_kb': filesize
            })
        except Exception as e:
            logger.error(f"   ✗ Помилка при збереженні: {e}")
    
    def download_all_authors(self):
        """Завантаження творів всіх авторів"""
        logger.info(f"\n{'='*60}")
        logger.info("РОЗПОЧИНАЄМО ЗАВАНТАЖЕННЯ З CHTYVO.ORG.UA")
        logger.info(f"{'='*60}")
        
        for author_name, author_url in DatasetConfig.MAIN_AUTHORS.items():
            self.download_author_works(author_name, author_url)
        
        return self.downloaded_works

# ============================================================================
# СКРЕЙПЕР ДЛЯ INTERNET ARCHIVE
# ============================================================================

class InternetArchiveScraper:
    """Скрейпер для завантаження з Internet Archive"""
    
    @staticmethod
    def search_ukrainian_texts():
        """Пошук українських текстів у Internet Archive"""
        logger.info(f"\n{'='*60}")
        logger.info("ПОШУК УКРАЇНСЬКИХ ТЕКСТІВ НА ARCHIVE.ORG")
        logger.info(f"{'='*60}")
        
        # Приклад: завантаження через API
        # Справжня URL залежить від доступних API Archive.org
        search_url = (
            "https://archive.org/advancedsearch.php?"
            "q=ukrainian+language+1800&"
            "output=json&"
            "rows=100&"
            "fl=identifier,title,date"
        )
        
        try:
            response = requests.get(search_url, timeout=10)
            data = response.json()
            
            logger.info(f"Знайдено результатів: {data['response']['numFound']}")
            
            results = data['response']['docs']
            for doc in results[:10]:  # Перші 10
                logger.info(f"  - {doc.get('title', 'N/A')} ({doc.get('date', 'N/A')})")
            
            return results
        except Exception as e:
            logger.error(f"Помилка при пошуку на Archive.org: {e}")
            return []

# ============================================================================
# ОБРОБКА ТА ОЧИЩЕННЯ ДАНИХ
# ============================================================================

class TextCleaner:
    """Очищення та нормалізація текстів"""
    
    @staticmethod
    def clean_text(text):
        """Комплексне очищення тексту"""
        # Видалення повторюючихся пробілів
        text = re.sub(r'\s+', ' ', text)
        
        # Видалення часто виникаючих OCR помилок
        text = re.sub(r'1\|', 'І', text)  # I до І
        text = re.sub(r'0', 'О', text)    # 0 до О
        
        # Нормалізація переносів рядків
        text = text.replace('-\n', '')
        text = text.replace('- ', '')
        
        # Видалення сторінок та меток (Page 123 тощо)
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\[стор\. \d+\]', '', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    @staticmethod
    def clean_all_texts(source_dir, output_dir):
        """Очищення всіх текстів у директорії"""
        logger.info(f"\n{'='*60}")
        logger.info("ОЧИЩЕННЯ ТЕКСТІВ")
        logger.info(f"{'='*60}")
        
        files = list(source_dir.glob('*.txt'))
        logger.info(f"Файлів для очищення: {len(files)}")
        
        for i, filepath in enumerate(files, 1):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                cleaned = TextCleaner.clean_text(text)
                
                output_path = output_dir / filepath.name
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned)
                
                logger.info(f"[{i}/{len(files)}] ✓ {filepath.name}")
            except Exception as e:
                logger.error(f"[{i}/{len(files)}] ✗ Помилка з {filepath.name}: {e}")

# ============================================================================
# ОБ'ЄДНАННЯ ДАТАСЕТУ
# ============================================================================

class DatasetAggregator:
    """Об'єднання всіх текстів у один корпус"""
    
    @staticmethod
    def combine_texts(source_dir, output_file):
        """Об'єднання всіх очищених текстів"""
        logger.info(f"\n{'='*60}")
        logger.info("ОБ'ЄДНАННЯ ТЕКСТІВ У КОРПУС")
        logger.info(f"{'='*60}")
        
        files = list(source_dir.glob('*.txt'))
        total_text = ""
        total_chars = 0
        
        logger.info(f"Об'єднання {len(files)} файлів...")
        
        for i, filepath in enumerate(files, 1):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    total_text += text + "\n\n"
                    total_chars += len(text)
                
                if i % 10 == 0:
                    logger.info(f"  Оброблено {i} файлів ({total_chars:,} символів)")
            except Exception as e:
                logger.error(f"Помилка з {filepath.name}: {e}")
        
        # Збереження об'єднаного корпусу
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(total_text)
        
        logger.info(f"\n✓ Корпус збережено: {output_file}")
        logger.info(f"  Загальний обсяг: {total_chars:,} символів (~{total_chars / 1_000_000:.1f} МБ)")
        logger.info(f"  Кількість файлів: {len(files)}")
        
        # Статистика
        cyrillic_count = sum(1 for c in total_text if '\u0400' <= c <= '\u04FF')
        cyrillic_percent = (cyrillic_count / len(total_text) * 100) if total_text else 0
        logger.info(f"  Кірилиця: {cyrillic_percent:.1f}%")
        
        return {
            'total_chars': total_chars,
            'total_files': len(files),
            'cyrillic_percent': cyrillic_percent,
            'output_file': str(output_file)
        }

# ============================================================================
# ГЕНЕРАЦІЯ ЗВІТУ
# ============================================================================

class ReportGenerator:
    """Генерація звіту про завантажений датасет"""
    
    @staticmethod
    def generate_report(downloaded_works, corpus_stats):
        """Генерація JSON звіту"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'settings': {
                'period_start': DatasetConfig.PERIOD_START,
                'period_end': DatasetConfig.PERIOD_END,
                'region': DatasetConfig.REGION
            },
            'downloaded_works': downloaded_works,
            'corpus_statistics': corpus_stats,
            'summary': {
                'total_works': len(downloaded_works),
                'total_authors': len(set(w['author'] for w in downloaded_works))
            }
        }
        
        # Збереження звіту
        report_path = DatasetConfig.BASE_DIR / 'dataset_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✓ Звіт збережено: {report_path}")
        
        # Вивід статистики
        logger.info(f"\n{'='*60}")
        logger.info("ПІДСУМКОВА СТАТИСТИКА")
        logger.info(f"{'='*60}")
        logger.info(f"Завантажено творів: {report['summary']['total_works']}")
        logger.info(f"Унікальних авторів: {report['summary']['total_authors']}")
        logger.info(f"Загальний обсяг: {corpus_stats['total_chars']:,} символів")
        logger.info(f"Кірилиця: {corpus_stats['cyrillic_percent']:.1f}%")

# ============================================================================
# ГОЛОВНА ФУНКЦІЯ
# ============================================================================

def main():
    """Основна функція"""
    logger.info("\n" + "="*60)
    logger.info("СКРЕЙПЕР УКРАЇНСЬКИХ ІСТОРИЧНИХ ТЕКСТІВ (18-19 ст.)")
    logger.info("="*60)
    
    # 1. Ініціалізація
    init_directories()
    
    # 2. Завантаження з Chtyvo
    scraper = ChtyvoScraper()
    downloaded_works = scraper.download_all_authors()
    
    logger.info(f"\n✓ Завантажено творів: {len(downloaded_works)}")
    
    # 3. Очищення текстів (ОПЦІОНАЛЬНО - якщо потрібна обробка)
    # TextCleaner.clean_all_texts(DatasetConfig.BOOKS_DIR, DatasetConfig.PROCESSED_DIR)
    
    # 4. Об'єднання у корпус
    corpus_output = DatasetConfig.PROCESSED_DIR / 'combined_corpus.txt'
    corpus_stats = DatasetAggregator.combine_texts(DatasetConfig.BOOKS_DIR, corpus_output)
    
    # 5. Генерація звіту
    ReportGenerator.generate_report(downloaded_works, corpus_stats)
    
    logger.info(f"\n✓ ЗАВДАННЯ ЗАВЕРШЕНО!")
    logger.info(f"  Датасет знаходиться у: {DatasetConfig.BASE_DIR}")

# ============================================================================
# ДОПОМІЖНІ ФУНКЦІЇ ДЛЯ РОЗШИРЕНОГО ПОШУКУ
# ============================================================================

def scrape_nbuv_newspapers():
    """
    Завантаження газет з НБКУ (Національна бібліотека України)
    Потребує вручну налаштування, бо НБКУ часто має доступ без API
    """
    logger.info(f"\n{'='*60}")
    logger.info("ПОШУК ГАЗЕТ НА САЙТІ НБКУ")
    logger.info(f"{'='*60}")
    
    # Приклад URL (потребує налаштування)
    # URL: https://www.nbuv.gov.ua/ → Цифрові колекції → Історичні видання
    
    logger.warning("⚠ НБКУ потребує вручної навігації через браузер.")
    logger.warning("  Завантажте газети вручну з https://www.nbuv.gov.ua/")
    logger.info("\nРекомендовані газети для завантаження:")
    newspapers = [
        "Руська видомість (1848-1849)",
        "Основа (1861-1862)",
        "Часопис (1860-1880)",
    ]
    for newspaper in newspapers:
        logger.info(f"  - {newspaper}")

def estimate_corpus_size():
    """Оцінка розміру датасету"""
    logger.info(f"\n{'='*60}")
    logger.info("ОЦІНКА РОЗМІРУ ДАТАСЕТУ")
    logger.info(f"{'='*60}")
    
    total_size = 0
    total_files = 0
    
    for filepath in DatasetConfig.BOOKS_DIR.glob('*.txt'):
        size = os.path.getsize(filepath)
        total_size += size
        total_files += 1
    
    logger.info(f"Файлів: {total_files}")
    logger.info(f"Загальний розмір: {total_size / 1_000_000:.1f} МБ")
    logger.info(f"Середній розмір файлу: {total_size / total_files / 1024:.1f} KB")

if __name__ == "__main__":
    main()
    
    # Опціональні функції
    # scrape_nbuv_newspapers()
    # estimate_corpus_size()
