"""
Ukrainian Historical Texts Dataset Scraper (Expansion Pack)
Завантаження Франка, Українки, Котляревського та інших.
"""

import os
import sys
import io
import time
import logging
import re
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

# --- Налаштування консолі (Windows Fix) ---
if sys.platform == "win32":
    try:
        os.system("chcp 65001 > nul")
    except Exception:
        pass
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# --- Логування ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("scraper_authors.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

class DatasetConfig:
    BASE_DIR = Path("dataset")
    BOOKS_DIR = BASE_DIR / "raw" / "books"
    
    # Список нових авторів для завантаження
    # Шевченко закоментований, бо вже є
    MAIN_AUTHORS = {
        "Марко Вовчок": "https://chtyvo.org.ua/authors/Markovych_Opanas_Vasylovych/",
        "Панас Мирний": "https://chtyvo.org.ua/authors/Myrnyi_Panas/",
        "Ольга Кобилянська": "https://chtyvo.org.ua/authors/Kobylianska_Olha/",
        "Михайло Коцюбинський": "https://chtyvo.org.ua/authors/Kotsiubynskyi/"
    }

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
    }
    TIMEOUT = 20
    DELAY = 1

def init_directories():
    DatasetConfig.BOOKS_DIR.mkdir(parents=True, exist_ok=True)

class ChtyvoScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(DatasetConfig.HEADERS)

    def download_text(self, url):
        try:
            time.sleep(DatasetConfig.DELAY)
            resp = self.session.get(url, timeout=DatasetConfig.TIMEOUT)
            resp.encoding = "utf-8"
            if resp.status_code == 200:
                return resp.text
        except Exception as e:
            logger.warning(f"Помилка завантаження сторінки {url}: {e}")
        return None

    def download_file(self, url, dest_path):
        try:
            time.sleep(DatasetConfig.DELAY)
            with self.session.get(url, timeout=DatasetConfig.TIMEOUT, stream=True) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
            return True
        except Exception as e:
            logger.warning(f"Помилка завантаження файлу {url}: {e}")
            return False

    def extract_works(self, author_html, author_url):
        soup = BeautifulSoup(author_html, "html.parser")
        works = []
        seen = set()
        
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            text = a.get_text(strip=True)
            
            # Фільтрація сміття
            if not text or len(text) < 2: continue
            if any(x in href for x in ["facebook", "twitter", "rss", "authors//"]): continue
            
            # Шукаємо посилання, які ведуть "всередину" розділу автора
            full_url = urljoin(author_url, href)
            
            # Перевірка, що це дійсно твір цього автора (починається з URL автора)
            # Але обережно: іноді посилання виглядають як /authors/Franko/Tvory/...
            if full_url.startswith(author_url) and full_url != author_url:
                if full_url not in seen:
                    works.append({"title": text, "url": full_url})
                    seen.add(full_url)
        return works

    def extract_downloads(self, work_html):
        soup = BeautifulSoup(work_html, "html.parser")
        links = []
        exts = (".txt", ".epub", ".fb2", ".doc", ".docx",)
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if any(href.lower().endswith(ext) for ext in exts) or "download" in href:
                full_url = urljoin("https://chtyvo.org.ua", href)
                if full_url not in links:
                    links.append(full_url)
        return links

    def run(self):
        logger.info("START - Завантаження додаткових авторів")
        
        for author, url in DatasetConfig.MAIN_AUTHORS.items():
            logger.info(f"\n======== {author} ========")
            html = self.download_text(url)
            if not html: continue
            
            works = self.extract_works(html, url)
            logger.info(f"Знайдено творів: {len(works)}")
            
            for i, work in enumerate(works, 1):
                title = work['title']
                # Пропускаємо статті про автора
                if "Про автора" in title or "Стаття" in title or "Біографія" in title:
                    continue

                logger.info(f"  [{i}/{len(works)}] {title[:40]}...")
                
                work_html = self.download_text(work['url'])
                if not work_html: continue
                
                dl_links = self.extract_downloads(work_html)
                if not dl_links: continue

                # Пріоритет: TXT > EPUB > FB2 > PDF (щоб менше сміття)
                sorted_links = sorted(dl_links, key=lambda x: (
                    0 if ".txt" in x else 
                    1 if ".epub" in x else 
                    2 if ".fb2" in x else 
                    3
                ))
                
                # Качаємо найкращий доступний формат
                best_link = sorted_links[0]
                
                # Визначаємо розширення
                ext = ".bin"
                for e in [".txt", ".epub", ".fb2", ".doc"]:
                    if e in best_link.lower(): ext = e; break
                
                # Формуємо ім'я: Author_Title.ext
                safe_author = re.sub(r"[^\w]", "", author)[:10]
                safe_title = re.sub(r"[^\w\s-]", "", title)[:40].strip().replace(" ", "_")
                filename = f"{safe_author}_{safe_title}{ext}"
                dest = DatasetConfig.BOOKS_DIR / filename
                
                if not dest.exists():
                    if self.download_file(best_link, dest):
                        logger.info(f"    -> OK: {filename}")
                else:
                    logger.info(f"    -> SKIP (існує)")

if __name__ == "__main__":
    init_directories()
    scraper = ChtyvoScraper()
    scraper.run()
