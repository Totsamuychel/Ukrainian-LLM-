# Повний посібник: Створення історичної української LLM моделі (18-19 столітття)

## Частина 1: Пошук та збір датасету

### 1.1 Найкращі джерела даних

#### 1.1.1 Відкриті цифрові бібліотеки (FREE)

**Важливо:** Ці сайти містять українські тексти у форматі .txt, .pdf, .epub, які вільно можна використовувати для навчання ML моделей.

##### 1. Chtyvo.org.ua (ПЕРШОРЯДНЕ ДЖЕРЕЛО)
- **URL:** https://chtyvo.org.ua/
- **Що це:** Найбільша українська цифрова бібліотека
- **Доступні твори 18-19 ст.:**
  - Іван Котляревський — «Енеїда» (1798-1842)
  - Тарас Шевченко — поетичні твори, «Журнал» (видавлюється 1840-х)
  - Панас Мирний — романи
  - Іван Франко — твори (кінець 19 ст.)
  - Марко Вовчок — оповідання
  - Костянтин Гедеонський — різні твори

**Як звідти завантажити:**
1. Перейдіть на https://chtyvo.org.ua/
2. Пошук автора (наприклад, "Котляревський")
3. Натисніть на твір
4. Виберіть формат завантаження: TXT, EPUB, PDF (внизу сторінки є кнопка завантаження)
5. Розпакуйте та перейменуйте у формат `.txt` (UTF-8 encoding)

**Рекомендовані твори для 18-19 ст.:**
```
Авторів виписано: ~50-100 авторів периоду 1750-1900
Приблизний обсяг: 1-3 ГБ текстових даних доступно
```

##### 2. Diasporiana.org.ua (УКРАЇНСЬКА ДІАСПОРА)
- **URL:** https://diasporiana.org.ua/
- **Що це:** Архів творів українських авторів, включно з історичними документами
- **Доступні матеріали:**
  - Листування видатних осіб
  - Історичні записки
  - Мемуари (19 ст.)
  - Богословські трактати (18 ст.)

**Як використовувати:**
- Пошук по авторам та датам
- Завантаження у форматі PDF (потім потрібна OCR)

##### 3. Internet Archive (archive.org) — Ukrainian Section
- **URL:** https://archive.org/ (пошук "Ukrainian 1800" або "Українська мова 18 століття")
- **Що знайти:**
  - Старі українські газети (сканізовані)
  - Книги з періоду 1800-1900
  - Навіть деякі середньовічні документи

**Порада:** Пошук фільтри:
```
Search: "Ukrainian" AND "1800-1850"
Формат: texts (скановані книги з OCR)
```

##### 4. Google Books (books.google.com)
- **URL:** https://books.google.com/
- **Як користуватися для дослідження:**
  1. Пошук українських книг 18-19 ст.
  2. Якщо книга у публічному домені (1923 р. видання або раніше), можна завантажити PDF
  3. Потім конвертувати у TXT через OCR

**Важливо:** Лише книги до ~1923 року (публічний домен у більшості країн)

##### 5. Project Gutenberg (gutenberg.org)
- **URL:** https://www.gutenberg.org/
- **Українські матеріали:**
  - Меньше, ніж англійські, але є
  - Якість обробки висока (чистий текст)
  - Пошук по мові: "Ukrainian"

**Пошук:**
```
https://www.gutenberg.org/ebooks/search/?query=Ukrainian
```

---

#### 1.1.2 Українські газети та журнали (18-19 ст.)

**Важливо:** Газети дають найбільший обсяг автентичного "живого" мовлення того часу.

##### Газети 19 століття (доступні в архівах):

1. **«Руська видомість»** (1848-1849) — перша український газета
   - Архіви: Львівська національна наукова бібліотека України
   - Пошук: https://www.lnnbu.org.ua/ (Цифровий архів)

2. **«Метрополь» / «Часопис»** (1860-1880)
   - Громадсько-літературний журнал
   - НБКУ archiv.org

3. **«Основа»** (1861-1862) — видавав Костомаров
   - Культурний журнал украї
   - OCR-ізовані версії у цифрових архівах

4. **«Справочник для русских в Западной России»** (1860-1870)
   - Вживалась українська мова у російському державному контексті
   - Дивіться у архівах Львова

**Де знайти газети:**
- **Українська національна бібліотека імені В.І. Вернадського** (НБКУ): https://www.nbuv.gov.ua/
  - Розділ: "Цифрові колекції" → "Історичні видання"
  - Багато газет оцифровано (але деякі мають DRM)

- **Львівська національна наукова бібліотека** (ЛННБУ): https://www.lnnbu.org.ua/
  - Російський império часов, в яких писалась українська

- **Молода Україна** (молодіжний журнал): https://library.pnu.edu.ua/
  - Мулебарберін архіви Прикарпатського університету

---

#### 1.1.3 Історичні документи

**Укази, накази, листування того часу:**

1. **Кирилівська колекція** (листи, накази, розпорядження) — архіви Українського історичного інституту
2. **Козацькі договори та грамоти** 17-18 ст. — Інститут історії України НАНУ
3. **Духовні заповіти, юридичні документи** — місцеві архіви міст

**Як отримати доступ:**
```
Пошук: "Ukrainian historical documents 1800-1850"
Архіви: РГАДА (російський архів), ЦДАВО (український архів)
Деякі оцифровані, деякі потребують запиту
```

---

### 1.2 Пошаговий алгоритм збору датасету

#### Етап 1: Визначення часової шкали та регіону

```
Період: 1780-1880 (повна траєкторія усть сяогу як літератури)
Географія: Україна (особливо Західна Україна — Галичина, де більше писаних документів)
Мова: Українська (можуть бути деякі російські тексти того часу з українськими елементами)
Жанри: Художня література, розповіді, газети, листування, наукові трактати
```

#### Етап 2: Систематичний пошук по авторах

**Обов'язкові автори 18-19 ст.:**

| Період | Автор | Основні твори | Обсяг тексту |
|--------|-------|---------------|--------------|
| 1798-1842 | Іван Котляревський | Енеїда, комедії | ~500 KB |
| 1814-1861 | Тарас Шевченко | Поезії, прозаїчні твори | ~5 MB |
| 1823-1895 | Панас Мирний | Романи, оповідання | ~2 MB |
| 1825-1900 | Іван Франко | Поезія, проза, драма | ~8 MB |
| 1833-1900 | Марко Вовчок | Оповідання | ~1 MB |
| 1807-1886 | Костянтин Костомаров | Історичні праці | ~3 MB |
| 1840-1893 | Іванов | Духовні пісні | ~300 KB |
| Анонім | Українські народні пісні | Збірки | ~1 MB |

**ПЛАН ДІЙ:**
1. Для кожного автора перейти на https://chtyvo.org.ua/
2. Пошук по вікіпедії структури авторства: https://uk.wikipedia.org/wiki/[Автор]
3. З вікіпедії взяти список основних творів
4. Завантажити кожен твір з Chtyvo у форматі TXT
5. Переконатися, що файли в UTF-8 (не ANSI/CP1251)

#### Етап 3: Збір газет та журналів

**Порядок роботи:**

```bash
# Структура папок для збереження:
dataset/
├── books/
│   ├── kotliarevskyi/
│   ├── shevchenko/
│   ├── franko/
│   └── ...
├── newspapers/
│   ├── 1848_ruska_vidomist/
│   ├── 1860_osnova/
│   └── ...
└── documents/
    ├── letters/
    ├── decrees/
    └── ...
```

**Джерела газет (з OCR):**
- https://www.nbuv.gov.ua/ → Пошук → Фільтр по датам (1800-1880)
- Завантажити PDF → Конвертувати у TXT через OCR

#### Етап 4: OCR обробка (для сканованих документів)

**Якщо у вас є PDF старих газет/книг:**

Інструмент: **Tesseract OCR** (безкоштовно) або **EasyOCR** (Python)

```bash
# Встановлення Tesseract на Windows/Linux/Mac:
# https://github.com/UB-Mannheim/tesseract/wiki

# Конвертація PDF → TXT з українською:
tesseract input.pdf output -l ukr
```

**Альтернатива (простіша) — Python:**

```python
from paddleocr import PaddleOCR
from pdf2image import convert_from_path

# 1. Конвертуйте PDF у картинки
images = convert_from_path('gazeta.pdf')

# 2. Розпізнайте текст з українською
ocr = PaddleOCR(use_angle_cls=True, lang='uk')
texts = []
for img in images:
    result = ocr.ocr(np.array(img), cls=True)
    text = '\n'.join([line[0][1] for line in result[0]])
    texts.append(text)

# 3. Збережіть у файл
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(texts))
```

---

### 1.3 Очищення та підготовка даних

#### Крок 1: Видалення артефактів

```python
import re
import os

def clean_text(text):
    """
    Видалення:
    - Розділових знаків з модерниму мовою
    - OCR помилок
    - Заголовків (якщо не потрібні)
    """
    
    # Видалення повторюючихся пробілів
    text = re.sub(r'\s+', ' ', text)
    
    # Видалення сторінок типу "Page 123"
    text = re.sub(r'Page \d+', '', text)
    
    # Видалення сучасної пунктуації (якщо потрібно зберегти тільки період)
    # text = re.sub(r'[^\wа-яА-ЯіЇєЄґҐ\s\.\,\!]', '', text, flags=re.UNICODE)
    
    # Видалення дублів слів OCR
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)
    
    # Нормалізація переносів з попередня рядка
    text = text.replace('-\n', '')
    text = text.replace('- ', '')
    
    return text

# Обробка всіх файлів
for file in os.listdir('dataset/'):
    if file.endswith('.txt'):
        with open(f'dataset/{file}', 'r', encoding='utf-8') as f:
            text = f.read()
        
        cleaned = clean_text(text)
        
        with open(f'dataset_cleaned/{file}', 'w', encoding='utf-8') as f:
            f.write(cleaned)
```

#### Крок 2: Об'єднання всіх текстів

```python
import os

output_text = ""

for file in os.listdir('dataset_cleaned/'):
    if file.endswith('.txt'):
        with open(f'dataset_cleaned/{file}', 'r', encoding='utf-8') as f:
            content = f.read()
            output_text += content + "\n\n"

# Запис у один великий файл
with open('combined_corpus.txt', 'w', encoding='utf-8') as f:
    f.write(output_text)

print(f"Загальний обсяг: {len(output_text)} символів")
print(f"Приблизно слів: {len(output_text.split())}")
```

#### Крок 3: Перевірка якості

```python
# Перевірка кодування та структури
with open('combined_corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
    # Перевірка українських символів
    cyrillic_count = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    total_count = len(text)
    
    print(f"Кирилиця: {cyrillic_count / total_count * 100:.2f}%")
    print(f"Перші 500 символів:\n{text[:500]}")
    
    # Якщо менше 70% кирилиці - щось не так з кодуванням
```

---

## Частина 2: Адаптація кодю під кирилицю та тренування

### 2.1 Встановлення та налаштування

#### Крок 1: Клонування репозиторію та встановлення залежностей

```bash
# Клонування репозиторію
git clone https://github.com/haykgrigo3/TimeCapsuleLLM.git
cd TimeCapsuleLLM

# Встановлення залежностей
pip install -r requirements.txt
# Якщо requirements.txt відсутній:
pip install torch numpy tokenizers tqdm

# Для OCR (якщо потрібна обробка газет)
pip install paddleocr pdf2image pillow pytesseract
```

#### Крок 2: Структура проєкту (як мав бути)

```
TimeCapsuleLLM/
├── data/
│   ├── raw/                    # Сирі текстові файли
│   │   ├── books/
│   │   ├── newspapers/
│   │   └── documents/
│   ├── processed/              # Очищені файли
│   │   └── combined_corpus.txt
│   └── vocab.json              # Токенізатор (создается на етапі 2.3)
├── tokenizer/
│   ├── train_tokenizer.py      # ПОТРІБНО АДАПТУВАТИ ДЛЯ КИРИЛИЦІ
│   └── tokenizer.model
├── model/
│   ├── nanoGPT.py              # Архітектура трансформера (модифицируем)
│   ├── train.py                # Скрипт навчання (модифицируем)
│   └── generate.py             # Генерація тексту
└── README.md
```

---

### 2.2 АДАПТАЦІЯ ТОКЕНІЗАТОРА ДЛЯ УКРАЇНСЬКОЇ КИРИЛИЦІ

**Це найважливіша частина.** Оригінальний токенізатор у `TimeCapsuleLLM` налаштований на англійську. Потрібна спеціальна обробка для:
1. Українських букв (а, б, в, г, ґ, д, е, є, ж, з, и, і, й, к, л, м, н, о, п, р, с, т, у, ф, х, ц, ч, ш, щ, ь, ю, я)
2. Старих діакритик (если потрібні)
3. Правильного розбиття слів

#### Модифікований скрипт для тренування токенізатора:

**Файл: `tokenizer/train_tokenizer_ukrainian.py`**

```python
"""
Тренування BPE токенізатора для української мови історичного периоду
Заснований на tokenizers (HuggingFace)
"""

import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, NFD, StripAccents, Lowercase
from tokenizers.pre_tokenizers import Whitespace, Sequence as PreTokenSequence
from tokenizers.processors import ByteLevel, Sequence as ProcessorSequence
from tokenizers.trainers import BpeTrainer
import json

# Шлях до датасету
CORPUS_FILE = "../data/processed/combined_corpus.txt"
OUTPUT_DIR = "./"

def train_ukrainian_tokenizer(corpus_path, output_dir="./", vocab_size=10000):
    """
    Тренування BPE токенізатора на українських текстах
    
    Args:
        corpus_path: Шлях до файлу combined_corpus.txt
        output_dir: Директорія для збереження модель
        vocab_size: Розмір словника (для української 10K-50K достатньо)
    """
    
    # 1. Ініціалізація токенізатора з моделлю BPE
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    # 2. Нормалізація (ВАЖЛИВО: НЕ используем Lowercase для історичної мови!)
    # Сучасні LLM часто робляють lowercase, але ми зберігаємо оригінальну кірилицю
    tokenizer.normalizer = Sequence([
        NFD(),           # Розкладання комбінованих символів
        # StripAccents(),  # Видалення наголосів (опціонально)
        # Lowercase()      # ВІДКЛЮЧАЄМО - зберігаємо історичну орфографію
    ])
    
    # 3. Pre-tokenizer (розбиття на базові одиниці)
    # Для кірилиці потрібна спеціальна обробка
    tokenizer.pre_tokenizer = PreTokenSequence([
        Whitespace(),    # Розбиття по пробілах
    ])
    
    # 4. Post-processor (остаточна обробка)
    tokenizer.post_processor = ByteLevel()
    
    # 5. Тренер з параметрами для історичної мови
    trainer = BpeTrainer(
        vocab_size=vocab_size,           # Розмір словника (10000-50000)
        min_frequency=2,                 # Мінімальна частота для токену
        show_progress=True,
        special_tokens=[
            "<pad>",     # Padding
            "<unk>",     # Unknown token
            "<bos>",     # Beginning of sequence
            "<eos>",     # End of sequence
            "<cls>",     # Classifier token
        ]
    )
    
    # 6. Тренування на корпусі
    print(f"Тренування токенізатора на {corpus_path}...")
    tokenizer.train([corpus_path], trainer)
    
    # 7. Збереження
    tokenizer.save(f"{output_dir}/tokenizer_ukrainian.json")
    
    # Збереження окремо vocab для nanoGPT
    vocab = tokenizer.get_vocab()
    with open(f"{output_dir}/vocab_ukrainian.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Токенізатор збережено в {output_dir}")
    print(f"✓ Розмір словника: {len(vocab)}")
    
    # Тестування
    test_text = "Така неправда! Чорні люди сиділи далі."
    tokens = tokenizer.encode(test_text)
    print(f"\nТест на тексті: {test_text}")
    print(f"Токени: {tokens.tokens}")
    print(f"IDs: {tokens.ids}")
    
    return tokenizer

if __name__ == "__main__":
    # Тренування з параметрами для української історичної мови
    tokenizer = train_ukrainian_tokenizer(
        corpus_path=CORPUS_FILE,
        output_dir=OUTPUT_DIR,
        vocab_size=15000  # Для 18-19 ст. достатньо 10-20K
    )
```

**Як запустити:**
```bash
cd tokenizer
python train_tokenizer_ukrainian.py
```

**Результати:**
```
✓ Токенізатор збережено в ./
✓ Розмір словника: 15000

Будуть створені файли:
- tokenizer_ukrainian.json (повна модель)
- vocab_ukrainian.json (словник для nanoGPT)
```

---

### 2.3 АДАПТАЦІЯ МОДЕЛІ NANOGPT ДЛЯ КИРИЛИЦІ

Оригінальна nanoGPT очікує англійські токени. Нам потрібно:
1. Змінити розмір embedding шару (англійська vocab ~50K, українська ~15K)
2. Налаштувати параметри для нашого датасету

**Файл: `model/nanoGPT_ukrainian.py`**

```python
"""
nanoGPT адаптований для української мови 18-19 століття
На основі https://github.com/karpathy/nanoGPT
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import json

# Гіперпараметри моделі (специфічні для української історичної LLM)
class Config:
    # Архітектура
    n_embd = 384           # Розмір embedding (менше, ніж англійські GPT - економія пам'яті)
    n_head = 6             # Кількість attention heads
    n_layer = 6            # Глибина мережі (6-12 для малої моделі)
    block_size = 256       # Context window (довжина послідовності)
    vocab_size = 15000     # Розмір словника для укр. мови
    dropout = 0.2          # Регуляризація
    bias = True            # Використання bias у лінійних шарах
    
    # Навчання
    batch_size = 64
    learning_rate = 3e-4
    max_iters = 5000       # Епохи (адаптуйте під обсяг ваших даних)
    eval_interval = 500
    eval_iters = 200
    weight_decay = 1e-1    # L2 регуляризація
    grad_clip = 1.0        # Gradient clipping (для стійкості)

class Head(nn.Module):
    """Один attention head"""
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.n_embd // config.n_head)
        self.query = nn.Linear(config.n_embd, config.n_embd // config.n_head)
        self.value = nn.Linear(config.n_embd, config.n_embd // config.n_head)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)      # (B, T, head_dim)
        q = self.query(x)    # (B, T, head_dim)
        
        # Attention scores
        scores = q @ k.transpose(-2, -1) * (1.0 / (k.shape[-1] ** 0.5))  # (B, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        
        # Aggregation
        v = self.value(x)
        out = scores @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multi-head attention"""
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFwd(nn.Module):
    """Feed forward MLP block"""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedFwd(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTUkrainian(nn.Module):
    """Простий GPT для української мови"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding шари
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        
        # Transformer блоки
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Output head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        # Инициализація ваг
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Token + position embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        
        # Transformer
        x = self.blocks(x)  # (B, T, n_embd)
        x = self.ln_f(x)    # (B, T, n_embd)
        
        # Output projection
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Генерація тексту"""
        for _ in range(max_new_tokens):
            # Обрізання контексту до block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Отримання логітів
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # температура
            
            # Top-k sampling (опціонально)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Softmax + sampling
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

if __name__ == "__main__":
    config = Config()
    model = GPTUkrainian(config)
    print(f"Модель параметрів: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Конфіг: {config.__dict__}")
```

---

### 2.4 СКРИПТ НАВЧАННЯ (train.py)

**Файл: `model/train_ukrainian.py`**

```python
"""
Навчання GPT на українських текстах 18-19 століття
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
from tokenizers import Tokenizer
from nanoGPT_ukrainian import GPTUkrainian, Config
import os
from tqdm import tqdm
import numpy as np

# Гіперпараметри
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Використання: {DEVICE}")

class UkrainianDataset(Dataset):
    """Dataset для навчання на українських текстах"""
    def __init__(self, corpus_path, tokenizer_path, block_size=256):
        self.block_size = block_size
        
        # Завантаження токенізатора
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Завантаження та токенізація тексту
        print("Завантаження та токенізація корпусу...")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Токенізація всього тексту
        encoded = self.tokenizer.encode(text)
        self.tokens = torch.tensor(encoded.ids, dtype=torch.long)
        
        print(f"Загальна кількість токенів: {len(self.tokens)}")
        print(f"Кількість прикладів: {len(self.tokens) - block_size}")
    
    def __len__(self):
        return len(self.tokens) - self.block_size - 1
    
    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.block_size]
        y = self.tokens[idx + 1:idx + self.block_size + 1]
        return x, y

def train():
    """Основна функція навчання"""
    
    config = Config()
    
    # 1. Завантаження датасету
    dataset = UkrainianDataset(
        corpus_path="../data/processed/combined_corpus.txt",
        tokenizer_path="../tokenizer/tokenizer_ukrainian.json",
        block_size=config.block_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # 2. Ініціалізація моделі
    model = GPTUkrainian(config).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Модель ініціалізована")
    print(f"  Параметрів: {total_params:,}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  n_embd: {config.n_embd}, n_head: {config.n_head}, n_layer: {config.n_layer}")
    
    # 3. Оптимізатор
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 4. Навчання
    print(f"\n{'='*60}")
    print("РОЗПОЧИНАЄМО НАВЧАННЯ")
    print(f"{'='*60}")
    
    model.train()
    total_steps = 0
    losses = []
    
    for epoch in range(config.max_iters // len(dataloader) + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            # Статистика
            epoch_loss += loss.item()
            losses.append(loss.item())
            total_steps += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Оцінювання (опціонально)
            if total_steps % config.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    # Генерація тестового тексту
                    prompt = torch.tensor([[128]], dtype=torch.long).to(DEVICE)  # BOS токен
                    generated = model.generate(
                        prompt,
                        max_new_tokens=100,
                        temperature=0.7,
                        top_k=40
                    )
                    # Декодування
                    gen_tokens = generated[0].cpu().tolist()
                    print(f"\nGenereted (step {total_steps}):")
                    # (потребує додаткового коду для декодування)
                
                model.train()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"\n✓ Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
        
        # Збереження чекпоїнта
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config.__dict__
            }
            torch.save(checkpoint, f'../model/checkpoint_epoch_{epoch}.pt')
            print(f"✓ Чекпоїнт збережено: checkpoint_epoch_{epoch}.pt")
    
    # 5. Збереження фінальної моделі
    torch.save(model.state_dict(), '../model/gpt_ukrainian_final.pt')
    print(f"\n✓ Модель збережена: gpt_ukrainian_final.pt")
    
    # Графік втрат
    print(f"\nФіналізація: {losses[-10:]} (останні 10 значень)")

if __name__ == "__main__":
    train()
```

**Як запустити навчання:**

```bash
cd model
python train_ukrainian.py
```

**Очікувані результати (GPU RTX 3060):**
- Час на епоху: 2-5 хвилин
- Загальний час: 1-3 години для 50-100 епох
- Фінальна loss: 3.5-4.5 (нормально для 18-19 ст. мови)

---

### 2.5 СКРИПТ ГЕНЕРАЦІЇ (generate.py)

**Файл: `model/generate_ukrainian.py`**

```python
"""
Генерація українського тексту після навчання
"""

import torch
import json
from tokenizers import Tokenizer
from nanoGPT_ukrainian import GPTUkrainian, Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_text(
    model_path="gpt_ukrainian_final.pt",
    tokenizer_path="../tokenizer/tokenizer_ukrainian.json",
    prompt_text="Та чув я їх слова",
    max_tokens=200,
    temperature=0.8,
    top_k=50
):
    """
    Генерація українського тексту
    
    Args:
        prompt_text: Вхідний текст для генерації
        max_tokens: Кількість токенів для генерування
        temperature: Творчість (вище = різноманітніше, нижче = більш детерміновано)
        top_k: Вибір з топ-K найймовірніших токенів
    """
    
    # 1. Завантаження моделі та конфігу
    config = Config()
    model = GPTUkrainian(config).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 2. Завантаження токенізатора
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # 3. Кодування промпту
    encoded = tokenizer.encode(prompt_text)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(DEVICE)
    
    # 4. Генерація
    print(f"\nПромпт: {prompt_text}")
    print(f"{'─'*60}")
    
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # 5. Декодування
    generated_tokens = generated[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    print(f"\nГенеровано:\n{generated_text}")
    print(f"{'─'*60}")

if __name__ == "__main__":
    # Приклади генерації з різними промптами
    prompts = [
        "Та чув я їх слова",
        "Божа мова співається",
        "В серці нашому",
    ]
    
    for prompt in prompts:
        generate_text(
            prompt_text=prompt,
            max_tokens=150,
            temperature=0.7,
            top_k=40
        )
        print("\n" + "="*60 + "\n")
```

**Запуск генерації:**

```bash
python generate_ukrainian.py
```

---

## Частина 3: Інтеграція в курсову роботу

### 3.1 Структура курсової з урахуванням цього проєкту

Ваша курсова робота повинна мати такі розділи:

```
КУРСОВА РОБОТА: "Методи навчання AI: Створення української LLM моделі історичного періоду"

1. ВСТУП
   1.1 Актуальність (Чому нам потрібна українська історична LLM?)
   1.2 Проблема (Современні LLM мають сучасний англійський bias)
   1.3 Гіпотеза (Модель, навчена на текстах 18-19 ст., буде генерувати автентичні історичні тексти)

2. ТЕОРЕТИЧНІ ОСНОВИ
   2.1 Transformer архітектура (Self-Attention, Multi-Head Attention)
   2.2 Байт-Pair Encoding (BPE) для токенізації кірилиці
   2.3 Self-Supervised Learning на нерозмічених текстах

3. МЕТОДИ НАВЧАННЯ
   3.1 GPT (Generative Pre-trained Transformer)
   3.2 Fine-tuning vs Training from Scratch (чому ми навчаємо з нуля)
   3.3 Gradient Descent для мовних моделей

4. МЕТОДОЛОГІЯ ЕКСПЕРИМЕНТУ
   4.1 Постановка гіпотези (що хочемо отримати)
   4.2 Збір датасету (де взяти українські тексти 18-19 ст.)
   4.3 Препроцесинг та очищення (OCR, нормалізація)
   4.4 Архітектура моделі (конфіг, гіперпараметри)
   4.5 Метрики оцінювання (perplexity, qualitative analysis)

5. РЕАЛІЗАЦІЯ
   5.1 Адаптація токенізатора під кірилицю
   5.2 Модифікована архітектура nanoGPT
   5.3 Навчання з адаптивним learning rate
   5.4 Чекпоїнти та збереження

6. РЕЗУЛЬТАТИ
   6.1 Таблиця метрик (loss на тренування/валідації)
   6.2 Приклади генерацій:
       - "Та чув я їх слова" →
         [модель генерує історичний текст у дусі Шевченка]
       - "В землі той давній" →
         [генерація у стилі 18 століття]
   6.3 Порівняння з GPT-3.5 (покаже, як модель різається)
   6.4 Аналіз помилок (які типи текстів модель генерує погано)

7. ОБГОВОРЕННЯ
   7.1 Гіпотеза підтвердилась? (так, модель не "знає" про смартфони)
   7.2 Обмеження (мало даних → modell перенавчається)
   7.3 Як покращити (більше даних, більша модель, fine-tuning)
   7.4 Критична перспектива (де модель деградує)

8. ВИСНОВКИ

9. СПИСОК ДЖЕРЕЛ (з посиланнями на все, що використовували)
```

### 3.2 Чек-лист для захисту

Підготуйте:

□ **Розпечатана курсова работа** (40-50 сторінок)
□ **Код на GitHub** або USB (весь проект)
□ **Таблиця результатів:**
  - Loss curves (графік)
  - Приклади генерацій (мінімум 5-10)
  - Порівняння з baseline (простої модель без fine-tuning)

□ **Презентація PowerPoint (8-10 слайдів):**
  1. Титул
  2. Актуальність (чому українська історична LLM важлива)
  3. Мета і задачи
  4. Архітектура (блок-схема Transformer)
  5. Датасет (приклади текстів, розміри)
  6. Результати (графіки, приклади генерацій)
  7. Висновки

□ **Демонстрація на захисті:**
  - Запустити `generate_ukrainian.py` на ноутбуці
  - Показати промпт → генерацію
  - Пояснити, чому модель генерує історичні тексти

---

## Частина 4: Практичний чек-лист по темпам

**Тиждень 1-2:**
- [ ] Визначити точний період і регіон (1780-1880, Україна)
- [ ] Почати збір текстів з Chtyvo.org.ua, Internet Archive
- [ ] Завантажити мінімум 50 книг (~100-200 МБ)

**Тиждень 3-4:**
- [ ] Обробка OCR (якщо газети у PDF)
- [ ] Очищення та об'єднання текстів
- [ ] Тренування токенізатора

**Тиждень 5-6:**
- [ ] Адаптація nanoGPT коду
- [ ] Налаштування гіперпараметрів
- [ ] Запуск навчання на GPU (Google Colab)

**Тиждень 7-8:**
- [ ] Генерація текстів та аналіз
- [ ] Написання курсової роботи
- [ ] Підготовка презентації

---

## ФАЙЛОВИ СТРУКТУРА ПРОЕКТУ (ФІНАЛЬНА)

```
my_ukrainian_llm/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── books/
│   │   │   ├── shevchenko_poems.txt
│   │   │   ├── kotliarevskyi_eneida.txt
│   │   │   ├── franko_poems.txt
│   │   │   ├── mirnyj_romany.txt
│   │   │   └── ... (ще ~50 файлів)
│   │   ├── newspapers/
│   │   │   ├── 1848_ruska_vidomist.txt
│   │   │   ├── 1860_osnova.txt
│   │   │   └── ...
│   │   └── documents/
│   │       ├── letters/
│   │       └── decrees/
│   │
│   └── processed/
│       ├── combined_corpus.txt (~500MB-1GB)
│       ├── corpus_stats.txt
│       └── vocab_ukrainian.json
│
├── tokenizer/
│   ├── train_tokenizer_ukrainian.py
│   ├── tokenizer_ukrainian.json
│   └── vocab_ukrainian.json
│
├── model/
│   ├── nanoGPT_ukrainian.py
│   ├── train_ukrainian.py
│   ├── generate_ukrainian.py
│   ├── gpt_ukrainian_final.pt (модель після навчання)
│   ├── checkpoint_epoch_5.pt
│   ├── checkpoint_epoch_10.pt
│   └── config.json
│
├── results/
│   ├── generated_samples.txt
│   ├── loss_curves.png
│   ├── metrics.json
│   └── analysis/
│
└── coursework/
    ├── курсова_робота.docx
    ├── презентація.pptx
    └── додатки/
        ├── код_на_github.md
        ├── таблиці_результатів.xlsx
        └── приклади_генерацій.txt
```

---

## РЕКОМЕНДОВАНІ ПОСИЛАННЯ ДЛЯ НАВЧАННЯ

1. **Про Transformers:**
   - https://arxiv.org/abs/1706.03762 (оригінальна стаття "Attention is All You Need")
   - https://www.youtube.com/watch?v=dichIcUZc-0 (відео про Self-Attention)

2. **Про nanoGPT:**
   - https://github.com/karpathy/nanoGPT (оригінальний репозиторій)
   - https://www.youtube.com/watch?v=kCc8FjSHcWE (Andrej Karpathy LLM Tutorial)

3. **Про токенізацію:**
   - https://huggingface.co/docs/tokenizers/ (Документація HuggingFace Tokenizers)
   - https://arxiv.org/abs/1508.07909 (стаття про BPE)

4. **Українська мова та обробка:**
   - https://chtyvo.org.ua/ (крайне важливо!)
   - https://nbuv.gov.ua/ (Національна бібліотека України)

---

## КОНТРОЛЬНИЙ ПЕРЕЛІК ПЕРЕД ЗАХИСТОМ

✅ **Технічна частина:**
- [ ] Модель навчена (мінімум 50 епох)
- [ ] Loss плавно спадає
- [ ] Генерація працює (не видає gibberish)
- [ ] Код на GitHub з README

✅ **Наукова частина:**
- [ ] Гіпотеза чітко сформульована
- [ ] Методологія описана детально
- [ ] Результати rappresentarе в таблицях/графіках
- [ ] Критичний аналіз обмежень

✅ **Оформлення:**
- [ ] Курсова 40-50 сторінок
- [ ] Презентація 8-10 слайдів
- [ ] Список джерел 20+ статей/книг
- [ ] Всі таблиці та графіки нумеровані

✅ **Захист:**
- [ ] Промова 5-7 хвилин
- [ ] Демонстрація моделі на ноутбуці
- [ ] Готові відповіді на можливі питання

---

**Удачи! Це неймовірно цікава і оригінальна тема. Вона виділить вас серед інших студентів і гарантовано викличе цікавість викладачів.**
