# Інструкція з обучення моделі на українській кириллиці

## Оцінка датасету
- **Розмір**: 112.77 MB
- **Мова**: Українська
- **Період**: 1800-1850
- **Статус**: ✅ Готовий до обучення

Це хороший розмір датасету для початкового обучення. Порівняння:
- v0: ~187MB → 16M параметрів
- v0.5: ~435MB → 123M параметрів
- Твій датасет: ~113MB → рекомендується 10-20M параметрів

## Кроки обучення

### 1. Обучення токенайзера
```bash
python london_1800_1850_v0/train_tokenizer_hf.py
```
Це створить:
- `tokenizer_london/tokenizer.json` - основний файл
- `tokenizer_london/vocab.json` - словник
- `tokenizer_london/merges.txt` - правила об'єднання

**Час**: ~2-5 хвилин

### 2. Підготовка датасету
```bash
python london_1800_1850_v0/prepare_dataset.py
```
Це створить:
- `gpt_data_london/train.bin` - 90% даних для обучення
- `gpt_data_london/val.bin` - 10% даних для валідації
- `gpt_data_london/meta.pkl` - метаданні

**Час**: ~1-2 хвилини

### 3. Обучення моделі
```bash
# На одному GPU (рекомендується для початку)
python london_1800_1850_v0/train.py --batch_size=16 --compile=False

# Або з DDP на кількох GPU
torchrun --standalone --nproc_per_node=4 london_1800_1850_v0/train.py
```

**Рекомендовані параметри для твого датасету:**
- `batch_size=16` (для GPU з 6GB+ пам'яті)
- `n_layer=8` (менше шарів для меншого датасету)
- `n_head=8` (менше голів)
- `n_embd=512` (менша розмірність)
- `max_iters=50000` (менше ітерацій)

Приклад:
```bash
python london_1800_1850_v0/train.py \
  --batch_size=16 \
  --n_layer=8 \
  --n_head=8 \
  --n_embd=512 \
  --max_iters=50000 \
  --compile=False
```

## Перевірка скриптів

### ✅ train_tokenizer_hf.py
- Адаптований для української кириллиці
- Використовує ByteLevel pre-tokenizer
- Vocab size: 50000

### ✅ prepare_dataset.py
- Оновлений для роботи з твоїм датасетом
- Завантажує токенайзер з `tokenizer_london/tokenizer.json`
- Читає дані з `dataset/dataset_train_clean.txt`
- Створює train/val split 90/10

### ✅ train.py
- Оновлений для завантаження даних з `gpt_data_london/`
- Додана перевірка наявності meta.pkl
- Готовий до обучення

### ✅ model.py
- Без змін потрібно
- Підтримує довільний vocab_size

### ✅ configurator.py
- Без змін потрібно
- Дозволяє перевизначати параметри з командного рядка

## Файли датасету

```
london_1800_1850_v0/
├── dataset/
│   ├── dataset_train_clean.txt (112.77 MB) ← твій датасет
│   └── metadata.csv ← створено
├── tokenizer_london/ ← буде створено після кроку 1
│   ├── tokenizer.json
│   ├── vocab.json
│   └── merges.txt
├── gpt_data_london/ ← буде створено після кроку 2
│   ├── train.bin
│   ├── val.bin
│   └── meta.pkl
└── out/ ← буде створено після кроку 3
    └── ckpt.pt (чекпоінти моделі)
```

## Рекомендації

1. **Почни з малої моделі** - 8-12M параметрів для твого датасету
2. **Моніторь loss** - повинен зменшуватися з часом
3. **Зберігай чекпоінти** - вони автоматично зберігаються в `out/`
4. **Експериментуй** - спробуй різні параметри

## Потенційні проблеми

- Якщо `prepare_dataset.py` падає - перевір що `tokenizer_london/tokenizer.json` існує
- Якщо `train.py` падає - перевір що `gpt_data_london/` містить train.bin і val.bin
- Якщо GPU memory error - зменш `batch_size` або `n_embd`
