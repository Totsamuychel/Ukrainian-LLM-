# Навчання GPT на українській мові (18-19 століття)

## 🎯 Оптимізація для української мови

### Ключові зміни:

1. **Vocab Size: 50K → 15K**
   - Англійська мова потребує ~50K токенів
   - Українська мова потребує ~15K токенів
   - Економія пам'яті та прискорення навчання

2. **Embedding Layer**
   - Оригінальний: 768 × 50K = 38.4M параметрів
   - Оптимізований: 640 × 15K = 9.6M параметрів
   - Економія: 75% параметрів у embedding шарі

3. **Параметри моделі**
   - Block size: 1024 → 512 (менший контекст)
   - Layers: 12 → 10 (менше шарів)
   - Heads: 12 → 10 (менше голів)
   - Embedding: 768 → 640 (менша розмірність)
   - Dropout: 0.0 → 0.1 (регуляризація)

4. **Параметри навчання**
   - Learning rate: 6e-4 → 4e-4 (стабільність)
   - Max iterations: 600K → 30K (менший датасет)
   - Warmup: 2000 → 1000 (менше прогріву)
   - Batch size: 12 → 32 (оптимально для RTX 3090)

## 📊 Порівняння конфігурацій

| Параметр | Оригінальна | Українська | Економія |
|----------|------------|-----------|----------|
| Vocab Size | 50K | 15K | 70% ↓ |
| Block Size | 1024 | 512 | 50% ↓ |
| Layers | 12 | 10 | 17% ↓ |
| Embedding Dim | 768 | 640 | 17% ↓ |
| Total Params | 124M | ~50M | 60% ↓ |
| VRAM (batch=32) | OOM | 18GB | ✅ |

## 🚀 Кроки навчання

### 1. Обучення токенайзера (оптимізований для української)
```bash
python train_tokenizer_hf.py
```
- Vocab size: 15K (замість 50K)
- Час: ~5-10 хвилин
- Вихід: `tokenizer_london/tokenizer.json`

### 2. Підготовка датасету
```bash
python prepare_dataset.py
```
- Читає: `dataset/dataset_train_clean.txt` (118MB)
- Створює: `gpt_data_london/train.bin`, `val.bin`
- Час: ~2-3 хвилини

### 3. Навчання моделі (оптимізовано для RTX 3090)
```bash
# Базовий запуск
python train_ukrainian.py

# Або з параметрами
python train_ukrainian.py --batch_size=32 --max_iters=30000

# Відновлення навчання
python train_ukrainian.py --init_from=resume
```

**Параметри для RTX 3090:**
- Batch size: 32 (оптимально)
- Block size: 512
- Max iterations: 30000
- Очікуваний час: 4-6 годин
- VRAM: ~18-20GB

## 📈 Очікувані результати

### Розмір моделі
- Параметрів: ~50M (замість 124M)
- Checkpoint: ~200MB (замість 500MB)
- Inference: швидше на 40%

### Якість
- Для датасету 118MB: хороша якість
- Порівняння з v0 (187MB): подібна якість
- Порівняння з v0.5 (435MB): трохи гірша, але швидше

### Час навчання
- RTX 3090: 4-6 годин (30K ітерацій)
- RTX 4060: 12-16 годин
- CPU: не рекомендується

## 🔧 Файли конфігурації

### train_ukrainian.py
- Спеціалізований скрипт для української мови
- Оптимізовані параметри за замовчуванням
- Українські коментарі

### train_tokenizer_hf.py
- Vocab size: 15K (замість 50K)
- ByteLevel pre-tokenizer для кириллиці
- Оптимізовано для історичних текстів

### model.py
- Оновлені default параметри
- Block size: 512
- Vocab size: 15360 (15K + padding)

### train.py
- Оновлені параметри для української мови
- Менший batch size за замовчуванням
- Менше ітерацій

## 💾 Структура файлів

```
london_1800_1850_v0/
├── dataset/
│   ├── dataset_train_clean.txt (118MB)
│   └── metadata.csv
├── tokenizer_london/
│   ├── tokenizer.json (5.1MB)
│   ├── vocab.json (1.6MB)
│   └── merges.txt (1.3MB)
├── gpt_data_london/
│   ├── train.bin (~100MB)
│   ├── val.bin (~11MB)
│   └── meta.pkl
├── out_ukrainian/
│   └── ckpt.pt (чекпоінти)
├── train_ukrainian.py ✨ (новий)
├── train_tokenizer_hf.py (оновлений)
├── prepare_dataset.py (оновлений)
├── train.py (оновлений)
└── model.py (оновлений)
```

## ⚠️ Потенційні проблеми

### CUDA out of memory
```bash
# Зменш batch size
python train_ukrainian.py --batch_size=16

# Або зменш embedding dimension
python train_ukrainian.py --n_embd=512
```

### Повільне навчання
```bash
# Включи компіляцію (PyTorch 2.0+)
python train_ukrainian.py --compile=True

# Або збільш batch size
python train_ukrainian.py --batch_size=48
```

### Низька якість
```bash
# Збільш max_iters
python train_ukrainian.py --max_iters=50000

# Або зменш learning rate
python train_ukrainian.py --learning_rate=2e-4
```

## 📝 Рекомендації

1. **Почни з train_ukrainian.py** - він оптимізований для твого датасету
2. **Моніторь loss** - повинен зменшуватися з часом
3. **Зберігай чекпоінти** - вони автоматично зберігаються
4. **Експериментуй** - спробуй різні параметри

## 🎓 Навчання

Для навчання на українській мові:
- Vocab size: 15K (оптимально)
- Block size: 256-512 (залежно від VRAM)
- Layers: 8-12 (залежно від датасету)
- Batch size: 16-64 (залежно від GPU)

## 📚 Посилання

- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/)
- [PyTorch CUDA](https://pytorch.org/get-started/locally/)
