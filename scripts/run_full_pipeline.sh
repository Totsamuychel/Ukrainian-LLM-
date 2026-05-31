#!/bin/bash
# Повний pipeline для навчання GPT на українській мові

set -e

echo "=========================================="
echo "НАВЧАННЯ GPT НА УКРАЇНСЬКІЙ МОВІ"
echo "=========================================="

# Крок 1: Перевірка оптимізацій
echo ""
echo "📊 Крок 1: Перевірка оптимізацій..."
python check_ukrainian_optimization.py

# Крок 2: Обучення токенайзера
echo ""
echo "🔤 Крок 2: Обучення токенайзера (15K vocab)..."
python train_tokenizer_hf.py

# Крок 3: Підготовка датасету
echo ""
echo "📊 Крок 3: Підготовка датасету..."
python prepare_dataset.py

# Крок 4: Навчання моделі
echo ""
echo "🚀 Крок 4: Навчання моделі на RTX 3090..."
python train_ukrainian.py

echo ""
echo "=========================================="
echo "✅ НАВЧАННЯ ЗАВЕРШЕНО!"
echo "=========================================="
echo ""
echo "Результати збережено в: out_ukrainian/"
echo "Чекпоінт: out_ukrainian/ckpt.pt"
