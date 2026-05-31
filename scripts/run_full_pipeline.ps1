# Повний pipeline для навчання GPT на українській мові (Windows)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "НАВЧАННЯ GPT НА УКРАЇНСЬКІЙ МОВІ" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Крок 1: Перевірка оптимізацій
Write-Host ""
Write-Host "📊 Крок 1: Перевірка оптимізацій..." -ForegroundColor Yellow
python check_ukrainian_optimization.py
if ($LASTEXITCODE -ne 0) { exit 1 }

# Крок 2: Обучення токенайзера
Write-Host ""
Write-Host "🔤 Крок 2: Обучення токенайзера (15K vocab)..." -ForegroundColor Yellow
python train_tokenizer_hf.py
if ($LASTEXITCODE -ne 0) { exit 1 }

# Крок 3: Підготовка датасету
Write-Host ""
Write-Host "📊 Крок 3: Підготовка датасету..." -ForegroundColor Yellow
python prepare_dataset.py
if ($LASTEXITCODE -ne 0) { exit 1 }

# Крок 4: Навчання моделі
Write-Host ""
Write-Host "🚀 Крок 4: Навчання моделі на RTX 3090..." -ForegroundColor Yellow
python train_ukrainian.py
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "✅ НАВЧАННЯ ЗАВЕРШЕНО!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Результати збережено в: out_ukrainian/" -ForegroundColor Green
Write-Host "Чекпоінт: out_ukrainian/ckpt.pt" -ForegroundColor Green
