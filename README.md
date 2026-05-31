# Ukrainian Historical LLM (nanoGPT)

## Project Overview
This is a student research project focused on adapting a small-scale Large Language Model (based on the **nanoGPT** architecture) for the Ukrainian language. The primary focus is on historical literature from the 18th and 19th centuries, including works by Taras Shevchenko, Hryhoriy Kvitka-Osnovyanenko, Ivan Nechuy-Levytsky, and others.

The goal is to explore the effectiveness of character-level and BPE tokenization on historical Ukrainian texts and to train a generative model capable of capturing the stylistic and linguistic nuances of that era.

## Features
- **Custom BPE Tokenizer**: Optimized for Ukrainian Cyrillic with a 15,000 token vocabulary to handle historical spellings and morphological richness.
- **nanoGPT Architecture**: A lightweight Transformer-based model optimized for training on consumer-grade hardware (e.g., NVIDIA RTX 3090).
- **Automated Data Pipeline**: Scripts for extracting text from PDF, EPUB, and DOCX formats, followed by cleaning and normalization for training.
- **Historical Focus**: Specifically curated dataset of 18th-19th century Ukrainian literature.

## Project Structure
- `src/`: Core source code.
  - `data/`: Scrapers and cleaning utilities.
  - `model/`: nanoGPT model definition and configuration.
  - `tokenizer/`: BPE tokenizer training and utility scripts.
- `scripts/`: Entry points for the main workflow.
  - `prepare_dataset.py`: Processes raw files into training data.
  - `train.py`: Main training loop for the LLM.
  - `sample.py`: Inference script for generating text.
- `data/`: Local storage for raw and processed datasets (ignored by Git).
- `out/`: Directory for model checkpoints and training logs (ignored by Git).
- `docs/`: Project documentation, research notes, and assessment reports.

## Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install torch numpy tokenizers tqdm PyMuPDF ebooklib beautifulsoup4
   ```

### Usage
1. **Data Preparation**: Place your raw files in `data/raw/` and run:
   ```bash
   python scripts/prepare_dataset.py
   ```
2. **Training**: Start the training process:
   ```bash
   python scripts/train.py
   ```
3. **Inference**: Generate text using a trained checkpoint:
   ```bash
   python scripts/sample.py
   ```

## License
This project is for educational and research purposes.
