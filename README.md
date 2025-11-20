# Plant Nucleotide Transformer Prediction Tool

A specialized PyTorch implementation for plant genomic sequence classification using the Nucleotide Transformer model, with local prediction capabilities optimized for plant biology research.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Model Download](#-model-download)
- [Prediction](#-prediction)
- [Input Format](#-input-format)
- [Output Format](#-output-format)
- [Troubleshooting](#-troubleshooting)

## 🎯 Overview

This project provides specialized tools for plant genomic sequence classification using a fine-tuned Nucleotide Transformer model. The model is optimized for predicting functional elements in plant genomes, trained on diverse plant species data.

**Pre-trained Model Checkpoint:**
- `plant_best_epoch66_auc0.9588.pt` (AUC: 0.9588) - Optimized for plant genomic sequences

## ✨ Features

### 🌱 Plant-Specific Optimizations
- **Multi-species training** on major crop genomes
- **Plant-specific sequence handling** optimized for plant genomic patterns
- **High-accuracy classification** for plant regulatory elements

### 🔬 Technical Features
- 🧬 Optimized for plant genomic sequence classification
- 🌿 Local model loading (no internet required for prediction)
- 📊 Comprehensive prediction outputs with confidence scores
- ⚡ Batch prediction for large plant genomic datasets
- 🎯 High-accuracy classification (AUC > 0.95 on plant datasets)

## 🛠 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 4GB+ RAM

### Install Dependencies
```bash
pip install torch transformers pandas numpy scikit-learn tqdm
```

## 📥 Model Download

### Pre-trained Checkpoint
Download the fine-tuned plant model checkpoint:

```bash
# Download plant_best_epoch66_auc0.9588.pt from Google Drive
# Place the file in your project directory
```

### Base Model Setup
The prediction script can automatically download the base Nucleotide Transformer model:

```bash
# Download base model to local directory
python plant_nt_predict.py --download_model
```

This will create a `local_models/` directory containing the model files for offline use.

## 🚀 Quick Start

### 1. Prepare Your Data
Create a CSV file named `input.csv` with your plant sequences:

```csv
Off,Epi_satics,CFD_score,CCTop_Score,Moreno_Score,CROPIT_Score,MIT_Score
AACTGAATATAAAAATCCTATGG,0,0,0.85,0.105721231,0.247619048,0.999000864
AGCGGCGTCGGCGGGGTCCTCGG,0,0.0535,0.9,0.139114361,0.176190476,0.999157044
AGAGGAGATTTTAGCTGCTTTGG,1,0.0937,0.905,0.109747353,0.147619048,0.996678771
GAAGAAGAGGGTGCTGTTCGCGG,0,0,0.885,0.17321379,0.19047619,0.979384701
GCTTGGCCACAAGGCATTGCGAG,1,0.0281,0.96,0.253765489,0.114285714,0.993211988
```

### 2. Run Prediction
```bash
python plant_nt_predict.py \
    --checkpoint plant_best_epoch66_auc0.9588.pt \
    --input_csv input.csv \
    --output_csv output.csv \
    --download_model  # Auto-download base model if needed
```

## 🔮 Prediction

### Basic Usage
```bash
python plant_nt_predict.py \
    --checkpoint plant_best_epoch66_auc0.9588.pt \
    --input_csv input.csv \
    --output_csv output.csv
```

### Advanced Options
```bash
python plant_nt_predict.py \
    --checkpoint plant_best_epoch66_auc0.9588.pt \
    --input_csv input.csv \
    --output_csv output.csv \
    --local_model_dir ./local_models/nucleotide-transformer-2.5b-multi-species \
    --batch_size 32 \
    --max_length 64 \
    --device cuda
```

### Command Line Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | Required | Path to trained model checkpoint (.pt file) |
| `--input_csv` | `input.csv` | Input CSV file with plant sequences |
| `--output_csv` | `output.csv` | Output CSV file for predictions |
| `--local_model_dir` | `./local_models/...` | Local directory containing base model |
| `--batch_size` | 16 | Batch size for prediction |
| `--max_length` | 64 | Maximum sequence length |
| `--device` | cuda | Device for inference (cuda or cpu) |
| `--download_model` | False | Auto-download base model if missing |



## 📈 Output Format

The prediction output `output.csv` includes:

```csv
sequence,prediction,probability_class_0,probability_class_1,confidence,predicted_label
ATCGATCGATCG,1,0.023,0.977,0.977,positive
GCTAGCTAGCTA,0,0.891,0.109,0.891,negative
TTTTAAAACCCC,1,0.156,0.844,0.844,positive
```

### Output Columns Description
| Column | Description |
|--------|-------------|
| `sequence` | Original input sequence |
| `prediction` | Binary prediction (0 or 1) |
| `probability_class_0` | Probability of negative class |
| `probability_class_1` | Probability of positive class |
| `confidence` | Maximum prediction confidence |
| `predicted_label` | Human-readable label (positive/negative) |


## 🐛 Troubleshooting

** Missing Base Model**
```bash
# Auto-download base model
python plant_nt_predict.py --download_model

# Or specify local path
python plant_nt_predict.py --local_model_dir /path/to/local/model
```

### Performance Tips
- Use `--device cuda` for GPU acceleration
- Adjust `--batch_size` based on available GPU memory (8-32 recommended)
- For long plant genomic sequences, increase `--max_length` (up to 1000)
- Use `--download_model` once, then reuse local model for faster startup

## 📊 Performance

The provided plant model checkpoint achieves:
- **AUC: 0.9588**
- **Accuracy: >92%**
- **F1-Score: >0.91**

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests related to plant genomics applications.

## 📄 License

This project is for academic and research use. Please check the original Nucleotide Transformer license for commercial use.

## 🙏 Acknowledgments

- InstaDeepAI for the Nucleotide Transformer model
- Hugging Face for the Transformers library  
- The plant genomics community for datasets and tools

---

**Note:** The `plant_best_epoch66_auc0.9588.pt` checkpoint file is available for download via Google Drive. Please contact the maintainers for access.

**For questions and support:**
- 📧 Email: your-email@domain.com
- 💬 Issues: GitHub Issues
- 🐛 Bug Reports: Please include your plant species and sequence length information
