# Estimating Urban Poverty via Satellite Image Classification of Urban Morphology

**A Swin Transformer Approach for Greater Dhaka, Bangladesh**

> Munshi Abu Muhammad · Muhammad Ali  
> Department of Computer Science and Engineering, BRAC University

---

## Overview

This project presents a fully automated, survey-free pipeline for estimating urban poverty across the Greater Dhaka Metropolitan Area using Sentinel-2 satellite imagery and a fine-tuned Swin-Tiny Transformer.

Instead of costly ground surveys, we use the **Metal Roof Index** (SWIR/NIR ratio) derived from Sentinel-2 spectral bands to automatically label poverty levels across the city. A deep learning model then learns to classify satellite image patches into three poverty classes: **Wealthy**, **Medium**, and **Poor**.

---

## Results

| Metric | Swin-Base / 8k | Swin-Tiny / 17.7k |
|--------|---------------|-------------------|
| Best Val Accuracy | 93.5% | **97.19%** |
| Test Accuracy | 94% | **97%** |
| Macro F1 | 76% | **91%** |
| Wealthy F1 | 0.98 | 0.99 |
| Medium F1 | 0.59 | 0.82 |
| Poor F1 | 0.72 | 0.92 |
| Overfitting | Heavy | Mild |
| Poor→Wealthy confusion | Present | **Zero** |

---

## Pipeline

```
Google Earth Engine (Sentinel-2)
        ↓
Metal Roof Index → Poverty Score per 800m grid cell
        ↓
Python Patching Pipeline → 17,771 labeled 112×112px patches
        ↓
Swin-Tiny Fine-tuning (Kaggle T4 GPU)
        ↓
Poverty Map of Dhaka
```

---

## Repository Structure

```
urban-poverty-dhaka/
│
├── gee/
│   └── poverty_labels_export.js     # Google Earth Engine script
│
├── dataset/
│   └── making_more_patches.ipynb    # Colab patching pipeline
│
├── model/
│   └── train_swin_tiny.ipynb        # Kaggle training notebook
│
├── results/
│   ├── confusion_matrix.png
│   ├── poverty_map.png
│   ├── training_curves.png
│   └── attention_maps.png
│
├── paper/
│   ├── urban_poverty_swin_tiny.pdf  # IEEE-format paper
│   └── progress_report.pdf          # Full project progress report
│
├── requirements.txt
└── README.md
```

---

## Dataset

- **17,771 georeferenced 112×112px RGB satellite patches**
- Study area: 120×120 km centered on BRAC University, Dhaka
- Labels: Wealthy (0–0.33), Medium (0.33–0.66), Poor (0.66–1.0)
- Available on Kaggle: [poverty-dataset-17-7k](https://www.kaggle.com/datasets/mamalaga/poverty-dataset-17-7k)

---

## Model

- **Architecture:** `swin_tiny_patch4_window7_224` (timm, pretrained on ImageNet-1K)
- **Trainable params:** 15.4M / 27.5M total
- **Fine-tuned layers:** `layers.3`, layer norm, classification head
- **Optimizer:** AdamW (lr=1e-4, weight decay=0.05)
- **Scheduler:** Cosine Annealing
- **Loss:** Class-weighted CrossEntropy [0.3773, 4.2372, 8.8150]
- **Platform:** Kaggle T4 GPU, 30 epochs, batch size 64

---

## How to Run

### 1. Google Earth Engine
- Open `gee/poverty_labels_export.js` in the [GEE Code Editor](https://code.earthengine.google.com/)
- Run to export satellite tiles and poverty labels to Google Drive

### 2. Dataset Generation
- Open `dataset/making_more_patches.ipynb` in Google Colab
- Mount Drive and run all cells
- Outputs saved to `MyDrive/BRACU_DATASET/34k/dataset_34k/`

### 3. Model Training
- Upload dataset to Kaggle
- Open `model/train_swin_tiny.ipynb` in Kaggle
- Enable T4 GPU accelerator and run all cells

### 4. Install dependencies (local)
```bash
pip install -r requirements.txt
```

---

## Study Area

Greater Dhaka Metropolitan Area, Bangladesh  
Center: 23.7806°N, 90.4003°E (BRAC University)  
Buffer: 60 km radius → 120×120 km coverage

Neighborhoods covered: Gulshan, Banani, Uttara, Mirpur, Tejgaon, Old Dhaka, Keraniganj, Demra, Narayanganj, Savar, Ashulia

---

## Citation

If you use this work, please cite:

```
Munshi Abu Muhammad, Muhammad Ali,
"Estimating Urban Poverty via Satellite Image Classification of Urban Morphology:
A Swin Transformer Approach for Greater Dhaka",
BRAC University, 2024.
```

---

## Acknowledgements

- [Google Earth Engine](https://earthengine.google.com/) for satellite data access
- [Kaggle](https://www.kaggle.com/) for GPU computing resources
- [Copernicus/Sentinel-2](https://sentinel.esa.int/) for satellite imagery
