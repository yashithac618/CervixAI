# CervixAI — AI-Assisted Cervical Cytology Analysis

CervixAI is a research-grade tool for cervical cytology image classification. It uses a DenseNet121 model trained on the SIPaKMeD dataset to classify cervical cell images, estimate a clinical risk level, and generate a full pathology-style report — complete with a Grad-CAM heatmap that visually explains which region of the cell drove the prediction.

![Dashboard]<img width="1887" height="970" alt="Screenshot 2026-07-03 221219" src="https://github.com/user-attachments/assets/f1031ad5-93aa-4683-9918-a70e5a40d58f" />

---

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Model Details](#model-details)
- [Tech Stack](#tech-stack)
- [API Reference](#api-reference)
- [Disclaimer](#disclaimer)

---

## Features

### Classification
Upload a single cell image or a full batch, and CervixAI classifies each image into one of five cytology classes:

| Class | Description |
|---|---|
| Dyskeratotic | Abnormal keratinization pattern |
| Koilocytotic | HPV-associated cellular changes |
| Metaplastic | Squamous metaplasia |
| Parabasal | Immature squamous cells |
| Superficial-Intermediate | Normal mature squamous cells |

### Risk Stratification
Each prediction is automatically mapped to a clinical risk tier — **Normal**, **Low**, **Moderate**, or **High Risk** — based on the predicted cell type.

### Explainable AI (Grad-CAM)
Every analysis produces a Grad-CAM heatmap overlay, highlighting the exact region of the image that most influenced the model's decision — shown side-by-side with the original.

![Grad-CAM heatmap analysis]<img width="1882" height="975" alt="Screenshot 2026-07-03 221203" src="https://github.com/user-attachments/assets/4f358619-2549-4bcd-9b28-a5016fa1256c" />


### Auto-Generated Reports
Each result includes:
- Classification and confidence score
- Full probability breakdown across all five classes
- A plain-language clinical note
- An educational summary explaining the finding, its significance, and the typical next clinical step

### PDF Export
Export any report as a clean, single-page clinical PDF, including scan ID, date, risk badge, classification breakdown, both images (original and heatmap), the clinical note, and the educational summary.

![Exported PDF report]<img width="666" height="942" alt="Screenshot 2026-07-03 221551" src="https://github.com/user-attachments/assets/01b26c98-3f27-44c4-8b95-08b0151f762e" />

### Second-Opinion Workflow
Flag any result for manual review directly from the report card — useful for borderline or low-confidence predictions.

### Dashboard & History
- **Dashboard** — running totals, a risk-distribution donut chart, and a feed of recent analyses
- **History** — every past analysis is stored and searchable, with the full report (including heatmap) retrievable at any time

### Interface
Single-file, responsive UI with dark mode enabled by default (toggle in the top-right).

---

## Getting Started



1. Open the **Analyze** tab
2. Drag and drop one or more cervical cell images (JPG, PNG, or TIFF)
3. Click **Run Analysis**
4. Review the classification, confidence score, Grad-CAM heatmap, and clinical note
5. Export a PDF report, or flag the result for a second opinion if it's borderline
6. Visit the **Dashboard** for aggregate stats, or **History** to browse all past scans

---

## Model Details

| Property | Value |
|---|---|
| Architecture | DenseNet121 (`torchvision.models`) with a custom classifier head |
| Training data | SIPaKMeD cervical cytology dataset |
| Classes | 5 (listed above) |
| Explainability | Grad-CAM, computed on the final dense block's feature maps |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch, DenseNet121 |
| Backend | FastAPI, SQLite |
| Frontend | Single-file HTML/CSS/JS, jsPDF for report export |
| Deployment | Docker, Hugging Face Spaces |

---

## Disclaimer

CervixAI is a research and educational tool. It is **not** a certified medical device, and its output must **not** be used as a substitute for professional diagnosis, biopsy, or clinical judgment. Always consult a qualified pathologist or clinician for actual patient care decisions.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
