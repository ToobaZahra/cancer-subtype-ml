# Cancer Subtype Classification using Machine Learning

Hi, I'm Tooba Zahra — an MS Bioinformatics student at COMSATS University Islamabad (graduating Feb 2027). This project applies machine learning to classify breast cancer subtypes from gene expression data.

## What this project does

Given 30 features extracted from breast cancer cell nuclei, this model predicts whether a tumor is malignant or benign. This kind of analysis is directly applicable to cancer genomics research and clinical decision support.

## Results

- **Accuracy: 96.5%**
- Precision: 0.97 | Recall: 0.96 | F1-score: 0.96
- Most important features: worst area, worst concave points, mean concave points

## Pipeline Steps

1. Load breast cancer gene expression dataset (569 samples, 30 features)
2. Split into train/test sets (80/20)
3. Standardize features using StandardScaler
4. Train Random Forest Classifier (100 estimators)
5. Evaluate with accuracy, classification report, confusion matrix
6. Visualize with PCA plot and feature importance

## Outputs

- `confusion_matrix.png` — model prediction results
- `feature_importance.png` — top 10 most important features
- `pca_plot.png` — PCA visualization of cancer subtypes

## Tools used

| Tool | Purpose |
|------|---------|
| scikit-learn | ML model, preprocessing, evaluation |
| pandas | Data handling |
| numpy | Numerical operations |
| matplotlib | Plotting |
| seaborn | Heatmap visualization |

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python3 cancer_ml.py
```

## Dataset

Breast Cancer Wisconsin dataset from scikit-learn — a standard benchmark dataset in biomedical ML research.
