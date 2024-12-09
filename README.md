# AI-Cybersecurity-Project
This project focuses on building a robust classification pipeline for analyzing URLs and their corresponding titles to identify potential phishing attempts. The project employs various machine learning and deep learning models, integrates metadata, and explores adversarial robustness to enhance detection accuracy.

## Key Features:
- **Custom Tokenizer**: Efficient tokenization of URLs for model input.
- **Diverse Models**: Implementation of several models including Logistic Regression, MLP, Transformer, LSTM, KNN, and XGBoost.
- **Adversarial Testing**: Evaluation of model robustness against adversarial perturbations such as casing changes, random Unicode characters, and similar-looking character replacements.
- **Extensive Metrics**: Accuracy, precision, recall, F1-score, confusion matrices, AUC-ROC metrics for all models.
- **Custom Training Pipeline**: Designed for PyTorch, ensuring flexibility and scalability.

## Dataset
The dataset consists of URLs labeled as phishing or legitimate.

### 1. Features:
- URLs (tokenized and preprocessed).
- Metadata: Numerical features provided in the dataset.
- Labels: Binary classification target (phishing or legitimate).
### 2.Dataset Preparation:
- Preprocessing:
- Normalized metadata.
- Tokenized URL strings.
= Generated attention masks.
### 3.Splitting:
- Stratified splitting into training, validation, and testing datasets.
