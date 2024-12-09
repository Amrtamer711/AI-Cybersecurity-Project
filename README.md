# AI-Cybersecurity-Project
![image](https://github.com/user-attachments/assets/e1db3cf5-c66c-465d-a4c3-58f951e00ecd)

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
### 2.  Dataset Preparation:
- Preprocessing:
- Normalized metadata.
- Tokenized URL strings.
- Generated attention masks.
### 3. Splitting:
- Stratified splitting into training, validation, and testing datasets.

## Models
The following models were implemented and evaluated:

- Logistic Regression
- Multi-Layer Perceptron (MLP)
- XGBoost
- K-Nearest Neighbors (KNN)
- Transformer
- LSTM

Each model was implemented from scratch or adapted with libraries like PyTorch and scikit-learn, integrating a shared preprocessing pipeline.

## Methodology
### 1. Tokenization:

- URLs were tokenized using a custom tokenizer, mapping characters to indices.
- Metadata was normalized and converted into tensors.
### 2. Model Training:

- Models were trained using a PyTorch-based custom pipeline.
- Hyperparameter optimization was performed using Optuna for models like Transformer and MLP.
## Evaluation:

- Models were tested on a held-out test set.
- Robustness was assessed through adversarial testing with similar character replacements, casing and symbol addition and random Unicode character replacement.

## Observations
- Deep learning models (Transformer, LSTM) showed high performance and resilience to adversarial attacks.
- Simple models like Logistic Regression and KNN struggled under adversarial conditions.
- Adversarial testing highlights the importance of robustness in cybersecurity applications.

## How to Run
### 1. Clone the Repository
```bash
git clone https://github.com/Amrtamer711/AI-Cybersecurity-Project.git
cd AI-Cybersecurity-Project
```
### 2. Optional: Run Tokenization
Run ```Tokenization.ipynb``` to perform tokenization training
WARNING: This is optional as it has already been run and the vocabuary has been stored in the ```Tokenizer``` folder. It is advised not to re-run it because it will take a very long time.

### 4. Perform Dataset Preparation 
Run ```dataset_prep.ipynb``` to prepare dataset as this will normalize and prepare metadata, produce text examples in the form of <URL>{url} <TITLE>{title}, tokenize the examples and create the advarsarial dataset.

### 5. Run all models
Run all the notebooks labeled by their corresponding model name (eg.: ```Tokenization.ipynb```) and this will perform training and testing for both original and advarsarial examples.

# Report is available for more detailed breakdown of project.
