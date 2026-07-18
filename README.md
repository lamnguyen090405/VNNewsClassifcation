# Vietnamese News Classification System

This repository builds an automated Vietnamese news classification system categorizing articles into **17 distinct news topics**. The system utilizes the pre-trained language model **PhoBERT** (`vinai/phobert-base`) to extract deep contextual embeddings from both the title and the content, combined with a feed-forward neural network **Multi-Layer Perceptron (MLP)** for classification. Additionally, the project provides an interactive web interface powered by **FastAPI** to facilitate real-time testing and text analytics.

## Table of Contents
- [Vietnamese News Classification System](#vietnamese-news-classification-system)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Key Features](#key-features)
  - [Directory Structure](#directory-structure)
  - [List of 17 Classification Categories](#list-of-17-classification-categories)
  - [Workflow (Approach)](#workflow-approach)
  - [Model Architecture](#model-architecture)
  - [Installation \& Usage](#installation--usage)
  - [Training \& Evaluation Results](#training--evaluation-results)

## Introduction
Automated news classification is a critical component for organizing, searching, and recommending articles on large-scale news platforms. This project addresses the multi-class text classification problem specifically for Vietnamese news articles. Instead of traditional representation methods (e.g., TF-IDF) combined with shallow classifiers (e.g., SVM/Random Forest), this approach harnesses **PhoBERT**'s deep semantic understanding to achieve superior classification performance.

## Key Features
- **Sophisticated Vietnamese Text Preprocessing**: NFC normalization, stripping of HTML tags, emails, links, special characters, numbers, word segmentation using the `underthesea` library, and stopword removal utilizing a curated `vietnamese-stopwords.txt` dictionary.
- **PhoBERT Feature Extraction**: Leveraging PhoBERT-base to generate rich semantic embeddings from the title (which acts as a summarized representation) and the main content.
- **MLP Classifier Training**: A Multi-Layer Perceptron neural network trained to categorize the combined feature representations into 17 distinct classes.
- **Real-time Visualization Dashboard**:
  - Predicts the main topic along with a confidence score.
  - Interactive bar charts representing top-3 predicted probabilities and keyword frequencies using Chart.js.
  - A correlation matrix heatmap displaying the co-occurrence frequency of the top-5 keywords within sentences.
  - Keyword highlighting directly in the processed text.
  - Text metrics including total word count, sentence count, and average sentence length.

## Directory Structure
```
├── Data/                       # Dataset directory (downloaded from Google Drive)
├── models/                     # Trained models and encoder
│   ├── label_encoder.pkl       # Topic label encoder
│   └── news_classifier_mlp.pkl # Trained MLP classifier model
├── output/                     # EDA and evaluation visualization charts
├── pipelines/                  # Extracted feature arrays (.npy) for rapid re-training
│   ├── X_features.npy          # Stacked feature array
│   └── y_labels.npy            # Encoded target labels
├── src/                        # Jupyter Notebooks for research and experimentation
│   ├── EDA.ipynb               # Exploratory Data Analysis
│   ├── Pre-processing.ipynb    # Data cleaning and word segmentation
│   ├── Feature Engneering.ipynb# Feature extraction using PhoBERT
│   └── Train.ipynb             # MLP model training and evaluation
├── templates/                  # Frontend UI files
│   ├── index.html              # Main web application interface
│   └── styles/
│       └── style.css           # CSS stylesheet for UI
├── app.py                      # FastAPI application backend server
├── requirements.txt            # List of dependencies
└── vietnamese-stopwords.txt    # Vietnamese stopwords dictionary
```

## List of 17 Classification Categories
The model classifies Vietnamese news articles into 17 diverse topics:
1. Bất động sản (Real Estate)
2. Công đoàn (Labor Union)
3. Du lịch (Tourism)
4. Gia đình (Family)
5. Giáo dục (Education)
6. Giải trí (Entertainment)
7. Khoa học công nghệ (Science & Technology)
8. Kinh doanh (Business)
9. Media (Media)
10. Pháp luật (Law)
11. Sức khỏe (Health)
12. Thế giới (World News)
13. Thể thao (Sports)
14. Thời sự (Current Affairs)
15. Xe (Vehicles)
16. Xã hội (Society)
17. Đời sống (Lifestyle)

## Workflow (Approach)
1. **Exploratory Data Analysis (EDA)** [src/EDA.ipynb]:
   - Analysed class distribution to identify data imbalance.
   - Evaluated title and content text lengths to decide optimal sequence cutoff limits for PhoBERT.
2. **Data Preprocessing** [src/Pre-processing.ipynb]:
   - Cleaned raw text by removing HTML tags, emails, links, punctuation, and numeric digits.
   - Performed word segmentation using `underthesea.word_tokenize` to properly capture compound words (e.g., `học_sinh`, `công_nghệ`).
   - Removed common Vietnamese stopwords to retain core semantic keywords.
3. **Feature Engineering** [src/Feature Engneering.ipynb]:
   - Tokenized text inputs using the PhoBERT tokenizer.
   - Extracted embedding representations from PhoBERT (`vinai/phobert-base`) by computing the mean pooling of the last hidden states.
   - Concatenated the title embeddings (64 dimensions), content embeddings (256 dimensions), and total content token length (1 dimension) to form the final feature vector.
4. **Model Training** [src/Train.ipynb]:
   - Encoded the 17 categorical target labels using `LabelEncoder`.
   - Trained a Multi-Layer Perceptron (MLP) Classifier for multi-class prediction.
   - Evaluated model accuracy and generated a confusion matrix on the test set.

## Model Architecture
- **Embedding Layer**: PhoBERT (Base) - A state-of-the-art RoBERTa-based model optimized for Vietnamese language tasks.
- **Classification Head**: An MLP Classifier accepting the stacked input vector $v = [v_{title} \parallel v_{content} \parallel v_{length}]$ with a total size of $64 + 256 + 1 = 321$ dimensions, processed through hidden layer(s) with ReLU activation, and a final Softmax output layer providing probability distribution over the 17 classes.

## Installation & Usage
### 1. Environment Setup
Python 3.8+ is required. It is highly recommended to set up a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate # On Linux/macOS
```

### 2. Install Dependencies
Install all required packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Obtain Dataset
Download the preprocessed dataset from [Google Drive](https://drive.google.com/drive/folders/1wVbwVoVhSVCKcp_w7pkfmdePHkRSeu3o?usp=sharing) and place the CSV files inside the `Data/` folder.

### 4. Run the Web Application
Start the FastAPI server via uvicorn:
```bash
uvicorn app:app --reload
```
Navigate to [http://localhost:8000](http://localhost:8000) in your web browser to access the dashboard and classify custom articles.

## Training & Evaluation Results
The experimental results are visualized in the following plots:

*   **Class Distribution & Label Balance:**
    ![Class balance](output/balance_data.png)
*   **Data Density Pre and Post Preprocessing:**
    ![Preprocessing distribution](output/density_data.png)
*   **Most Common Words by Topic (Word Clouds):**
    ![Word Cloud](output/wordcloud_data.png)
*   **Confusion Matrix on Test Dataset:**
    ![Confusion Matrix](output/confusion_matrix_data.png)
*   **Accuracy Graph during Training Phase:**
    ![Accuracy](output/Accuracy_data.png)
