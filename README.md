# Spam Detection with Machine Learning

This repository contains a complete workflow for detecting phishing (spam) emails using multiple machine learning methods. The project demonstrates how to preprocess text data, extract relevant features, train various classification models, and evaluate their performance in classifying emails as "Safe Email" or "Phishing Email."

## Table of Contents
1. [Introduction](#introduction)  
2. [Dataset Overview](#dataset-overview)  
3. [Data Preprocessing](#data-preprocessing)  
4. [Text Preprocessing](#text-preprocessing)  
5. [Data Visualization](#data-visualization)  
6. [Feature Extraction](#feature-extraction)  
7. [Model Building and Evaluation](#model-building-and-evaluation)  
8. [Results and Comparison](#results-and-comparison)  
9. [How to Use](#how-to-use)  
10. [Contributing](#contributing)  
11. [License](#license)  

---

## Introduction
The primary goal of this project is to automatically identify phishing emails by leveraging various machine learning algorithms. Phishing emails often exhibit specific patterns in text that differentiate them from legitimate emails, and this project illustrates how to exploit these patterns for accurate classification.

---

## Dataset Overview
- **Source**: [Phishing Emails Dataset from Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails)
- **Structure**: 
  - **Rows**: 18,650 (reduced to 17,538 after removing duplicates)
  - **Columns**: 3
    1. `Unnamed: 0`: Index column (removed during preprocessing)
    2. `Email Text`: Contains the email message
    3. `Email Type`: Label indicating "Safe Email" or "Phishing Email"

---

## Data Preprocessing
1. **Missing Values**: Verified none were present with `df.isnull().sum()`.  
2. **Dropping Unnecessary Columns**: Removed the `Unnamed: 0` index column.  
3. **Removing Duplicates**: Used `df.drop_duplicates()` to reduce rows from 18,650 to 17,538.  
4. **Label Encoding**: Encoded email labels (“Safe Email” → 1, “Phishing Email” → 0) using `LabelEncoder`.

---

## Text Preprocessing
- **Cleaning**: Defined a `preprocess_text()` function to:
  - Convert text to lowercase.
  - Remove hyperlinks and punctuation.
  - Remove extra whitespace.
- This step standardizes text data so that features are more meaningful.

---

## Data Visualization
1. **Word Cloud** (including stopwords)  
2. **Word Cloud** (excluding stopwords)  

These visualizations help identify the most frequent terms in both safe and phishing emails, providing insights into recurring patterns and vocabulary usage.

---

## Feature Extraction
- Utilized **TfidfVectorizer** to convert cleaned text data into numerical features suitable for machine learning algorithms. TF-IDF captures the importance of words in the context of entire emails.

---

## Model Building and Evaluation
- **Train-Test Split**: 80% training, 20% testing
- **Metrics Used**: Accuracy, Precision, Recall, F1-score
- Models:
  1. **Logistic Regression**  
     - Accuracy: 97.95%  
     - Precision: 98%  
     - Recall: 98%  
     - F1-score: 98.34%  

  2. **Decision Tree Classifier**  
     - Accuracy: 93.19%  
     - Precision: 93%  
     - Recall: 93%  
     - F1-score: 94.40%  

  3. **Random Forest Classifier**  
     - Accuracy: 97.72%  
     - Precision: 97%  
     - Recall: 98%  
     - F1-score: 98.14%  

  4. **MLP Classifier** (Multi-Layer Perceptron)  
     - Accuracy: 98.43%  
     - Precision: 98%  
     - Recall: 98%  
     - F1-score: 98.73%  

  5. **SGD Classifier** (Stochastic Gradient Descent)  
     - Accuracy: 98.60%  
     - Precision: 98%  
     - Recall: 99%  
     - F1-score: 98.73%  

---

## Results and Comparison
- **Logistic Regression** exhibits strong performance with 97.95% accuracy and 98.34% F1-score.  
- **Decision Tree** lags behind with the lowest accuracy of 93.19% and an F1-score of 94.40%.  
- **Random Forest** shows improved performance over the Decision Tree, attaining 97.72% accuracy.  
- **MLP Classifier** achieves a strong balance of accuracy (98.43%) and F1-score (98.73%).  
- **SGD Classifier** yields the highest accuracy of 98.60%, tying with MLP for the highest F1-score (98.73%).  

Hence, the SGD Classifier and MLP Classifier demonstrate superior performance in modeling and classifying emails correctly.

---

## How to Use
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/Spam-Detection-ML.git
