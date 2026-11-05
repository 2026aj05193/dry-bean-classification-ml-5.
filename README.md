# Dry Bean Variety Classification

## 1. Introduction

This project is part of Machine Learning Assignment 2. The aim is to predict the variety of a dry bean using its physical measurements. Five classification algorithms were trained and tested, and their performance was compared using different evaluation metrics.

## 2. Dataset

The Dry Bean dataset was taken from the UCI Machine Learning Repository.

- **Number of records:** 13,611
- **Number of input features:** 16
- **Target column:** `Class`
- **Number of classes:** 7
- **Missing values:** None reported

The dataset contains measurements such as area, perimeter, major axis length, minor axis length, eccentricity, compactness and other shape-related features.

The seven bean varieties are Seker, Barbunya, Bombay, Cali, Dermason, Horoz and Sira.

Dataset link: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

## 3. Classification Algorithms

The following five algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The dataset was divided into 80% training data and 20% testing data. A stratified split with `random_state=42` was used so that the class distribution remained similar in both sets.

Feature scaling was applied to Logistic Regression, KNN, Gaussian Naive Bayes and SVM. The Decision Tree was trained without scaling.

## 4. Model Evaluation

All five models were evaluated using the same test data.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9948 | 0.9349 | 0.9314 | 0.9329 | 0.9042 |
| Decision Tree | 0.8917 | 0.9448 | 0.9084 | 0.9088 | 0.9085 | 0.8691 |
| KNN | 0.9137 | 0.9858 | 0.9290 | 0.9245 | 0.9265 | 0.8956 |
| Gaussian Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Support Vector Machine | **0.9221** | **0.9949** | **0.9359** | **0.9330** | **0.9343** | **0.9059** |

AUC was calculated using macro-averaged one-vs-rest (OVR). Precision, recall and F1 score use macro averaging.

### Results

SVM gave the highest accuracy of 92.21% and also achieved the highest scores for AUC, precision, recall, F1 and MCC. Logistic Regression was a close second.

The Decision Tree had the lowest accuracy among the five models. KNN also performed well, while Gaussian Naive Bayes achieved a high AUC but lower accuracy.

Based on these results, SVM was the best-performing model for this experiment.

## 5. Streamlit Application

A Streamlit application was developed to test the saved models.

The application allows users to:

- Upload a labelled CSV test file.
- Select one of the five trained models.
- View accuracy, AUC, precision, recall, F1 score and MCC.
- View the confusion matrix and classification report.
- Compare the performance of all five models.
- Download the prediction results.

The application uses the previously saved models. It does not train the models again when a test CSV file is uploaded.

The uploaded CSV should contain all 16 input features and the `Class` column.

**Live application:**  
https://dry-bean-classification-ml-5-123.streamlit.app/

## 6. Project Files

- `app.py` – Streamlit application.
- `train_models.py` – Script used to train and evaluate the models.
- `requirements.txt` – Required Python packages.
- `test_data.csv` – Test dataset used for evaluation.
- `model/` – Contains the saved models, label encoder, metrics and metadata.
- `data/` – Contains the dataset used for training.

## 7. How to Run the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python train_models.py
```

Start the Streamlit application:

```bash
streamlit run app.py
```

## 8. Repository

GitHub:  
https://github.com/2026aj05193/dry-bean-classification-ml-5.

## 9. Reference

Koklu, M. and Ozkan, I.A. (2020). Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques. *Computers and Electronics in Agriculture*, 174, 105507.

UCI Machine Learning Repository – Dry Bean Dataset:  
https://doi.org/10.24432/C50S4B
