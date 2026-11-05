# Dry Bean Variety Classification — Machine Learning Assignment 2

> **Student work note:** This is a project starter. Run the training script yourself in BITS Virtual Lab, inspect the generated metrics, write your own observations in the report, and add your own GitHub/Streamlit links and BITS Lab screenshot. Do not submit unverified or placeholder results.

## a. Problem statement

The objective is to build and compare five supervised classification models that predict the variety of a dry bean from 16 numerical geometric measurements extracted from bean images. This is a seven-class classification task.

## b. Dataset description

- **Dataset:** Dry Bean Dataset
- **Source:** UCI Machine Learning Repository, dataset ID 602
- **Instances:** 13,611
- **Input features:** 16 numerical features
- **Target:** `Class`
- **Classes:** Seker, Barbunya, Bombay, Cali, Dermason, Horoz and Sira
- **Missing values:** The UCI description reports no missing values.
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0)

The 16 measurements describe bean size and shape, including area, perimeter, major/minor axis lengths, aspect ratio, eccentricity, convex area, equivalent diameter, extent, solidity, roundness, compactness and four shape factors.

Official source: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset  
Dataset DOI: https://doi.org/10.24432/C50S4B

## c. GitHub Repository Link

**Add your own repository URL here after creating and pushing this project:**

`https://github.com/<your-username>/<your-repository>`

## d. Models used and evaluation

The five required classifiers are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The training script uses an 80:20 stratified train/test split with `random_state=42`. Scaling is included in pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and SVM. The Decision Tree is trained without feature scaling. Each model is fitted only on the training partition.

AUC is calculated as **macro-averaged one-vs-rest (OVR)** because the target has seven classes. Precision, recall and F1 are macro averages so each class contributes equally to those summary metrics.

### Evaluation comparison table

Run `python train_models.py`. The script writes the actual results to `model/metrics.json`. Copy the six values for each model into this table after running the experiment.

| ML Model | Accuracy | AUC (macro OVR) | Precision (macro) | Recall (macro) | F1 (macro) | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Fill from run | Fill from run | Fill from run | Fill from run | Fill from run | Fill from run |
| Decision Tree | Fill from run | Fill from run | Fill from run | Fill from run | Fill from run | Fill from run |
| KNN | Fill from run | Fill from run | Fill from run | Fill from run | Fill from run | Fill from run |
| Gaussian Naive Bayes | Fill from run | Fill from run | Fill from run | Fill from run | Fill from run | Fill from run |
| Support Vector Machine | Fill from run | Fill from run | Fill from run | Fill from run | Fill from run | Fill from run |

### Observations on model performance

Complete these observations only after examining your actual test results:

| Model | Observation to write after running |
|---|---|
| Logistic Regression | Discuss its test accuracy, macro AUC and whether its errors are concentrated in particular bean classes. |
| Decision Tree | Discuss its performance and whether the tree appears to generalise as well as the other models. |
| KNN | Discuss how the scaled feature distances affect its predictions and class-wise results. |
| Gaussian Naive Bayes | Discuss its results in light of the model's conditional-independence assumption. |
| Support Vector Machine | Discuss its performance relative to the other models and its class-wise strengths/weaknesses. |
| Overall winner | Name the model based on your chosen primary metric, and use the other metrics to explain the decision. |

Do not assume the winner before running the experiment. If two models are close, explain the trade-off rather than claiming one is best in every metric.

## e. Streamlit application

The application (`app.py`) includes:
- CSV upload for labelled test data
- A model-selection dropdown
- Accuracy, AUC, precision, recall, F1 and MCC
- Confusion matrix
- Classification report
- Side-by-side comparison of all five models
- Downloadable predictions

The app loads saved estimators and does **not** retrain models on uploaded test data. The uploaded CSV must include the 16 feature columns and the `Class` target column.

## f. Methodology

1. Load the official UCI dataset using `ucimlrepo`.
2. Separate the 16 input columns from the `Class` target.
3. Encode the seven target labels.
4. Split the data into 80% training and 20% test partitions using stratification.
5. Fit the five classifiers on the training data only.
6. Evaluate on the same held-out test partition.
7. Save the fitted estimators, label encoder, metrics and metadata.
8. Use the saved estimators in Streamlit to evaluate an uploaded labelled test CSV.

## g. How to run

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

If the UCI fetch fails in your lab, download the original dataset from the UCI page and use the `ucimlrepo` fetch method when internet access is restored. Do not use test data to fit or tune the models.

## h. GitHub and Streamlit deployment

1. Create a new GitHub repository under your own account.
2. Upload this project folder, including the generated `model/` files and `test_data.csv`.
3. Commit your own changes with meaningful commit messages.
4. Open https://share.streamlit.io/ and connect your GitHub account.
5. Create a new app, select your repository and branch, and set `app.py` as the entry point.
6. Deploy and test the live URL in a private/incognito browser window.

**GitHub URL:** Add your actual URL here.  
**Live Streamlit URL:** Add your actual URL here.  
**BITS Virtual Lab screenshot:** Insert your own screenshot in the final PDF.

## i. References

Koklu, M. and Ozkan, I.A. (2020). *Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques*. Computers and Electronics in Agriculture, 174, 105507. Dataset hosted by the UCI Machine Learning Repository: https://doi.org/10.24432/C50S4B
