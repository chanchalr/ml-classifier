# ml-classifier

## Problem statement

The goal is to **predict whether a bank client will subscribe to a term deposit** (variable `y`: yes/no) after a marketing campaign (e.g. phone calls). This is a **binary classification** problem: given client and campaign attributes, the model outputs the likelihood of subscription so the bank can prioritise follow-up and improve campaign efficiency.

Models are evaluated with accuracy, precision, recall, F1, AUC-ROC, and Matthews Correlation Coefficient (MCC).

---

## Dataset description

The project uses the **Bank Marketing** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing). It comes from a Portuguese retail bank and describes outcomes of phone-based marketing campaigns for a term deposit product.

- **Source:** UCI ML Repository (Dataset 222)  
- **Format:** CSV (semicolon-separated in the original; comma-separated in the provided batches)  
- **Target:** `y` — whether the client subscribed to a term deposit (`yes` / `no`)  
- **Size:** ~45,211 records in the full dataset; the repo also provides batched files (e.g. ~9,000 rows per batch in `dataset/`).

### Input features (summary)

| Feature      | Description |
|-------------|-------------|
| `age`       | Client age |
| `job`       | Job type (e.g. management, technician, blue-collar) |
| `marital`   | Marital status |
| `education` | Education level |
| `default`   | Has credit in default? (yes/no) |
| `balance`   | Average yearly balance (euros) |
| `housing`   | Has housing loan? (yes/no) |
| `loan`      | Has personal loan? (yes/no) |
| `contact`   | Contact communication type |
| `day`       | Last contact day of month |
| `month`     | Last contact month |
| `duration`  | Last contact duration (seconds) |
| `campaign`  | Number of contacts during this campaign |
| `pdays`     | Days since previous campaign contact (-1 if not contacted) |
| `previous`  | Number of contacts before this campaign |
| `poutcome`  | Outcome of previous marketing campaign |

The full dataset can be downloaded and split into batches using the `prepare_data.py` script.

---

## Results (full dataset)

Metrics from running all models on the full Bank Marketing dataset (train/test split 80/20, default pipeline).

| ML Model Name       | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|---------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression | 0.8988   | 0.9046| 0.8838    | 0.8988 | 0.8846| 0.4250|
| Decision Tree       | 0.8752   | 0.7107| 0.8764    | 0.8752 | 0.8758| 0.4174|
| K-Nearest Neighbor  | 0.8933   | 0.8296| 0.8766    | 0.8933 | 0.8795| 0.3973|
| Naive Bayes         | 0.8569   | 0.8151| 0.8652    | 0.8569 | 0.8608| 0.3639|
| Random Forest       | 0.9037   | 0.9261| 0.8914    | 0.9037 | 0.8930| 0.4697|
| XGBoost             | 0.9069   | 0.9297| 0.8992    | 0.9069 | 0.9017| 0.5195|

### Observations on model performance

| ML Model Name       | Observation about model performance | Possible reasons |
|---------------------|--------------------------------------|------------------|
| Logistic Regression | Strong overall: high accuracy and AUC with balanced precision–recall. MCC (0.43) is moderate. Interpretable and stable baseline. | Linear decision boundary fits the separable structure in the data; L2 regularisation limits overfitting; moderate MCC possibly due to imbalanced classes (minority “yes” is harder to predict well). |
| Decision Tree       | Lowest AUC (0.71); probability rankings are weak. Accuracy is reasonable but less reliable for ranking or threshold tuning. | Tree outputs are step-like, so probability estimates are poorly calibrated; single tree possibly overfits to training splits |
| K-Nearest Neighbor  | Good accuracy and recall; AUC is mid-range. Lower MCC suggests sensitivity to class imbalance. | Majority class in the neighbourhood can dominate the vote; choice of k and distance metric affects bias–variance |
| Naive Bayes         | Lowest accuracy and MCC; fast to train but weaker discriminative performance. | Strong independence assumption between features is violated (e.g. job, education, balance are related); Gaussian assumption may not match feature distributions; prior dominated by majority class. |
| Random Forest       | High accuracy and second-best AUC; MCC (0.47) indicates good performance on the minority class. Robust. | Averaging many trees reduces variance and overfitting; bootstrap + feature subsampling increases diversity; ensemble probabilities are smoother than a single tree. |
| XGBoost             | Best overall: top accuracy, highest AUC (0.93) and MCC (0.52). Best balance of precision, recall and ranking. | Sequential boosting corrects errors ; regularisation controls overfitting; handles mixed and non-linear relationships and imbalance effectively. |
