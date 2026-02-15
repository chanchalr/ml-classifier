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

## Results (test dataset)

Models are trained on the **training split** of `dataset/bank_full.csv` (80%) via `generate_models.py`; metrics below are on the held-out **test split** (20%, saved as `test.csv`), using the saved `.pkl` models and `run_model()`.

| ML Model Name       | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|---------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression | 0.8988   | 0.9046| 0.8838    | 0.8988 | 0.8846| 0.4250|
| Decision Tree       | 0.8764   | 0.7162| 0.8782    | 0.8764 | 0.8773| 0.4260|
| K-Nearest Neighbor  | 0.8933   | 0.8296| 0.8766    | 0.8933 | 0.8795| 0.3973|
| Naive Bayes         | 0.8569   | 0.8151| 0.8652    | 0.8569 | 0.8608| 0.3639|
| Random Forest       | 0.9049   | 0.9273| 0.8934    | 0.9049 | 0.8951| 0.4810|
| XGBoost             | 0.9069   | 0.9297| 0.8992    | 0.9069 | 0.9017| 0.5195|

### Observations on model performance

| ML Model Name       | Observation about model performance | Possible reasons |
|---------------------|--------------------------------------|------------------|
| Logistic Regression | Strong overall: high accuracy and AUC with balanced precision–recall. MCC (0.43) is moderate. Interpretable and stable baseline. | Linear decision boundary overall fits the data fairly reasonably; L2 regularisation limits overfitting; moderate MCC possibly due to imbalanced classes (minority “yes” is harder to predict well). |
| Decision Tree       | Lowest AUC (0.72) on test; probability rankings are weak. Accuracy and MCC are reasonable but less reliable for ranking or threshold tuning. | possibly estimates are poorly calibrated; single tree prone to overfitting on training splits. |
| K-Nearest Neighbor  | Good accuracy and recall; AUC is mid-range. Lower MCC suggests sensitivity to class imbalance. | Majority class in the neighbourhood can dominate the vote; choice of k and distance metric affects bias–variance |
| Naive Bayes         | Lowest accuracy and MCC; fast to train but weaker discriminative performance. | Strong independence assumption between features is violated (e.g. job, education, balance are related); Gaussian assumption may not match feature distributions; prior dominated by majority class. |
| Random Forest       | High accuracy and second-best AUC (0.93); MCC (0.48) indicates good performance on the minority class. Robust. | Averaging many trees reduces variance and overfitting; bootstrap + feature subsampling increases diversity; ensemble improves metrics over single tree. |
| XGBoost             | Best overall on test: top accuracy (0.91), highest AUC (0.93) and MCC (0.52). Best balance of precision, recall and ranking | Sequential boosting corrects errors ; regularisation controls overfitting; handles mixed and non-linear relationships and imbalance effectively. |
