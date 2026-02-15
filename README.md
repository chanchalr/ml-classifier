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
