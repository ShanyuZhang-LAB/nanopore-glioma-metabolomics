# Nanopore-Based Glioma Metabolomics Analysis Code

This repository provides the Python code and data organization for machine-learning analysis of nanopore event features and classification of patient-sample events. The two scripts can be run independently.

## Project structure

```text
Nanopore-Based Glioma Metabolomics Analysis Code/
├── 1_Training_Feature_Matrices/
│   ├── D-2-HG.xlsx
│   └── Others.xlsx
├── 2_Machine_Learning_Training/
│   └── Machine_Learning_Training.py
├── 3_Machine_Learning_Training_results/
├── 4_Trained_Models.joblib
├── 5_Patient_Event_Feature_Matrices/
│   ├── Patient_1_T.xlsx               # Example name only
│   └── ...
├── 6_Clinical_Event_Classification/
│   ├── Clinical_Event_Classification.py
│   └── results/                        # Created automatically
├── README.md
└── requirements.txt
```

The file `4_Trained_Models.joblib` is the unchanged original archived model, stored directly in the project root. The patient filenames shown above are illustrative; actual patient files are not included unless separately provided. Empty output directories are created by the scripts. Both scripts locate the project directory from their own file location, so no absolute path configuration is required. If you rename the scripts, their functionality is unchanged, provided they remain in the indicated directories.

## Environment and installation

Python 3.10 or 3.11 is recommended. The archived model was saved with scikit-learn 1.4.2; use the pinned version in `requirements.txt` for consistent model loading.

From the project root, run:

```bash
python -m pip install -r requirements.txt
```

Only load `.joblib` files obtained from trusted sources: loading a Python-serialized model may execute code.

## 1. Machine-learning training and evaluation

**Script:** `2_Machine_Learning_Training/Machine_Learning_Training.py`

**Inputs:** Exactly one `D-2-HG` and one `Others` training matrix, each in CSV, XLS, or XLSX format, in `1_Training_Feature_Matrices`. The provided files are Excel workbooks. The script uses three numerical features:

| Feature column | Meaning |
|---|---|
| `Relative_Blockade` | Relative ionic-current blockade |
| `Current_SD` | Event current standard deviation |
| `Current_Range` | Event current range (original amplitude feature) |

If `IsNoise` is present, rows with `IsNoise == -1` are excluded. The class is determined by the source training file, not by a column in the workbook.

**Analysis:** The script compares nine algorithms: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Gaussian Naive Bayes, Linear Discriminant Analysis (LDA), and Multilayer Perceptron (MLP). It uses a stratified 80% training / 20% holdout split (`random_state=520`) and stratified 10-fold cross-validation within the 80% training subset (`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` for the supplied dataset). The implementation caps the fold count at the number of samples in the smallest training class if fewer than 10 are available. The best algorithm is selected by mean cross-validation accuracy. For the supplied dataset, the expected best algorithm is Random Forest with 100 trees (`random_state=42`). The script deliberately stops if a different algorithm is selected, rather than silently generating figures for an unexpected model.

**Run from the project root:**

```bash
python 2_Machine_Learning_Training/Machine_Learning_Training.py
```

Alternatively, open this script in PyCharm and select **Run**.

**Saved outputs** (in `3_Machine_Learning_Training_results/`):

| File | Content |
|---|---|
| `model_training_accuracy.csv` | Cross-validation mean, standard deviation, and per-fold accuracy for all nine models |
| `confusion_matrix_Random_Forest_holdout_set.png` | Random Forest confusion matrix on the independent 20% holdout set, normalized by true class and shown as percentages |
| `confusion_matrix_Random_Forest_holdout_set.csv` | Underlying row-normalized confusion matrix values (fractions between 0 and 1) |
| `feature_importance_Random_Forest.png` | Impurity-based feature importance from the Random Forest fitted on the 80% training subset |
| `feature_importance_data_Random_Forest.csv` | Numerical feature importance values |
| `learning_curve_Random_Forest.png` | Training and validation accuracy learning curve |
| `learning_curve_data_Random_Forest.csv` | Numerical learning-curve data, including standard deviations |

The learning curve intentionally follows the original analysis settings: **all standard-sample events**, **5-fold cross-validation** (`learning_curve(..., cv=5)`, using scikit-learn's default stratified folds without shuffling for classification), **10 training-size fractions from 0.1 to 1.0**, and accuracy scoring (`n_jobs=-1`). Thus, its cross-validation design is distinct from the 10-fold model-selection procedure and its curves should not be interpreted as holdout-test performance.

The holdout confusion matrix is computed only after fitting the selected Random Forest to the 80% training subset. The training script **does not overwrite or update** `4_Trained_Models.joblib`. Its outputs are evaluation tables and figures; it does not save a newly fitted model.

## 2. Classification of patient nanopore events

**Script:** `6_Clinical_Event_Classification/Clinical_Event_Classification.py`

**Inputs:**

- The unchanged original pretrained model at `4_Trained_Models.joblib`.
- Patient event feature matrices in `5_Patient_Event_Feature_Matrices/` (CSV, XLS, or XLSX). Multiple files are processed separately. File names are retained in the output summaries.
- Each patient matrix must contain `Relative_Blockade`, `Current_SD`, and `Current_Range`. An optional `IsNoise` column excludes rows with value `-1` from prediction.

**Run from the project root:**

```bash
python 6_Clinical_Event_Classification/Clinical_Event_Classification.py
```

Alternatively, open this script in PyCharm and select **Run**.

**Model compatibility:** The archived model retains its original serialized feature names and class labels. The classification script maps the three input feature columns (`Relative_Blockade`, `Current_SD`, and `Current_Range`) to the corresponding archived feature names in the required order. Model outputs are reported as **D-2-HG** and **Others**. The archived model and its metadata are not modified.

**Classification:** For each analyzed event, the script records the model-predicted class, probability assigned to D-2-HG, and maximum predicted class probability. With a confidence threshold of **0.90**, predictions below this threshold are labeled `Unclassified`. Rows excluded by `IsNoise == -1` are labeled `Excluded_Noise` and are not counted as analyzed events. A missing `IsNoise` column means all rows are eligible for prediction.

**Saved outputs** (in `6_Clinical_Event_Classification/results/`):

- `<patient_file_stem>_classified.csv` for each patient matrix, retaining original event columns and adding `Predicted_Class`, `D-2-HG_Probability`, `Prediction_Confidence`, and `Confidence_Filtered_Class`.
- `patient_event_summary.csv`, with input filename, total and analyzed event counts, excluded noise events, D-2-HG / Others / Unclassified counts, D-2-HG fraction among analyzed events, and confidence threshold.

The D-2-HG event fraction is a **classification-based event fraction**, not an absolute metabolite concentration or an estimate of clinical diagnostic sensitivity/specificity. If T/P/N region labels are encoded in filenames, they remain traceable through the `Sample_File` field; the script does not infer patient identity or tissue region from filename conventions.

## Model file location

The pretrained model is a **file** named `4_Trained_Models.joblib` in the project root, **not** a directory. The clinical classification script loads it from:

```python
model_path = root / "4_Trained_Models.joblib"
```

No model conversion, modification of internal feature names, retraining, or reserialization is needed. The training script does not read or overwrite this archived model.

## Reproducibility and integrity

- The pretrained model is an archived artifact used **only for inference** in the clinical script. It is not retrained, reserialized, or overwritten.
- The clinical script checks the model file's SHA-256 hash before and after classification to detect changes during the run.
- To verify that a renamed copy is byte-for-byte identical to the original `best_final_model.joblib`, compare their SHA-256 hashes externally; renaming alone does not alter the model.
- Training results may vary if the input data, row order, dependency versions, or computing environment differ from those used to produce the original results. Preserve the provided matrices and the pinned environment for reproduction.
- This repository contains analytical code; it does not itself establish clinical validation or intraoperative deployment.

## Troubleshooting

**Training files not found:** Check that `1_Training_Feature_Matrices` is directly inside the project root and contains `D-2-HG.xlsx` and `Others.xlsx` (or the corresponding supported CSV/XLS formats). Do not nest another project folder inside it.

**Missing feature columns:** Verify the three feature columns exactly as listed above. Renaming features inside the original `.joblib` is not required.

**Model loading warnings:** Install `scikit-learn==1.4.2` from `requirements.txt`. Do not load an untrusted `.joblib` file.

**Old results still present:** The scripts write their specified outputs but do not delete unrelated files from previous runs. Remove obsolete output files manually if a clean result folder is needed.
