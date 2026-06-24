# Predicting Corporate Bankruptcy — Traditional ML vs. Deep Learning

Summative project for the Introduction to Machine Learning module. The notebook frames corporate
bankruptcy as a **severely imbalanced** binary classification problem (3.23% positive) and compares
**traditional machine learning** (scikit-learn) against **deep learning** (TensorFlow), using
**ROC-AUC / PR-AUC as decision-driving diagnostics** rather than end-of-notebook plots.

## Dataset

Taiwan Economic Journal bankruptcy dataset — 6,819 listed companies, 95 financial-ratio features, binary
`Bankrupt?` target, no missing values. Not from `sklearn` or `keras` (an assignment requirement).

> D. Liang, C.-C. Lu, C.-F. Tsai, and G.-A. Shih, "Financial ratios and corporate governance indicators in
> bankruptcy prediction: A comprehensive study," *European Journal of Operational Research*, vol. 252,
> no. 2, pp. 561–572, 2016. Source: Taiwan Economic Journal via the UCI ML Repository / Kaggle.

## Files

| File | Purpose |
|------|---------|
| `summative.ipynb` | End-to-end notebook: EDA → preprocessing → ML → DL → experiments → error analysis |
| `data.csv` | The dataset (6,819 × 96) |
| `requirements.txt` | Pinned dependencies matching a Colab GPU runtime |

## How to run (Google Colab — recommended)

1. Open `summative.ipynb` in Colab and set **Runtime → Change runtime type → GPU**. or run it locally
2. Place `data.csv` where the notebook can find it — either upload it when prompted, or copy it to
   `MyDrive/bankruptcy_summative/data.csv` on your Drive.
3. Run **Runtime → Restart and run all**. The first cell mounts Google Drive (for checkpoints) and installs
   the few extra libraries.

### Disconnect resilience

Deep-learning training checkpoints to Google Drive **after every epoch** (`BackupAndRestore` + per-epoch and
best-weight `ModelCheckpoint` + `CSVLogger`). If the Colab runtime disconnects, simply re-run the training
cell — it resumes from the last completed epoch rather than starting over.

## How to run (local)

```bash
pip install -r requirements.txt
jupyter notebook summative.ipynb   # set DATA_PATH to ./data.csv
```

The Drive-mount cell is skipped automatically off Colab; checkpoints are written under `./bankruptcy_summative`.

## Method highlights

- **Leakage-free preprocessing:** stratified split first, then winsorization (1st–99th pct), `RobustScaler`,
  and variance + correlation pruning — all fit on the training split only.
- **Imbalance handled two ways and compared:** class-weighting vs. SMOTE, the winner chosen by validation
  PR-AUC per model.
- **ML:** Logistic Regression, Random Forest, XGBoost, SVM-RBF, tuned with stratified CV scored on PR-AUC.
- **DL:** TensorFlow `tf.data` pipeline, a **Sequential** MLP and a branched **Functional** model, with
  PR-AUC driving early stopping, the LR schedule, and checkpointing.
- **Evaluation:** confusion matrices, ROC and PR curves, a sorted comparison table, threshold tuning to a
  recall target, and permutation-importance error analysis centred on the costly false-negative case.
