# Player Position Classification Using Decision Tree and Random Forest
### FIFA 22 Dataset · DS703 Machine Learning

A comparative study of Decision Tree and Random Forest classifiers for predicting football player positions (GK, DEF, MID, FWD) from FIFA 22 player statistics, with systematic hyperparameter tuning using GridSearchCV.

## Results Summary

| Model | Train Acc | Test Acc | Overfit Gap | MID F1 | FWD F1 |
|---|---|---|---|---|---|
| DT Baseline | 99.99% | 83.58% | 16.41% | 0.71 | 0.76 |
| DT Tuned | 88.09% | 86.72% | 1.37% | 0.77 | 0.81 |
| RF Baseline | 99.99% | 87.71% | 12.28% | 0.78 | 0.82 |
| **RF Tuned** | **92.17%** | **88.62%** | **7.55%** | **0.80** | **0.83** |

**Winner: Tuned Random Forest** — 88.62% accuracy, best per-class F1 across all positions.

---

## Dataset

- **Source:** [FIFA 22 Complete Player Dataset — Kaggle](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset)
- **Size:** 19,239 players · 6 input features
- **Target:** Player position mapped to 4 classes

| Position | Players | Share |
|---|---|---|
| DEF | 8,059 | 41.89% |
| MID | 5,368 | 27.90% |
| FWD | 3,680 | 19.13% |
| GK | 2,132 | 11.08% |

- **Split:** 80% train (15,391) · 20% test (3,848), stratified

### Features Used
| Feature | Description |
|---|---|
| `pace` | Speed and acceleration |
| `shooting` | Finishing and attacking ability |
| `passing` | Vision, crossing, passing accuracy |
| `dribbling` | Ball control and agility |
| `defending` | Tackling and defensive awareness |
| `physic` | Strength, stamina, physicality |

> GK outfield stats are missing by design — filled with 0 to retain all 2,132 goalkeeper records.

---

## Models & Hyperparameter Tuning

Both models were tuned with **GridSearchCV (5-fold cross-validation)**.

### Decision Tree
| Hyperparameter | Values Tested | Best |
|---|---|---|
| `max_depth` | 3, 5, 10, 15, 20 | 10 |
| `min_samples_split` | 2, 10, 20, 50 | 50 |
| `min_samples_leaf` | 1, 5, 10, 20 | 5 |
| `criterion` | gini, entropy | entropy |

### Random Forest
| Hyperparameter | Values Tested | Best |
|---|---|---|
| `n_estimators` | 50, 100, 200 | 200 |
| `max_depth` | 5, 10, 15, None | None |
| `max_features` | sqrt, log2 | sqrt |
| `min_samples_split` | 2, 10, 20 | 20 |

---

## Key Findings

**Random Forest wins at every stage.** The RF baseline (87.71%) already beat the tuned Decision Tree (86.72%) without any tuning — ensemble averaging provides an inherent robustness that single-tree tuning cannot replicate.

**Depth constraints fix Decision Tree overfitting.** The baseline DT had a 16.41% train/test gap. Constraining `max_depth=10` collapsed it to just 1.37%.

**MID is the hardest class.** Across all models, midfielders were most often confused with forwards (CAM, wingers share similar pace/shooting/dribbling scores) and defenders (CDM shares high defending/physic). The RF handled this ambiguity better, pushing MID F1 from 0.71 → 0.80.

**Top features (RF importance):** `defending` (≈0.38) · `shooting` (≈0.21) · `passing` (≈0.13). Highly intuitive — defending separates DEF/GK from attacking positions; shooting separates FWD from GK; passing distinguishes MID.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Usage

```bash
# Clone the repo
git clone https://github.com/<your-username>/player-position-classification-fifa22.git
cd player-position-classification-fifa22

# Run
python data_science_cheick_moctar_traore.py
```

> The dataset (`players_22.csv`) must be downloaded from [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset) and placed in the project root before running.

---

## Project Structure

```
├── data_science_cheick_moctar_traore.py   # Full implementation
├── report.pdf                              # Full written report
└── README.md
```

---

## References

- Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32.
- Quinlan, J. R. (1986). Induction of Decision Trees. *Machine Learning, 1*(1), 81–106.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR, 12*, 2825–2830.
- FIFA 22 Dataset: https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset

---

## Author

**Cheick Moctar Traore** · Student ID: 2592106 · DS703 Machine Learning
