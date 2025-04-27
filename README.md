# Playground Series – Season 4, Episode 11: Depression Prediction

## Overview

This repository contains our entry for the Kaggle “Playground Series – Season 4, Episode 11” competition, a synthetic binary classification task aimed at predicting depression status from survey-style tabular data. We explore and compare three advanced ensemble and selection methods—Improved Random Forest pruning, a differentiable Selection Net, and Greedy Dynamic Feature Selection—before combining them into a single end-to-end pipeline.

Link to competition: https://www.kaggle.com/competitions/playground-series-s4e11/overview

## Competition Description

- **Host:** Kaggle  
- **Title:** Playground Series – Season 4, Episode 11  
- **Task:** Binary classification — predict a `Depression` label (0 or 1) for each sample.  
- **Data:** Synthetic tabular features (numerical, categorical, text-encoded), some missing values, engineered to mimic real-world survey data.  
- **Evaluation:**  
  - **Validation metric:** Accuracy  
  - **Leaderboard:** Public score (20% of test), Private score (80% of test)  



## Dataset

All files were provided by the competition:

- `train.csv` – 80% of the labeled data for training/validation  
- `test.csv` – 20% of unlabeled data for final submission  
- `sample_submission.csv` – format for submission  

These data files can be found in /Data folder

### Key columns

| Column                          | Type        | Description                                 |
|---------------------------------|-------------|---------------------------------------------|
| `id`                            | integer     | Unique sample identifier                    |
| `Depression`                    | 0/1         | Target (only in train split)                |
| Various survey and demographic  | numeric / categorical / text-derived | Input features             |


## Our Approach

1. **Baseline Random Forest**  
   - Grid-search CV to tune `max_depth`, `min_samples_split`, `min_samples_leaf`, `n_estimators`.  
   - Code for this can be found in main.ipynb
2. **Improved Random Forest**  
   - Train a large pool of trees, prune via correlation + accuracy thresholds.  
   - Code can be found under Magnus_implementation folder, and is instanciated main.ipynb.
3. **Greedy Dynamic Feature Selection (GDFS)**  
   - Two-network architecture to sequentially select the most informative features.  
    - Code can be found under Dynamic_selection_Isak folder, and is instanciated main.ipynb.
4. **Differentiable Selection Net (e2e-CEL)**  
   - Two-layer MLP outputs scores per learner, applies a Monte Carlo–smoothed top-k knapsack to pick a subset per sample.  
   - Code can be found under Elias_SelectionNet folder, and is instanciated main.ipynb.
5. **Combined Selection Net**  
   - Integrate the ten specialized learners (from steps 1–3) into a single Selection Net pipeline.  
   - Tune top-k ∈ {1…10} via CV. 
   - Code can be found in main.ipynb
