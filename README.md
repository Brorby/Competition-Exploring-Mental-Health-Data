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

The final submission is in the submission.csv file.

### Key columns

| Column                         | Type                                 | Description                  |
| ------------------------------ | ------------------------------------ | ---------------------------- |
| `id`                           | integer                              | Unique sample identifier     |
| `Depression`                   | 0/1                                  | Target (only in train split) |
| Various survey and demographic | numeric / categorical / text-derived | Input features               |

## Our Approaches

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
   - Integrate the nine specialized learners (from steps 1–3) into a single Selection Net pipeline.
   - Tune top-k ∈ {1…9} via CV.
   - Code can be found in main.ipynb

## More in-depth descriptions of the individual implementations:

### Improved Random Forest

First, we train a pool of trees of size N_final + m \* N_final via bagging. Each tree is scored on a held-out validation split. We compute cosine similarities between feature-subset vectors of all tree pairs and derive candidate correlation thresholds by interpolating between the minimum, mean, and maximum similarity values. For pairs exceeding the chosen threshold, the lower-accuracy tree is pruned. This process continues until only N_final trees remain. Final tree hyperparameters: N_final=200, max_depth=15, min_samples_leaf=4, min_samples_split=2, pruning multiplier (m)=0.7.

### Greedy Dynamic Feature Selection (GDFS)

GDFS alternates between a policy network that greedily selects the most informative feature at each step and a selector network that predicts the target given the chosen subset. We pretrain both networks separately for 100 epochs on class-balanced 95/5 splits, then jointly train end-to-end for 3 epochs per temperature parameter τ. Key hyperparameters: batch_size=128, dropout_rate=0.4, hidden_size=64, learning_rate=0.001, max_features=20.

### Differentiable Selection Net

We first train ten specialized learners (five model types on two 95/5 splits). A two-layer MLP (input → 128 ReLU → amount of base learners (10)) produces scores for each learner. We L2-normalize these scores and apply a Monte Carlo–smoothed top-k knapsack layer (ε=0.1, m=1000) to obtain a binary mask per sample. The masked learner probabilities are summed and re-softmaxed for the final prediction. We tune k over {1…10} with 5 epochs (batch_size=512, Adam lr=1e-3) and select the model state at best validation accuracy.
