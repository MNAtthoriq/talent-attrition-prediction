# Model Card: Talent Job-Change Intent Classifier

## A. Model Summary

| Item | Value |
| :--- | :--- |
| Model | LightGBM binary classifier inside a scikit-learn pipeline |
| Registered name | `talent-job-change-intent-classifier` |
| Registered version | `1` |
| Alias | `candidate` |
| Selected Optuna trial | `22` of `40` completed trials |
| Preprocessing | `ordinal_aware_without_city` |
| Decision threshold | `0.3792` |
| Created | 2026-07-27 |
| Source-data SHA-256 | `8b78da3482032500df40d5359c36ba4a59e28ccd6fc272b050dcf6dee37f114c` |
| Source Git commit | `0168c5d416bf059a8a3b8716a912d6bb4bdefec4` |

The model estimates the probability that a training-program participant is looking for a job change. Lower predicted intent receives higher training priority when training capacity is limited.

## B. Intended Use

### Primary use

- Rank eligible participants before training begins.
- Compare model-based selection with a same-size baseline.
- Support recruitment and training-capacity planning.
- Serve individual probabilities and batch priority rankings through an API.

### Intended users

- Data science and machine-learning reviewers.
- Recruitment or training-program analysts.
- Human decision-makers responsible for candidate selection.

### Out-of-scope use

The model must not be used as the sole basis for hiring, rejection, disciplinary action, or employee-retention decisions. It does not predict confirmed future attrition or guarantee whether a participant will work for the company.

## C. Data

### Source

[HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists), pinned to dataset version `1`.

| Split | Rows | Positive rows | Positive rate |
| :--- | ---: | ---: | ---: |
| Development | 15,326 | 3,822 | 24.94% |
| Test | 3,832 | 955 | 24.92% |
| Total | 19,158 | 4,777 | 24.94% |

The split is stratified with random state `42`. The test set is held out during model and threshold selection.

### Target

- `0`: not looking for a job change.
- `1`: looking for a job change.

This label represents stated intent in the source data, not an observed future resignation or employment outcome.

### Feature handling

The candidate feature set contains registration and enrollment information. The selected pipeline uses:

- `city_development_index`
- `relevant_experience`
- `enrolled_university`
- `education_level`
- `major_discipline`
- `experience`
- `company_size`
- `company_type`
- `last_new_job`

The winning trial excludes `city`.

The project also excludes:

- `enrollee_id`: identifier.
- `gender`: protected characteristic retained only for fairness auditing.
- `training_hours`: unavailable before training.
- `source_sha256`: lineage field.
- `target`: prediction label.

## D. Selected Pipeline

The search compares three model families:

1. Logistic Regression.
2. Random Forest.
3. LightGBM.

Each model is evaluated with:

- All categorical features one-hot encoded or ordinal-aware preprocessing.
- City included or excluded.
- Different minimum frequencies for infrequent categories.
- Model-specific regularization and complexity parameters.

The best preprocessing and model configuration is selected by Optuna. It shows:

### Preprocessing

The selected preprocessing configuration is `ordinal_aware_without_city`:

- Median imputation with missing indicators for numeric features.
- Standard scaling for numeric features.
- Constant-value imputation and one-hot encoding for nominal features.
- Most-frequent imputation, explicit order, and standard scaling for ordinal features.
- Unknown ordinal values encoded as `-1`.
- Infrequent nominal categories grouped with a minimum frequency of `25`.
- Sparse transformed output.
- City excluded.

### Model Parameters

The selected model is LightGBM with the following parameters:

| Parameter | Value |
| :--- | ---: |
| `n_estimators` | 550 |
| `learning_rate` | 0.01917 |
| `num_leaves` | 87 |
| `max_depth` | 14 |
| `min_child_samples` | 100 |
| `subsample` | 0.64375 |
| `colsample_bytree` | 0.67147 |
| `reg_alpha` | 8.93345 |
| `reg_lambda` | 8.48791 |
| `class_weight` | `balanced` |

## E. Technical Performance

### Development-set model selection

Models were compared using 5-fold stratified cross-validation on the development set. PR-AUC was the primary selection metric because participants with job-change intent represent about 25% of the data.

The dummy baseline achieved a validation PR-AUC of `0.2494`, while the selected model achieved `0.5553`, or approximately **122.7% higher**.

| Metric                               | Development Result |
| :----------------------------------- | -----------------: |
| Train PR-AUC                         |             0.5795 |
| Mean validation PR-AUC               |         **0.5553** |
| Validation PR-AUC standard deviation |             0.0116 |
| Train-validation PR-AUC gap          |             0.0242 |
| Mean validation ROC-AUC              |             0.8063 |
| Out-of-fold recall                   |             0.7972 |
| F2-selected threshold                |             0.3792 |

The `0.0242` train-validation gap and low variation across folds suggest limited overfitting during model selection.

The threshold of `0.3792` was selected from out-of-fold development predictions by maximizing F2, which gives recall more weight than precision.

### Final evaluation on the untouched test set

After model and threshold selection were complete, the pipeline was fitted on the full development set and evaluated once on the untouched test set.

| Metric                     | Test Result |
| :------------------------- | ----------: |
| PR-AUC / Average Precision |  **0.5461** |
| ROC-AUC                    |  **0.8013** |
| Recall                     |  **0.7770** |
| Precision                  |      0.4812 |
| F1                         |      0.5943 |
| F2                         |      0.6919 |
| Brier score                |      0.1686 |

The test PR-AUC is approximately **2.19 times** the positive-class prevalence baseline.

It is also close to the cross-validation result:

```text
Validation PR-AUC: 0.5553
Test PR-AUC:       0.5461
Difference:        0.0092
```

This small decrease suggests that the selected pipeline generalizes reasonably well to unseen data.

### Classification results at threshold 0.3792

|                  | Predicted no intent | Predicted intent |
| :--------------- | ------------------: | ---------------: |
| Actual no intent |               2,077 |              800 |
| Actual intent    |                 213 |              742 |

At this threshold, the model correctly identifies `742` of the `955` participants with job-change intent, producing a recall of `0.7770`.

The threshold intentionally favors recall. A false negative means that a participant with job-change intent is classified as having no intent and may incorrectly receive higher training priority.

The threshold is used only when a binary classification is required. The main training-selection workflow ranks participants by predicted probability and selects those with the lowest predicted job-change intent until the available capacity is filled.

## F. Business Selection Performance

The comparison uses the same number of training slots for model-based and baseline selection.

| Training capacity | Selected rows | Baseline intent rate | Model-selected intent rate | Absolute reduction | Relative reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25% | 958 | 24.92% | **6.68%** | **18.24 pp** | **73.19%** |
| 50% | 1,916 | 24.92% | **8.35%** | **16.57 pp** | **66.49%** |
| 75% | 2,874 | 24.92% | **13.29%** | **11.63 pp** | **46.67%** |

These figures measure performance on the held-out dataset. They do not prove real-world cost savings or future employee retention.

## G. Fairness Audit

Gender is excluded from the deployable model and used only for post-hoc evaluation.

| Group | Rows | Positive rate | Predicted-positive rate | Recall | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Female | 253 | 24.90% | 40.71% | 0.8095 | 0.6213 |
| Male | 2,648 | 22.70% | 36.29% | 0.7404 | 0.5136 |
| Other | 35 | 28.57% | 48.57% | 0.8000 | 0.7922 |
| Missing | 896 | 31.36% | 51.45% | 0.8470 | 0.5954 |

Observed maximum gaps:

- Predicted-positive rate: `0.1516`.
- Recall: `0.1065`.

This audit is descriptive, not proof of fairness. The `Other` group contains only 35 rows, and 896 test rows have missing gender values. Group metrics should be re-evaluated with representative production data and agreed fairness criteria before operational use.

## H. Final Evaluation Package

The complete generated evaluation package is available in the
[model results release asset](https://github.com/MNAtthoriq/talent-job-change-intent-prediction/releases/latest/download/model_results.zip).

It contains:

- `split_manifest.json`
- `model_selection_summary.json`
- `final_evaluation.json`
- Precision-recall and ROC curves
- Confusion matrix
- Training-capacity performance curve

The package contains aggregated metrics and hashes only; it does not include participant-level data.

## I. Limitations and Risks

1. **Intent is not outcome.** The target records job-change intent, not confirmed resignation, retention, or employment with the company.
2. **Limited generalization evidence.** The dataset represents one historical context and may not match another country, time period, labor market, or training program.
3. **No causal interpretation.** Features associated with job-change intent are not proven causes.
4. **Proxy risk remains.** Education and employment-history features may encode socioeconomic differences even though gender and city are excluded.
5. **Thresholds are context-dependent.** The selected F2 threshold reflects the current error trade-off and must be reviewed when training capacity or costs change.
6. **Probabilities are estimates.** Individual predictions are uncertain and must not be treated as guaranteed outcomes.
7. **Fairness evidence is limited.** Some groups are small or have substantial missingness.
8. **No verified financial outcome.** The dataset does not contain training costs, recruitment costs, or realized post-training employment.
9. **Public deployment is a demonstration.** Production use would require authentication, rate limiting, access controls, monitoring, and an incident-response process.