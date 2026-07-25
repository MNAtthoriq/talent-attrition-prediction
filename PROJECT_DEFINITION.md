# Project Definition

## Business Explanation

### A. Problem Statement

The company recruits data science talent from participants in its training program. Training capacity and investment are limited, while some participants leave the company after receiving training.

The company wants to prioritize eligible candidates who are most likely to remain after training, with the objective of improving retention and reducing unrecovered training investment and employee turnover costs.

### B. Supported Decision

Before training begins, the model ranks eligible candidates from lowest to highest predicted attrition risk.

Candidates with lower predicted attrition risk receive priority until the available training capacity is filled.

### C. Business Metrics

The primary business metric is the **attrition rate** among selected participants.

The project compares:

1. Baseline attrition rate without model-based selection.
2. Attrition rate with model-based selection.
3. Absolute and relative reductions in attrition rate.
4. Estimated training and turnover costs avoided.

Both approaches use the same number of training places so that the comparison measures the value of model-based selection rather than the effect of training fewer participants.

## Technical Explanation

### A. Prediction Target

The target is interpreted from the dataset's business context as:

* `0`: the participant **stays** with the company after training.
* `1`: the participant **leaves** the company after training.

The model estimates the probability of post-training attrition.

### B. Prediction Time

Predictions are made before training begins using information collected during candidate registration and enrollment.

Consequently, `training_hours` is unavailable at prediction time and excluded from the model.

### C. Prediction Features

| Column                   | Role                     | Available at prediction time? | Modeling decision                                                   |
| ------------------------ | ------------------------ | ----------------------------: | ------------------------------------------------------------------- |
| `enrollee_id`            | Identifier               |                           Yes | Exclude from model                                                  |
| `city`                   | Participant information  |                           Yes | Candidate feature; evaluate location-proxy risk                     |
| `city_development_index` | City characteristic      |                           Yes | Candidate feature                                                   |
| `gender`                 | Protected characteristic |                Yes or missing | Use for fairness auditing; exclude from deployable model by default |
| `relevent_experience`    | Experience information   |                           Yes | Candidate feature                                                   |
| `enrolled_university`    | Education information    |                           Yes | Candidate feature                                                   |
| `education_level`        | Education information    |                           Yes | Candidate feature                                                   |
| `major_discipline`       | Education information    |                Yes or missing | Candidate feature                                                   |
| `experience`             | Employment information   |                           Yes | Candidate feature                                                   |
| `company_size`           | Employer information     |                Yes or missing | Candidate feature                                                   |
| `company_type`           | Employer information     |                Yes or missing | Candidate feature                                                   |
| `last_new_job`           | Employment history       |                Yes or missing | Candidate feature                                                   |
| `training_hours`         | Training information     |                            No | Exclude because it is unavailable before training                   |

### D. Model-Evaluation Metrics

PR-AUC is the primary model-selection metric because the positive class is less common and identifying participants at risk of leaving is the main prediction objective.

Recall is the secondary model-selection metric because a false negative (an actual leaver predicted to stay) may result in unrecovered training investment and additional turnover costs.

Training and validation performance are compared to identify possible overfitting or underfitting.

Models and classification thresholds are selected using cross-validation on the development data. The test set is used only once for final unbiased evaluation.
