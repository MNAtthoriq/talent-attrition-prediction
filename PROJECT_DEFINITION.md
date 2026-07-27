# Project Definition

## Business Explanation

### A. Problem Statement

The company recruits data science talent from participants in its training program. Training capacity and budget are limited, while some participants are looking for other job opportunities instead of planning to work with the company after training.

The company wants to prioritize eligible candidates with lower job-change intent. This helps the company use training capacity, recruitment time, and training investment more effectively.

### B. Supported Decision

Before training begins, the model ranks eligible candidates from the lowest to the highest predicted job-change intent.

Candidates with lower predicted job-change intent receive higher priority until all available training slots are filled.

### C. Business Metrics

The primary business metric is the **job-change intent rate** among selected participants.

The project compares:

1. The baseline job-change intent rate without model-based selection.
2. The job-change intent rate with model-based selection.
3. The absolute and relative reduction in job-change intent rate.

Both approaches use the same number of training slots. This ensures that the comparison measures the value of model-based selection, rather than the effect of selecting fewer participants.

## Technical Explanation

### A. Prediction Target

Based on the dataset documentation, the target is interpreted as:

* `0`: the participant is not looking for a job change.
* `1`: the participant is looking for a job change.

The model estimates stated job-change intent at the time represented by the
dataset. It does not directly predict future employee attrition or confirm that
a participant will leave an employer.

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

PR-AUC is the primary model-selection metric because the positive class is less common and identifying participants with job-change intent is the main prediction objective.

Recall is the secondary model-selection metric because a false negative (a participant who is looking for a job change but is predicted as not looking) may incorrectly receive higher training priority.

Training and validation performance are compared to identify possible overfitting or underfitting.

Models and classification thresholds are selected using cross-validation on the development data. The test set is used only once for final unbiased evaluation.
