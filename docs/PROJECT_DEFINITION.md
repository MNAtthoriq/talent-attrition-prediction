# Project Definition

## Business Explanation

### A. Problem Statement

The company recruits data science talent from participants in its training program. Training capacity and budget are limited, while some participants are looking for other job opportunities instead of planning to work with the company after training.

The company wants to prioritize eligible candidates with lower job-change intent so that training capacity, recruitment time, and training investment are used more effectively.

### B. Supported Decision

Before data science training begins, the model ranks eligible candidates from the lowest to the highest predicted job-change intent.

Candidates with lower predicted job-change intent receive higher priority until all available training slots are filled. The model provides decision support; a human remains responsible for the final selection.

### C. Business Metrics

The primary business metric is the **job-change intent rate among selected participants**.

The project compares:

1. The baseline job-change intent rate without model-based selection.
2. The job-change intent rate with model-based selection.
3. The absolute and relative reduction in job-change intent rate.

Both approaches use the same number of training slots. This isolates the value of model-based ranking from the effect of selecting fewer participants.

## Technical Explanation

### A. Prediction Target

Based on the dataset documentation:

- `0`: the participant is not looking for a job change.
- `1`: the participant is looking for a job change.

The model estimates job-change intent at the time represented by the dataset. It does not directly predict future employee attrition, confirm that a participant will leave an employer, or guarantee that a participant will join the company.

### B. Prediction Time

Predictions are made **before training begins** using information collected during registration and enrollment.

Consequently, `training_hours` is unavailable at prediction time and excluded from the model.

### C. Prediction Features

| Column | Role | Modeling decision |
| :--- | :--- | :--- |
| `enrollee_id` | Identifier | Exclude from modeling; retain only for split auditing |
| `city_development_index` | Numeric city characteristic | Candidate feature |
| `city` | Anonymized location identifier | Evaluated during model selection; excluded by the selected pipeline |
| `gender` | Protected characteristic | Exclude from deployable model; use only for fairness auditing |
| `relevant_experience` | Relevant-work-experience category | Candidate feature |
| `enrolled_university` | University-enrollment status | Candidate feature |
| `education_level` | Education category | Candidate feature |
| `major_discipline` | Academic-discipline category | Candidate feature |
| `experience` | Experience band | Candidate feature |
| `company_size` | Employer-size band | Candidate feature |
| `company_type` | Employer type | Candidate feature |
| `last_new_job` | Time since last job change | Candidate feature |
| `training_hours` | Training information | Exclude because it is unavailable before training |
| `target` | Prediction label | Exclude from features |

The raw dataset spells `relevant_experience` as `relevent_experience`; the project normalizes the name in the modeling and API layers.

### D. Evaluation Contract

- Use a reproducible stratified 80/20 development-test split with random state `42`.
- Perform preprocessing, model comparison, and hyperparameter tuning using development data only.
- Use 5-fold stratified cross-validation for model selection.
- Use **PR-AUC** as the primary model-selection metric because the positive class represents about 25% of the data.
- Use recall, ROC-AUC, Brier score, cross-validation variability, and the train-validation gap as supporting diagnostics.
- Select the classification threshold from out-of-fold development predictions using F2.
- Evaluate the selected pipeline once on the untouched test set.
- Compare business selection performance at equal training-capacity levels.

## 8. Decision Boundaries

The system is intended to prioritize candidates for further review. It is **not** intended to:

- Automatically reject applicants.
- Replace interviews, assessments, or human judgment.
- Predict confirmed future resignation or employment.
