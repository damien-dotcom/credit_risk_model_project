# Credit Risk Scoring Model

## Problem
Predicting whether a loan applicant is a 'good' or 'bad' credit risk, using the Germaan Credit dataset (1000 applications, 20 features).

## Approach
1. Encoded ordinal categoricals, preserving natural order; one-hot encoded nominal categoricals
2. Scaled numeric features using StandardScaler
3. Created 80/20 train_test split, preserving 70/30 class balance
4. Trained a logistic regression baseline, then Random Forest with light hyperparameter adjustment(GridSearchCV, 5-fold CV on AUC-ROC)
5. Interpreted results with feature importances and SHAP
.
## Metrics
| Model | AUC-ROC | Precision (Bad) | Recall (Bad) |
|---|---|---|---|
| Logistic Regression | 0.76 | 0.61 | 0.38 |
| Random Forest (tuned) | 0.81 | 0.61 | 0.63 |

## Key Findings
- checking_account_status was the strongest predictor of default, with the highest feature importance and SHAP values
- credit_amount and duration_months also had strong influence, as 2nd and 3rd respectively
- The tuned Random Forest model improved recall for 'bad' borrowers a great deal, catching more defaulters with lowering precision much

## What I'd Improve
- Try using XGBoost for comparisons
- Use SMOTE instead of class_weight to treat the class imbalance

## Conclusion
The Random Forest model, tuned using GridSearchCV, outperformed the logistic regression baseline in AUC-ROC and recall for 'bad' borrowers. SHAP analysis confirmed that liquidity features greatly influenced credit risk prediction. Applications with low checking and savings account balances pushed predictions toward 'bad', while those with strong balances were classified as 'good'. Longer durations and higher credit amounts also played major roles, increasing the risk of default. These patterns were expected, and aligned with my expectations. With further work, the model could become a strong tool for credit risk scoring.
