
# 📊 Credit Card Default Prediction Using Behavioral Risk Modeling

## 📌 Project Overview

This project aims to build a forward-looking behavioral scorecard to predict whether a credit card customer will default in the next billing cycle. By leveraging demographic information, repayment history, billing patterns, and engineered behavioral features, the model helps financial institutions proactively identify high-risk customers and improve credit risk management.

---

## 🎯 Objectives

* Build a binary classification model to predict credit card default risk.
* Handle class imbalance using techniques such as SMOTE and class weighting.
* Perform exploratory and financial analysis to understand key drivers of default.
* Engineer meaningful behavioral features like credit utilization and delinquency patterns.
* Compare multiple machine learning models and select the best-performing one.
* Optimize classification threshold based on real-world credit risk trade-offs.
* Generate production-ready predictions on an unlabeled validation dataset.

---

## 📂 Dataset Description

The dataset contains anonymized records of over 25,000 credit card customers with features including:

* **Demographics:** Age, gender, education, marital status
* **Credit Information:** Credit limit, bill amounts, payment amounts
* **Repayment Behavior:** Payment delays (`pay_1` to `pay_6`)
* **Target Variable:** `next_month_default` (1 = Default, 0 = No Default)

---

## ⚙️ Feature Engineering

To capture real-world credit behavior, several derived features were created:

* **credit_utilization:** Ratio of total bill amount to credit limit
* **delinquency_streak:** Number of months with delayed payments
* **on_time_rate:** Proportion of months with timely payments
* **late_payment_rate:** Frequency of delayed payments
* **min_payment_rate:** Frequency of minimum payments
* **max_late_streak:** Longest continuous sequence of late payments

These features significantly improved the model’s ability to identify risky customers.

---

## 🧠 Modeling Approach

The following models were trained and evaluated:

* Logistic Regression
* Decision Tree
* XGBoost
* LightGBM

Class imbalance was handled using **SMOTE**, and models were evaluated using metrics aligned with credit risk priorities, including **F1-score, F2-score, Recall, Accuracy, and AUC-ROC**.

### ✅ Best Model

* **LightGBM** was selected as the final model due to its strong performance in identifying defaulters.
* Threshold tuning was applied to optimize recall and minimize false negatives, aligning with banking risk strategies.

---

## 📈 Model Explainability

SHAP (SHapley Additive exPlanations) was used to interpret model predictions.
Key drivers of default risk included:

* Payment delays (`pay_1`)
* Credit utilization
* Delinquency streaks

This ensured the model remained transparent and interpretable for business stakeholders.

---

## 💼 Business Impact

* Enables early identification of high-risk customers.
* Supports proactive credit risk mitigation strategies such as limit adjustments and targeted alerts.
* Improves decision-making in credit portfolio management by balancing false positives and false negatives.

---

## 📤 Output

* A prediction file (`submission_<EnrollmentNumber>.csv`) containing default probabilities for the validation dataset.
* A reproducible machine learning pipeline for training, evaluation, and prediction.
* Actionable insights into customer credit behavior.

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries:** pandas, numpy, scikit-learn, imbalanced-learn, XGBoost, LightGBM, SHAP, matplotlib, seaborn

---

## 🚀 Future Enhancements

* Incorporate time-series modeling for dynamic credit behavior tracking.
* Deploy the model as an API for real-time credit risk scoring.
* Integrate cost-sensitive learning to directly optimize financial loss.

