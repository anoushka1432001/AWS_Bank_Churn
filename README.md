CloudSageMaker: Bank Churn Prediction with AWS SageMaker

# 📊 CloudSageMaker: Bank Churn Prediction with AWS SageMaker

Welcome to the **CloudSageMaker** repository! 🚀  
This project focuses on **bank churn prediction** using **AWS SageMaker**, **SMOTE** for handling imbalanced data, **XGBoost** for modeling, and **Python's sklearn** for feature engineering.

---

## 🧠 Project Overview

Bank churn prediction helps financial institutions identify customers likely to leave.  
This model was developed using **AWS SageMaker** with data preprocessing, feature engineering, and model training.

**Key Techniques Used:**
- **AWS SageMaker**: Cloud-based machine learning model training.  
- **SMOTE**: Oversampling technique to balance the dataset.  
- **XGBoost**: Gradient boosting algorithm for predictive modeling.  
- **Pandas/Sklearn**: Data preprocessing and model evaluation.  

---

## 📁 **Project Structure**

```bash
.
├── CloudSageMaker.pdf          # Project documentation
├── Main.ipynb                  # Main notebook
├── script.py                   # Training script for Set 1 features
├── script_2.py                 # Training script for Set 2 features
├── Train_set1.csv              # Training dataset (Set 1)
├── Test_set1.csv               # Testing dataset (Set 1)
├── Train_set2.csv              # Training dataset (Set 2)
├── Test_set2.csv               # Testing dataset (Set 2)
└── README.md                   # Project README

---

## 📑 Dataset Description

The dataset contains **bank customer data** with **21 features** and **10,127 records**.  
It includes **demographic details**, **credit behavior**, and **account activity**.

### 🔍 Key Features:

- `Attrition_Flag`: Customer status (Existing/Attrited) → **Target Variable**  
- `Customer_Age`, `Gender`, `Dependent_count`: Demographic info  
- `Total_Trans_Ct`, `Credit_Limit`: Transaction details  
- `Months_on_book`: Customer relationship length  

---

## ⚙️ Setup & Installation

Install the necessary libraries:

```bash
pip install pandas numpy scikit-learn sagemaker imbalanced-learn xgboost boto3
```

Ensure AWS credentials are configured for SageMaker access.

---

## 🔨 Step-by-Step Execution

### 1️⃣ Preprocessing & Feature Engineering

- **Missing Value Imputation**: Replaced missing values with **mode**.  
- **Encoding**: Applied **Label Encoding** and **One-Hot Encoding**.  
- **Feature Selection**: Used **Random Forest** and **RFE** to select key features.

---

### 2️⃣ Synthetic Minority Over-sampling Technique (SMOTE)

Handled class imbalance by applying **SMOTE**:

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
```

---

### 3️⃣ Model Training with AWS SageMaker

We trained **two models** using **AWS SageMaker** with **Random Forest**.

**Training Command:**

```python
from sagemaker.sklearn.estimator import SKLearn

FRAMEWORK_VERSION = "0.23-1"

sklearn_estimator = SKLearn(
    entry_point="script.py",
    role="arn:aws:iam::<your-account-id>:role/SageMakerExecutionRole",
    instance_count=1,
    instance_type="ml.m5.4xlarge",
    framework_version=FRAMEWORK_VERSION,
    base_job_name="rf-scikit",
    hyperparameters={'n_estimators': 150, 'random_state': 0}
)

sklearn_estimator.fit({"train": trainpath_set1, "test": testpath_set1})
```

---

### 4️⃣ Model Evaluation

**Performance Metrics:**

| **Metric**       | **Score** |
|-------------------|----------|
| **Accuracy**      | 79.2%    |
| **Precision**     | 77-81%   |
| **Recall**        | 77-82%   |
| **F1-Score**      | 79%      |

**Confusion Matrix** analysis showed a balanced performance between **attrited** and **non-attrited** customers.

---

## 📊 Key Insights

- **Feature Importance**: `Total_Trans_Amt` and `Total_Revolving_Bal` were the **most predictive features**.  
- **SMOTE** effectively **balanced the dataset** from **8500:1627** to **8500:8500**.  
- **AWS SageMaker** provided a **scalable infrastructure** for training and deployment.

---

## 🚀 Deployment

The trained model can be deployed on AWS SageMaker for **real-time inference** or **batch predictions**.

**Sample Inference Code:**

```python
import boto3
import json

sm_client = boto3.client('sagemaker-runtime')
endpoint_name = 'bank-churn-predictor'

response = sm_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=json.dumps({'instances': [[45, 3, 5, 0.25, 7]]})
)

result = json.loads(response['Body'].read())
print(f"Predicted Class: {result['predictions'][0]['classes']}")
```

---

## 💡 Lessons Learned

1. **AWS SageMaker's flexibility** allows seamless scaling for **ML workloads**.  
2. **Feature engineering decisions** can significantly impact model performance.  
3. **Monitoring imbalanced datasets** requires a combination of **AUC-ROC** and **precision/recall metrics**.

---

## 🛠️ Potential Improvements

- **Experiment with other models**: LightGBM, CatBoost.  
- **Optimize hyperparameters** using **SageMaker's Hyperparameter Tuning**.  
- **Deploy a real-time inference endpoint** for continuous predictions.  

---

