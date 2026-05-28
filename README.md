
# Student Accommodation Management System (SAMS) — Payment Risk Prediction Model
Supervised ML system that predicts whether a student will delay or default on accommodation payments. Built with Python, scikit-learn, and XGBoost. Deployable via Flask REST API.

---

# Payment Risk Machine Learning Model
## Overview
1.  **Open the Model:**
<img width="942" height="432" alt="Capture" src="https://github.com/user-attachments/assets/953a0fc9-98bc-4314-87e0-67e113b01e79" />

2.  **Choose the required payment information:** Provide the required input data.
 <img width="940" height="427" alt="Capture2" src="https://github.com/user-attachments/assets/abf03cbe-1b6d-4a97-98d9-8bab040aa02a" />

5.  **Access payment risk:** Click the final run/submit button. 
<img width="944" height="428" alt="Capture3" src="https://github.com/user-attachments/assets/1c04e8bb-2f77-4493-9704-e108a745b996" />
<img width="931" height="431" alt="Captur4" src="https://github.com/user-attachments/assets/bc5e99c7-2f17-4795-8cd4-9da4d7bd43d4" />
<img width="944" height="431" alt="Capture4" src="https://github.com/user-attachments/assets/8a29a82d-50d9-4cf5-b03b-bdaee8b40c33" />

## 📁 Project Structure

```
sams_ml/
├── colab/
│   └── SAMS_Payment_Risk_Prediction.py   ← Run this in Google Colab
├── web/
│   ├── app.py                            ← Flask REST API
│   ├── models/                           ← Drop .pkl files here after training
│   │   ├── sams_xgb_model.pkl
│   │   ├── sams_rf_model.pkl
│   │   ├── sams_lr_model.pkl
│   │   ├── sams_scaler.pkl
│   │   ├── sams_le_accommodation.pkl
│   │   ├── sams_le_trend.pkl
│   │   └── training_logs.json
│   └── static/
│       └── index.html                    ← Frontend UI
├── requirements.txt
└── README.md
```
## Data augumentation, Preprocessing &  Exploratory Data Analysis(EDA)

Because no real student payment records were available, the SAMS dataset was constructed entirely through synthetic data generation, which serves the same purpose as data augmentation: ensuring the model has enough varied, representative examples to learn from. A total of 1,000 student records were generated using NumPy's random number generator, seeded at 42 for full reproducibility. 

Once the dataset was generated, preprocessing was applied in three sequential stages before any model saw the data. The first stage addressed the two categorical features, accommodation_type and payment_trend, which contained string values that machine learning algorithms cannot process arithmetically.

Exploratory data analysis was conducted on the encoded dataset before model training, with the goal of validating that the synthetic features contained genuine predictive signal and of identifying the most informative patterns for domain understanding

### EDA correlation

<img width="1430" height="1174" alt="eda_correlation" src="https://github.com/user-attachments/assets/b10a0ad0-f155-47f0-a032-6a10febc8ba2" />

### EDA distributions

<img width="2673" height="1471" alt="eda_distributions" src="https://github.com/user-attachments/assets/657a9f0f-c327-4ce5-8495-728a3e623ee9" />

### Feature importance

<img width="2373" height="882" alt="feature_importance" src="https://github.com/user-attachments/assets/0caf92fa-2fa1-436e-a361-c3b5f6188570" />

### Model Training
Three supervised classification models were trained on the preprocessed SAMS dataset: Logistic Regression, Random Forest, and XGBoost. Logistic Regression served as the baseline linear model and used class_weight='balanced' to address class imbalance. 

Random Forest, configured with 200 trees and max_depth=8, reduced variance through bagging and provided feature importance scores. XGBoost, the primary model, used gradient boosting with scale_pos_weight to improve minority class prediction. Hyperparameter tuning was performed using GridSearchCV with 3-fold cross-validation and F1-score optimisation. The best-tuned XGBoost model was saved and integrated into the Flask API. 

Model performance metrics, including accuracy, precision, recall, F1-score, ROC-AUC, training time, and cross-validation scores, were logged in a structured JSON file for reproducibilit

### Evaluation

Model performance was evaluated using accuracy, precision, recall, F1-score, and ROC-AUC to provide a balanced assessment of binary risk prediction. Accuracy alone was insufficient due to class imbalance, while precision and recall measured the model’s ability to correctly identify high-risk students and minimise missed cases.

The F1-score was treated as the main evaluation metric, and ROC-AUC measured overall classification ability across thresholds. Logistic Regression achieved about 80% accuracy and 0.85 ROC-AUC, serving as a strong linear baseline. 

Random Forest improved performance to approximately 87% accuracy and 0.93 ROC-AUC by capturing non-linear relationships. The tuned XGBoost model achieved the best overall results, with around 91% accuracy, F1-score near 0.90, and ROC-AUC of 0.96. Cross-validation showed stable performance with low variation, confirming good generalisation. Feature importance analysis identified late payments, outstanding balance, and monthly income as the strongest predictors across both ensemble models.


## Results and Visualizations

### Model Performance Comparison

<img width="1923" height="873" alt="model_comparison" src="https://github.com/user-attachments/assets/a43f07b3-df2c-4025-a721-24a8f5dc7ae2" />


### ROC Curve


<img width="1323" height="1023" alt="roc_curves" src="https://github.com/user-attachments/assets/8c23a66f-1c56-4aed-8acd-34aa15dbaa95" />


### Confusion Matrix 
<img width="2486" height="735" alt="confusion_matrices" src="https://github.com/user-attachments/assets/4417b0ec-870b-4890-a782-5f5c6bd99f78" />

---

## 🔌 API Reference

### POST /api/predict

**Request body (JSON):**
```json
{
  "payment_history":     12,
  "late_payments":        3,
  "outstanding_balance": 5000.00,
  "accommodation_type":  "shared",
  "monthly_income":      7500.00,
  "payment_trend":       "stable",
  "semester":             3,
  "length_of_stay":      12
}
```

**Response:**
```json
{
  "risk_label":    1,
  "risk_category": "High Risk",
  "confidence":    0.7842,
  "risk_factors":  ["High late payment count (7)", "Declining payment trend"],
  "timestamp":     "2025-01-15T14:32:00.123456"
}
```

### GET /api/health
```json
{"status": "ok", "models_loaded": true}
```

### GET /api/model-info
Returns model metadata and training logs.

---

## 📊 Dataset Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `payment_history` | int | 0–24 | Number of on-time payments |
| `late_payments` | int | 0–12 | Number of late payments |
| `outstanding_balance` | float | 0–30,000 | Unpaid balance (ZAR) |
| `accommodation_type` | str | single/shared | Room type |
| `monthly_income` | float | 2,000–20,000 | Income estimate (ZAR) |
| `payment_trend` | str | increasing/stable/decreasing | Payment trend |
| `semester` | int | 1–8 | Current semester |
| `length_of_stay` | int | 1–48 | Months in accommodation |
| `risk_label` | int | 0 or 1 | **Target**: 0=Low, 1=High |

---

## 🤖 Models Compared

| Model | Typical Accuracy | Typical F1 | Notes |
|-------|-----------------|------------|-------|
| Logistic Regression | ~78–82% | ~0.77–0.81 | Fast, interpretable |
| Random Forest | ~85–89% | ~0.85–0.88 | Good baseline tree model |
| XGBoost (tuned) | ~87–92% | ~0.87–0.91 | **Best performer** |

---

## 🔑 Key Insights (from EDA)

1. **Late payments** is the strongest predictor of high risk
2. **Outstanding balance** strongly correlates with risk
3. **Monthly income** is inversely correlated with risk
4. Students with a **decreasing payment trend** are 3× more likely to be high-risk
5. **Single room** students face higher risk due to higher costs

---


