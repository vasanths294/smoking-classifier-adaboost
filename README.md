# Smoking History Prediction Using AdaBoost (and Other ML Models)

## Project Overview

The objective of this project is to predict whether an individual is a smoker or a non-smoker based on various health indicators. This binary classification task plays a vital role in public health analytics and enables proactive, data-driven medical interventions.

- Dataset: Smoking and Drinking Dataset with Body Signal  
- Target variable: `smoking` status  
- Features: Blood pressure, cholesterol, vision, hearing, triglycerides, liver function, etc.  
- Problem Type: Binary Classification  
- Techniques Explored:  
  - Gradient Boosting  
  - AdaBoost (Scikit-learn and Custom from Scratch)  
  - Random Forest  
  - Neural Networks  
  - Support Vector Machines  

This repository focuses on the AdaBoost Classifier, implemented by Vasanthakumar Shanmugavadivel, using both Scikit-learn and a custom NumPy implementation.

## Dataset Source

This project uses the [Smoking and Drinking Dataset with Body Signal](https://www.kaggle.com/datasets/kukuroo3/smoking-and-drinking-dataset-with-body-signal), publicly available on Kaggle.


## What Is AdaBoost?

AdaBoost (Adaptive Boosting) is an ensemble method that combines multiple weak learners — typically decision stumps — to form a strong classifier. It iteratively adjusts sample weights to focus on difficult cases, improving performance over rounds.

## Implementation Details

### 1. Using Scikit-learn

- **Library**: `sklearn.ensemble.AdaBoostClassifier`
- **Base Estimator**: `DecisionTreeClassifier(max_depth=1)`
- **Parameters**:
  - `n_estimators = 10`
  - `learning_rate = 1.0`
- **Code Snippet**:

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

base_estimator = DecisionTreeClassifier(max_depth=1)
model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=10, learning_rate=1.0)
model.fit(X_train, y_train)
```

### 2. AdaBoost From Scratch (NumPy Only)

**Implements:**

- Decision stumps as weak learners  
- Weighted sampling  
- **Alpha computation:**  
  `α = log((1 - err) / err)`

- **Final classifier:**  
  `F(x) = sign(Σ αᵢ · hᵢ(x))`

**Focuses on:**

- Complete control and transparency  
- Manual handling of sample weights and decision thresholds

## Performance Comparison: Sklearn vs Scratch AdaBoost

### Scikit-learn AdaBoost

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| -1.0  | 0.75      | 0.81   | 0.78     | 1117    |
| 1.0   | 0.78      | 0.72   | 0.75     | 1080    |
| **Macro Avg** | **0.765** | **0.765** | **0.765** | **2197** |

---

### Custom AdaBoost (NumPy)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| -1.0  | 0.73      | 0.76   | 0.75     | 1117    |
| 1.0   | 0.74      | 0.71   | 0.73     | 1080    |
| **Macro Avg** | **0.735** | **0.735** | **0.74** | **2197** |

---

### Insights

| Metric     | Scikit-learn AdaBoost | Scratch AdaBoost |
|------------|------------------------|------------------|
| Precision  | 0.75–0.78              | 0.73–0.74        |
| Recall     | 0.72–0.81              | 0.71–0.76        |
| F1-Score   | 0.75–0.78              | 0.73–0.75        |
| Accuracy   | ~76–77% (inferred)     | ~74–75% (inferred) |

---

### Conclusion

The custom AdaBoost implementation is highly effective — matching the performance of Scikit-learn’s `AdaBoostClassifier` within 1–2% on all major metrics. This validates the correctness of the algorithm and demonstrates strong foundational understanding of ensemble methods.
