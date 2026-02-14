# 🍷 Wine Quality Classification System

A comprehensive machine learning web application for classifying wine quality using multiple ML algorithms.

## Problem Statement

This project implements a binary classification system to predict wine quality (Good vs Bad) based on physicochemical properties. The system compares six different machine learning algorithms to determine which performs best for wine quality prediction.

##  Dataset Description

**Dataset:** Wine Quality Dataset  
**Source:** Google Drive (winequalityN.csv)  
**Classification Type:** Binary Classification  
**Target Variable:** Quality (converted to binary: Good Wine ≥6, Bad Wine <6)

### Features:
1. **type** - Wine type (red/white) - Categorical
2. **fixed acidity** - Fixed acidity level
3. **volatile acidity** - Volatile acidity level
4. **citric acid** - Citric acid content
5. **residual sugar** - Residual sugar content
6. **chlorides** - Chloride content
7. **free sulfur dioxide** - Free SO2 content
8. **total sulfur dioxide** - Total SO2 content
9. **density** - Wine density
10. **pH** - pH value
11. **sulphates** - Sulphate content
12. **alcohol** - Alcohol percentage

##  Models Used

### Model Performance Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|----------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.7440 | 0.8046 | 0.7709 | 0.8472 | 0.8072 | 0.4331 |
| Decision Tree | 0.7819 | 0.7619 | 0.8213 | 0.8374 | 0.8293 | 0.5276 |
| KNN | 0.7602 | 0.8274 | 0.7880 | 0.8496 | 0.8176 | 0.4720 |
| Naive Bayes | 0.7022 | 0.7556 | 0.7446 | 0.8056 | 0.7739 | 0.3422 |
| Random Forest | 0.8314 | 0.9051 | 0.8521 | 0.8875 | 0.8695 | 0.6328 |
| XGBoost | 0.8121 | 0.8791 | 0.8410 | 0.8667 | 0.8537 | 0.5917 |

### Model Observations

| ML Model | Observation |
|----------|-------------|
| **Logistic Regression** | Demonstrates solid baseline performance with balanced metrics. The linear model achieves good AUC (0.8046) indicating effective class separation. Moderate accuracy (0.7440) with consistent precision (0.7709) and recall (0.8472). Best suited for interpretable predictions where model transparency is crucial. Shows stable performance across different evaluation metrics. |
| **Decision Tree** | Shows improved accuracy (0.7819) with strong precision (0.8213), making it reliable for positive predictions. The MCC score (0.5276) indicates good balanced performance. Non-linear decision boundaries effectively capture complex patterns in wine quality features. Provides interpretable rules and feature importance insights for wine quality assessment. |
| **KNN** | Instance-based learning demonstrates moderate performance with well-balanced metrics. Achieves good AUC (0.8274) and maintains high recall (0.8496). Accuracy of 0.7602 indicates effectiveness in capturing local patterns. The model benefits from standardized features and performs consistently with the wine dataset's continuous variables. |
| **Naive Bayes** | Probabilistic classifier with fastest training time but lowest overall performance. Despite feature independence assumption, achieves moderate AUC (0.7556). Lower accuracy (0.7022) and MCC (0.3422) compared to other models. Maintains reasonable recall (0.8056) for identifying good wines. Suitable for quick baseline predictions with limited computational resources. |
| **Random Forest** | **Best overall performer** with exceptional accuracy (0.8314) and outstanding AUC (0.9051). Achieves highest precision (0.8521) and excellent recall (0.8875), resulting in superior F1 score (0.8695). Top MCC (0.6328) indicates highly reliable predictions for both classes. Ensemble bagging effectively reduces overfitting while capturing complex feature interactions. Recommended for production deployment. |
| **XGBoost** | **Second-best performer** with strong accuracy (0.8121) and excellent AUC (0.8791). Gradient boosting achieves high precision (0.8410) and recall (0.8667) with robust MCC (0.5917). Regularization provides superior generalization capabilities. Balanced performance across all metrics makes it a reliable alternative to Random Forest, especially when feature importance and model interpretability are priorities. |
