# Tour & Travels Customer Churn Prediction

A comprehensive machine learning solution to predict customer churn in the travel industry. This project implements several classification models to identify customers likely to discontinue using travel services.

## Models Implemented

- Random Forest Classifier
- Logistic Regression
- Gradient Boosting
- XGBoost
- SVM (Support Vector Machine)

## Features

### Data Exploration

- Comprehensive analysis of customer demographics and behavior patterns
- Visualization of age distribution, frequent flyer status, and service usage
- Correlation analysis between features and churn status
- Income class analysis related to customer retention
- Target distribution analysis showing overall churn rate

### Data Preprocessing

- Handling of missing values and inconsistent data entries
- Conversion of 'No Record' values to standardized formats
- Outlier detection using IQR method for numerical features
- Categorical variable encoding and standardization
- Feature scaling via StandardScaler implementation

### Feature Engineering

- Creation of age groups for demographic segmentation
- Development of interaction features (FreqXServices, SocialXServices, HotelXServices)
- Ratio feature generation (ServicesPerAge)
- Income level encoding from categorical to ordinal representation
- Binary encoding of categorical variables (Yes/No features)

### Model Training

- Train-test splitting with stratification to maintain class distribution
- Implementation of preprocessing pipelines for numerical and categorical features
- Missing value imputation strategies for different feature types
- Cross-validation implementation for robust model performance evaluation
- Random state initialization for reproducibility

### Model Evaluation

- Comprehensive metrics calculation: accuracy, precision, recall, F1 score, ROC AUC
- Confusion matrix visualization for each model
- ROC curve plotting with AUC calculation
- Feature importance analysis and visualization
- Statistical significance testing of features using F-scores

### Visualization

- Distribution plots for key numerical features
- Count plots for categorical variables by churn status
- Correlation heatmaps to identify relationships between features
- Box plots comparing feature distributions across churn groups
- Feature importance bar charts from trained models

## Results

### Model Comparison

The evaluation results show performance metrics across multiple classification models:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.891213 | 0.800000 | 0.714286 | 0.754717 | 0.959407 |
| XGBoost | 0.887029 | 0.784314 | 0.714286 | 0.747664 | 0.962529 |
| Gradient Boosting | 0.882845 | 0.804348 | 0.660714 | 0.725490 | 0.965603 |
| SVM | 0.870293 | 0.765957 | 0.642857 | 0.699029 | 0.937354 |
| Logistic Regression | 0.836820 | 0.742857 | 0.464286 | 0.571429 | 0.848458 |

### Best Model

Random Forest was identified as the best performing model based on overall metrics. After hyperparameter tuning, the Random Forest model achieved the following performance:

- Accuracy: 0.8954
- Precision: 0.8163
- Recall: 0.7143
- F1 Score: 0.7619
- ROC AUC: 0.9626

The Random Forest model provides the best balance between precision and recall while maintaining high overall accuracy and excellent discrimination ability (ROC AUC).

### Feature Importance

Analysis reveals key predictors of customer churn:
- Interaction features (especially FreqXServices) show strong predictive power
- Service usage patterns are significant indicators of churn likelihood
- Account integration with social media demonstrates notable correlation with retention
- Age and income level provide demographic context to churn behaviors

## Outcome

### Best Performing Model

Random Forest classifier demonstrated superior performance across multiple metrics, particularly in balancing precision and recall while maintaining high accuracy and ROC AUC scores. The tuned version further improved these metrics, making it the optimal choice for churn prediction in this travel industry dataset.

## Future Work

- Implementation of advanced ensemble techniques
- Hyperparameter optimization through more extensive grid search
- Exploration of deep learning approaches for complex pattern recognition
- Development of customer segments for targeted retention strategies
- Time-series analysis to identify early warning signals of churn
- Deployment as a real-time prediction API

## Notes

- The preprocessing pipeline handles both numerical and categorical features appropriately
- Feature engineering significantly enhances model performance
- Missing value handling is implemented with domain-specific strategies
- Code includes comprehensive documentation and visualization for interpretability
- Solution prioritizes balanced metrics rather than just accuracy

## Contributing

Contributions to improve the model performance or extend the features are welcome. Please follow these steps:

## License

This project is licensed under the MIT License - see the LICENSE file for details.
