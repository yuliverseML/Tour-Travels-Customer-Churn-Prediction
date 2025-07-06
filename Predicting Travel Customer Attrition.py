###Tour & Travels Customer Churn Prediction - Complete Solution
#1. Data Loading and Initial Exploration


# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
data = pd.read_csv('Customertravel.csv')

# Display basic information
print("Dataset Overview:")
print(f"Shape: {data.shape}")
print("\nData Types:")
print(data.dtypes)

# Check for missing values in the original dataset
print("\nMissing values in original dataset:")
print(data.isnull().sum())

# Display basic statistics
print("\nBasic statistics for numerical features:")
print(data.describe())

# Check target distribution
print("\nTarget class distribution:")
print(data['Target'].value_counts())
print(f"Churn rate: {data['Target'].mean()*100:.2f}%")

# Display first few rows
print("\nSample data:")
print(data.head())


#2. Exploratory Data Analysis

# Create exploratory visualizations
plt.figure(figsize=(15, 10))

# Age distribution
plt.subplot(2, 3, 1)
sns.histplot(data=data, x='Age', kde=True)
plt.title('Age Distribution')

# Age vs Target
plt.subplot(2, 3, 2)
sns.boxplot(x='Target', y='Age', data=data)
plt.title('Age by Churn Status')

# Services Opted distribution
plt.subplot(2, 3, 3)
sns.countplot(data=data, x='ServicesOpted', hue='Target')
plt.title('Services Opted by Churn Status')

# Categorical features vs Target
plt.subplot(2, 3, 4)
sns.countplot(data=data, x='FrequentFlyer', hue='Target')
plt.title('Frequent Flyer Status by Churn')

plt.subplot(2, 3, 5)
sns.countplot(data=data, x='AccountSyncedToSocialMedia', hue='Target')
plt.title('Social Media Sync by Churn')

plt.subplot(2, 3, 6)
sns.countplot(data=data, x='BookedHotelOrNot', hue='Target')
plt.title('Hotel Booking by Churn')

plt.tight_layout()
plt.show()

# Income class analysis
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='AnnualIncomeClass', hue='Target')
plt.title('Annual Income Class by Churn')
plt.xticks(rotation=45)
plt.show()

# Create correlation heatmap for better understanding relationships
plt.figure(figsize=(8, 6))
# Convert categorical variables to numeric for correlation analysis
data_corr = data.copy()
for col in ['FrequentFlyer', 'AccountSyncedToSocialMedia', 'BookedHotelOrNot']:
    data_corr[col] = data_corr[col].map({'Yes': 1, 'No': 0, 'No Record': -1})
    
# Create ordinal encoding for income class
income_map = {'Low Income': 0, 'Middle Income': 1, 'High Income': 2}
data_corr['AnnualIncomeClass'] = data_corr['AnnualIncomeClass'].map(income_map)

# Generate correlation matrix
corr_matrix = data_corr.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()



#3. Data Cleaning and Missing Value Handling

# Create a clean copy of the data
df_clean = data.copy()

# Check for unique values in categorical variables
for col in ['FrequentFlyer', 'AnnualIncomeClass', 'AccountSyncedToSocialMedia', 'BookedHotelOrNot']:
    print(f"\n{col} value distribution:")
    print(df_clean[col].value_counts(dropna=False))

# Handle 'No Record' in FrequentFlyer - convert to standard 'No'
if 'No Record' in df_clean['FrequentFlyer'].unique():
    df_clean['FrequentFlyer'] = df_clean['FrequentFlyer'].replace('No Record', 'No')
    print("\nConverted 'No Record' to 'No' in FrequentFlyer column")

# Check for missing values after cleaning
print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

# Check for outliers using corrected IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Check outliers in numerical features
numerical_features = ['Age', 'ServicesOpted']
print("\nChecking for outliers using IQR method:")
for feature in numerical_features:
    outliers, lower_bound, upper_bound = detect_outliers_iqr(df_clean, feature)
    outlier_count = len(outliers)
    print(f"Outliers in {feature}: {outlier_count} ({outlier_count/len(df_clean)*100:.2f}%)")
    if outlier_count > 0:
        print(f"Acceptable range: {lower_bound:.2f} to {upper_bound:.2f}")
        print(f"Outlier values range: {outliers[feature].min()} to {outliers[feature].max()}")

# Check categorical variables for inconsistencies
print("\nCategorical variables after cleaning:")
for feature in ['FrequentFlyer', 'AnnualIncomeClass', 'AccountSyncedToSocialMedia', 'BookedHotelOrNot']:
    unique_values = df_clean[feature].unique()
    print(f"\n{feature} unique values: {unique_values}")
    print(df_clean[feature].value_counts())


#4. Feature Engineering
# Create a copy of the cleaned data for feature engineering
df_processed = df_clean.copy()

# Convert categorical variables to numeric
for col in ['FrequentFlyer', 'AccountSyncedToSocialMedia', 'BookedHotelOrNot']:
    df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})

# Encode Annual Income Class
income_map = {'Low Income': 0, 'Middle Income': 1, 'High Income': 2}
df_processed['IncomeLevel'] = df_processed['AnnualIncomeClass'].map(income_map)

# Create age groups for better segmentation
df_processed['AgeGroup'] = pd.cut(
    df_processed['Age'], 
    bins=[0, 25, 35, 45, 100], 
    labels=['Young', 'Adult', 'Middle-aged', 'Senior']
)

# Create interaction features
df_processed['FreqXServices'] = df_processed['FrequentFlyer'] * df_processed['ServicesOpted']
df_processed['SocialXServices'] = df_processed['AccountSyncedToSocialMedia'] * df_processed['ServicesOpted']
df_processed['HotelXServices'] = df_processed['BookedHotelOrNot'] * df_processed['ServicesOpted']

# Create ratio feature - safely handle potential division by zero
df_processed['ServicesPerAge'] = df_processed['ServicesOpted'] / df_processed['Age'].replace(0, 0.1)

# Check for any NaN values created during feature engineering
print("Checking for NaN values after feature engineering:")
print(df_processed.isna().sum())

# Print the new features
print("\nDataset after feature engineering (first 5 rows):")
print(df_processed.head())


#5. Feature Selection and Analysis
# Prepare dataset for modeling (drop original categorical and redundant features)
X = df_processed.drop(['Target', 'AnnualIncomeClass', 'AgeGroup'], axis=1)
y = df_processed['Target']

# Final verification for NaN values before modeling
print("Final check for NaN values in features:")
print(X.isna().sum())

# Feature correlation with target
print("\nFeature correlation with target:")
feature_corr = X.corrwith(y).sort_values(ascending=False)
print(feature_corr)

# Feature importance using Random Forest
from sklearn.ensemble import RandomForestClassifier

# Train a simple random forest for feature selection
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X, y)

# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_selector.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature importance from Random Forest:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Univariate feature selection
from sklearn.feature_selection import SelectKBest, f_classif

# Make sure there are no NaN values
X_clean = X.fillna(X.mean())
selector = SelectKBest(f_classif, k='all')
selector.fit(X_clean, y)
scores = pd.DataFrame({
    'Feature': X.columns,
    'F-Score': selector.scores_,
    'P-Value': selector.pvalues_
}).sort_values('F-Score', ascending=False)

print("\nUnivariate feature selection results:")
print(scores)


#6. Data Preparation for Modeling
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Define preprocessing for numerical and categorical features
numerical_features = ['Age', 'ServicesOpted', 'IncomeLevel', 
                      'FreqXServices', 'SocialXServices', 
                      'HotelXServices', 'ServicesPerAge']
categorical_features = ['FrequentFlyer', 'AccountSyncedToSocialMedia', 'BookedHotelOrNot']

# Create preprocessing steps with explicit missing value handling
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))
])

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Function to evaluate classification models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    # Print results
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    
    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Plot ROC curve if applicable
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

#7. Model Training and Evaluation
# Import classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Create model pipelines
models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    
    'Gradient Boosting': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ]),
    
    'SVM': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True, random_state=42))
    ]),
    
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(random_state=42))
    ])
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    results[name] = evaluate_model(model, X_train, X_test, y_train, y_test, name)

# Create model comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[model]['accuracy'] for model in results],
    'Precision': [results[model]['precision'] for model in results],
    'Recall': [results[model]['recall'] for model in results],
    'F1 Score': [results[model]['f1'] for model in results],
    'ROC AUC': [results[model]['roc_auc'] for model in results]
})

# Display model comparison
print("\nModel Performance Comparison:")
print(comparison_df.sort_values('F1 Score', ascending=False))

# Visualize model comparison
plt.figure(figsize=(12, 10))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
for i, metric in enumerate(metrics):
    plt.subplot(3, 2, i+1)
    sns.barplot(x='Model', y=metric, data=comparison_df)
    plt.title(f'Model Comparison - {metric}')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# Identify the best model based on F1 Score
best_model_name = comparison_df.sort_values('F1 Score', ascending=False).iloc[0]['Model']
best_model = results[best_model_name]['model']
print(f"\nBest model: {best_model_name}")


#8. Hyperparameter Tuning for Best Model
# Define hyperparameter grids for each model
param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['liblinear', 'saga']
    },
    'Random Forest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7]
    },
    'SVM': {
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 'auto']
    },
    'XGBoost': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }
}

# Get parameters for the best model
param_grid = param_grids[best_model_name]

# Perform grid search for the best model
print(f"Tuning hyperparameters for {best_model_name}...")
grid_search = GridSearchCV(
    best_model,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Display best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

# Evaluate the tuned model
tuned_model = grid_search.best_estimator_
tuned_results = evaluate_model(tuned_model, X_train, X_test, y_train, y_test, f"Tuned {best_model_name}")

# Compare original vs tuned model
comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
    'Original Model': [
        results[best_model_name]['accuracy'],
        results[best_model_name]['precision'],
        results[best_model_name]['recall'],
        results[best_model_name]['f1'],
        results[best_model_name]['roc_auc']
    ],
    'Tuned Model': [
        tuned_results['accuracy'],
        tuned_results['precision'],
        tuned_results['recall'],
        tuned_results['f1'],
        tuned_results['roc_auc']
    ]
})

print("\nOriginal vs Tuned Model Comparison:")
print(comparison)

# Visualize the comparison
plt.figure(figsize=(10, 6))
comparison_melted = pd.melt(comparison, id_vars='Metric', var_name='Model', value_name='Score')
sns.barplot(x='Metric', y='Score', hue='Model', data=comparison_melted)
plt.title(f'Performance Comparison: Original vs. Tuned {best_model_name}')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#9. Feature Importance Analysis
# Extract feature importance from the final model (if applicable)
if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    try:
        # Get the fitted classifier
        classifier = tuned_model.named_steps['classifier']
        
        # Get raw feature importances
        importances = classifier.feature_importances_
        
        # Apply preprocessor to get transformed feature names
        preprocessor = tuned_model.named_steps['preprocessor']
        preprocessor.fit(X_train)
        
        # Try to get feature names based on scikit-learn version
        try:
            # For newer scikit-learn versions with get_feature_names_out
            feature_names = preprocessor.get_feature_names_out()
        except:
            # For older versions - construct manually
            num_features = list(preprocessor.transformers_[0][2])
            
            # Get categorical feature names
            cat_features = []
            cat_cols = preprocessor.transformers_[1][2]
            for i, col in enumerate(cat_cols):
                # For binary categorical features (with drop='first'), only one column is created
                cat_features.append(f"{col}_1")
            
            feature_names = num_features + cat_features
        
        # Make sure feature_names and importances have same length
        if len(feature_names) == len(importances):
            # Create DataFrame with feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Display top 10 features
            print("\nTop 10 Feature Importance:")
            print(feature_importance.head(10))
            
            # Visualize feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
            plt.title(f'Top 10 Features - {best_model_name}')
            plt.tight_layout()
            plt.show()
        else:
            print(f"\nFeature names count ({len(feature_names)}) doesn't match importance count ({len(importances)})")
            print("Showing raw feature importance values:")
            for i, importance in enumerate(importances):
                print(f"Feature {i}: {importance:.6f}")
    except Exception as e:
        print(f"\nCould not extract detailed feature importance: {str(e)}")
        print("Showing raw feature importance values:")
        for i, importance in enumerate(classifier.feature_importances_):
            print(f"Feature {i}: {importance:.6f}")

#10. Save Model and Create Prediction Function
# Save the final model
import joblib

model_filename = 'travel_churn_prediction_model.pkl'
joblib.dump(tuned_model, model_filename)
print(f"\nFinal model saved as '{model_filename}'")

# Create a function for making predictions on new data
def predict_churn(model, customer_data):
    """
    Predict churn probability for new customer data
    
    Parameters:
    -----------
    model : trained model
        The trained churn prediction model
    customer_data : dict or DataFrame
        Customer data containing required features
    
    Returns:
    --------
    float
        Probability of customer churn (0-1)
    """
    # Convert to DataFrame if it's a dictionary
    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])
    
    # Make prediction
    churn_prob = model.predict_proba(customer_data)[:, 1]
    return churn_prob

# Example of using the prediction function
print("\nExample prediction on a new customer:")
new_customer = {
    'Age': 35,
    'FrequentFlyer': 0,  # No
    'AccountSyncedToSocialMedia': 1,  # Yes
    'BookedHotelOrNot': 0,  # No
    'ServicesOpted': 3,
    'IncomeLevel': 1,  # Middle Income
    'FreqXServices': 0,  # FrequentFlyer * ServicesOpted
    'SocialXServices': 3,  # AccountSyncedToSocialMedia * ServicesOpted
    'HotelXServices': 0,  # BookedHotelOrNot * ServicesOpted
    'ServicesPerAge': 3/35  # ServicesOpted / Age
}

churn_prob = predict_churn(tuned_model, new_customer)
print(f"Churn probability: {churn_prob[0]:.4f}")
print(f"Prediction: {'Likely to Churn' if churn_prob[0] > 0.5 else 'Not Likely to Churn'}")

# Another example with different characteristics
new_customer2 = {
    'Age': 28,
    'FrequentFlyer': 1,  # Yes
    'AccountSyncedToSocialMedia': 0,  # No
    'BookedHotelOrNot': 1,  # Yes
    'ServicesOpted': 5,
    'IncomeLevel': 2,  # High Income
    'FreqXServices': 5,  # FrequentFlyer * ServicesOpted
    'SocialXServices': 0,  # AccountSyncedToSocialMedia * ServicesOpted
    'HotelXServices': 5,  # BookedHotelOrNot * ServicesOpted
    'ServicesPerAge': 5/28  # ServicesOpted / Age
}

churn_prob2 = predict_churn(tuned_model, new_customer2)
print(f"\nChurn probability for second customer: {churn_prob2[0]:.4f}")
print(f"Prediction: {'Likely to Churn' if churn_prob2[0] > 0.5 else 'Not Likely to Churn'}")            
