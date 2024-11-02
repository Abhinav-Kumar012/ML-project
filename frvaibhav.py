import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df_train = pd.read_csv("input/credit-dset/train.csv")
df_test = pd.read_csv("input/credit-dset/test.csv")

# Columns for numerical conversion
columns_to_convert = [
    'Income_Annual', 'Base_Salary_PerMonth', 'Rate_Of_Interest', 
    'Credit_Limit', 'Current_Debt_Outstanding', 'Ratio_Credit_Utilization', 
    'Per_Month_EMI', 'Monthly_Investment', 'Monthly_Balance'
]
int_columns_to_convert = [
    'Age', 'Total_Bank_Accounts', 'Rate_Of_Interest', 
    'Total_Current_Loans', 'Delay_from_due_date', 'Total_Delayed_Payments'
]

# Convert columns to numeric and handle invalid entries
df_train[columns_to_convert] = df_train[columns_to_convert].apply(pd.to_numeric, errors='coerce')
df_train[int_columns_to_convert] = df_train[int_columns_to_convert].apply(pd.to_numeric, errors='coerce').astype('Int64')

# Apply similar conversions to the test set
df_test[columns_to_convert] = df_test[columns_to_convert].apply(pd.to_numeric, errors='coerce')
df_test[int_columns_to_convert] = df_test[int_columns_to_convert].apply(pd.to_numeric, errors='coerce').astype('Int64')

# Fix outliers and invalid values in train set
df_train['Age'] = df_train['Age'].apply(lambda x: x if 0 <= x <= 100 else np.nan)
df_train['Total_Bank_Accounts'] = df_train['Total_Bank_Accounts'].apply(lambda x: x if x > 0 else np.nan)
df_train['Total_Credit_Cards'] = df_train['Total_Credit_Cards'].apply(lambda x: x if x > 0 else np.nan)
df_train['Rate_Of_Interest'] = df_train['Rate_Of_Interest'].apply(lambda x: x if x >= 0 else np.nan)

# Generate new features
for dataset in [df_train, df_test]:
    dataset['Debt_Income_Ratio'] = dataset['Current_Debt_Outstanding'] / dataset['Income_Annual']
    dataset['Income_Credit_Limit_Ratio'] = dataset['Income_Annual'] / dataset['Credit_Limit']
    dataset['Debt_Credit_Limit_Ratio'] = dataset['Current_Debt_Outstanding'] / dataset['Credit_Limit']
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

# Impute missing values with KNN Imputer
numerical_features = df_train.select_dtypes(include=['float64', 'int64']).columns
knn_imputer = KNNImputer(n_neighbors=5)
df_train[numerical_features] = knn_imputer.fit_transform(df_train[numerical_features])
df_test[numerical_features] = knn_imputer.transform(df_test[numerical_features])

# Encode target variable
label_encoder = LabelEncoder()
df_train['Credit_Score'] = label_encoder.fit_transform(df_train['Credit_Score'])

# Split data into training and validation sets
X = df_train.drop(columns='Credit_Score')
y = df_train['Credit_Score']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define preprocessor for the pipeline
numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

categorical_features = X_train.select_dtypes(include=['object']).columns
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Initialize classifiers
xgb_model = XGBClassifier(learning_rate=0.05, max_depth=6, n_estimators=300, random_state=4, eval_metric='mlogloss')
rf_model = RandomForestClassifier(random_state=42)

# Define the pipeline for each classifier
xgb_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', xgb_model)])
rf_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', rf_model)])

# Parameter grid for tuning
param_grid_xgb = {
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_depth': [6],
    'classifier__n_estimators': [100, 300]
}

param_grid_rf = {
    'classifier__n_estimators': [100, 300],
    'classifier__max_depth': [6, 10]
}

# Perform grid search for both models
grid_search_xgb = GridSearchCV(xgb_pipeline, param_grid_xgb, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
grid_search_rf = GridSearchCV(rf_pipeline, param_grid_rf, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)

# Fit models and get best estimator
grid_search_xgb.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)

# Evaluate both models on the validation set
xgb_val_predictions = grid_search_xgb.best_estimator_.predict(X_val)
rf_val_predictions = grid_search_rf.best_estimator_.predict(X_val)
xgb_val_accuracy = accuracy_score(y_val, xgb_val_predictions)
rf_val_accuracy = accuracy_score(y_val, rf_val_predictions)

# Choose the model with the best validation accuracy
if xgb_val_accuracy > rf_val_accuracy:
    best_pipeline = grid_search_xgb.best_estimator_
    print("Using XGBoost with accuracy:", xgb_val_accuracy)
else:
    best_pipeline = grid_search_rf.best_estimator_
    print("Using RandomForest with accuracy:", rf_val_accuracy)

# Predict on the test set
test_predictions = best_pipeline.predict(df_test)
test_predictions_labels = label_encoder.inverse_transform(test_predictions)

# Prepare the submission file
test_ids = df_test['ID'].copy()
submission = pd.DataFrame({'ID': test_ids, 'Credit_Score': test_predictions_labels})
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
