import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Define preprocessing function for both train and test datasets
def preprocess_data(df):
    # Remove underscores in column names
    df.columns = df.columns.str.replace('_', '')

    # Check for the expected columns before processing
    expected_columns = [
        'IncomeAnnual', 'BaseSalaryPerMonth', 'RateOfInterest', 
        'CreditLimit', 'CurrentDebtOutstanding', 'RatioCreditUtilization', 
        'PerMonthEMI', 'MonthlyInvestment', 'MonthlyBalance', 
        'Age', 'TotalBankAccounts', 'TotalCurrentLoans', 
        'Delayfromduedate', 'TotalDelayedPayments', 'CreditHistoryAge',
        'LoanType'
    ]
    
    for col in expected_columns:
        if col not in df.columns:
            print(f"Warning: Expected column '{col}' is missing from the DataFrame.")
            # Handle the case as necessary (e.g., skip, fill with NaN, etc.)

    # Convert columns to numeric and handle non-numeric entries as NaN
    columns_to_convert = [
        'IncomeAnnual', 'BaseSalaryPerMonth', 'RateOfInterest', 
        'CreditLimit', 'CurrentDebtOutstanding', 'RatioCreditUtilization', 
        'PerMonthEMI', 'MonthlyInvestment', 'MonthlyBalance'
    ]
    int_columns_to_convert = [
        'Age', 'TotalBankAccounts', 'TotalCurrentLoans', 
        'Delayfromduedate', 'TotalDelayedPayments'
    ]
    
    for col in columns_to_convert + int_columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clip outliers
    def clip_outliers(df, column, lower=0.05, upper=0.95):
        lower_bound = df[column].quantile(lower)
        upper_bound = df[column].quantile(upper)
        df[column] = df[column].clip(lower_bound, upper_bound)
        
    for col in columns_to_convert + int_columns_to_convert:
        if col in df.columns:
            clip_outliers(df, col)
    
    # Convert 'CreditHistoryAge' to months and split into years and months
    def convert_years_months_to_months(age_str):
        if isinstance(age_str, str):
            years = int(age_str.split()[0])
            months = int(age_str.split()[3])
            return years * 12 + months
        return np.nan

    if 'CreditHistoryAge' in df.columns:
        df['CreditHistoryAgeMonths'] = df['CreditHistoryAge'].apply(convert_years_months_to_months)
    
    # Feature Engineering: Financial Ratios
    df['DebtIncomeRatio'] = df['CurrentDebtOutstanding'] / (df['IncomeAnnual'] + 1)
    df['IncomeCreditLimitRatio'] = df['IncomeAnnual'] / (df['CreditLimit'] + 1)
    df['DebtCreditLimitRatio'] = df['CurrentDebtOutstanding'] / (df['CreditLimit'] + 1)
    df['MonthlyBalanceToIncome'] = df['MonthlyBalance'] / (df['IncomeAnnual'] / 12 + 1)
    
    # Impute missing values using KNN for numerical and mode for categorical
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    
    # KNN Imputer for numerical features
    knn_imputer = KNNImputer()
    df[numerical_features] = knn_imputer.fit_transform(df[numerical_features])
    
    # Mode imputer for categorical features
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])
    
    return df

# Load the datasets with low_memory option
df_train = pd.read_csv("input/credit-dset/clean_trained.csv", low_memory=False)
df_test = pd.read_csv("input/credit-dset/test.csv", low_memory=False)

 
    

# Print the column names to verify
print("Train Columns:", df_train.columns.tolist())
print("Test Columns:", df_test.columns.tolist())

# df_train=df_train.drop(['Customer_ID'],axis=1)

# Apply preprocessing to both train and test data
# df_train = preprocess_data(df_train)
df_test = preprocess_data(df_test)

# Map the target variable in training data
credit_score_map = {'Poor': 0, 'Standard': 1, 'Good': 2}
df_train['Credit_Score'] = df_train['Credit_Score'].map(credit_score_map)

# Separate features and target
X = df_train.drop(columns='Credit_Score')
y = df_train['Credit_Score']

# Define preprocessor for pipeline
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
numerical_pipeline = Pipeline([('scaler', StandardScaler())])

categorical_features = X.select_dtypes(include=['object']).columns
categorical_pipeline = Pipeline([
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
    'classifier__max_depth': [6, 8],
    'classifier__n_estimators': [100, 200]
}

param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [6, 8]
}

# Perform grid search for both models
grid_search_xgb = GridSearchCV(xgb_pipeline, param_grid_xgb, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
grid_search_rf = GridSearchCV(rf_pipeline, param_grid_rf, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)

# Fit models and get best estimator
grid_search_xgb.fit(X, y)
grid_search_rf.fit(X, y)

# Choose the model with the best cross-validated score
if grid_search_xgb.best_score_ > grid_search_rf.best_score_:
    best_pipeline = grid_search_xgb.best_estimator_
    print("Using XGBoost with cross-validated accuracy:", grid_search_xgb.best_score_)
else:
    best_pipeline = grid_search_rf.best_estimator_
    print("Using RandomForest with cross-validated accuracy:", grid_search_rf.best_score_)

# Predict on the test set
test_predictions = best_pipeline.predict(df_test)
test_predictions_labels = pd.Series(test_predictions).map({v: k for k, v in credit_score_map.items()})

# Prepare the submission file
test_ids = df_test['ID'].copy()
submission = pd.DataFrame({'ID': test_ids, 'CreditScore': test_predictions_labels})
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
