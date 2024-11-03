import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Function to preprocess data
def preprocess_credit_data(df):
    # String to number conversion function for Credit_History_Age
    def convert_to_months(age_str):
        if pd.isna(age_str):
            return np.nan
        else: 
            parts = age_str.split(' and ')
            years = int(parts[0].split()[0])
            months = int(parts[1].split()[0])
            total_months = (years * 12) + months
            return total_months

    # Drop unnecessary columns and retain ID for submission use
    id_column = df['ID']
    df = df.drop(['Name', 'Loan_Type', 'ID'], axis=1, errors='ignore')

    # Convert various columns to numeric
    df['Base_Salary_PerMonth'] = pd.to_numeric(df['Base_Salary_PerMonth'], downcast='float', errors='coerce')
    df['Total_Delayed_Payments'] = df['Total_Delayed_Payments'].str.replace(r'[^-0-9]', '', regex=True)
    df['Total_Delayed_Payments'] = pd.to_numeric(df['Total_Delayed_Payments'], downcast='float', errors='coerce')
    df['Credit_History_Age'] = df['Credit_History_Age'].apply(convert_to_months)
    df['Credit_History_Age'] = pd.to_numeric(df['Credit_History_Age'], downcast='float', errors='coerce')
    df['Age'] = df['Age'].str.replace(r'[^-0-9]', '', regex=True)
    df['Age'] = pd.to_numeric(df['Age'], downcast='integer', errors='coerce')
    df['Income_Annual'] = df['Income_Annual'].str.replace(r'[^-.0-9]', '', regex=True)
    df['Income_Annual'] = pd.to_numeric(df['Income_Annual'], downcast='float', errors='coerce')
    df['Total_Current_Loans'] = df['Total_Current_Loans'].str.replace(r'[^-0-9]', '', regex=True)
    df['Total_Current_Loans'] = pd.to_numeric(df['Total_Current_Loans'], downcast='integer', errors='coerce')
    df['Current_Debt_Outstanding'] = df['Current_Debt_Outstanding'].str.replace(r'[^-.0-9]', '', regex=True)
    df['Current_Debt_Outstanding'] = pd.to_numeric(df['Current_Debt_Outstanding'], downcast='float', errors='coerce')
    df['Credit_Limit'] = pd.to_numeric(df['Credit_Limit'], downcast='float', errors='coerce')
    df['Monthly_Balance'] = pd.to_numeric(df['Monthly_Balance'], downcast='float', errors='coerce')
    df['Monthly_Investment'] = df['Monthly_Investment'].str.replace(r'[^-.0-9]', '', regex=True)
    df['Monthly_Investment'] = pd.to_numeric(df['Monthly_Investment'], downcast='float', errors='coerce')

    # Handle categorical columns with unknown values
    df['Payment_of_Min_Amount'].replace("NM", np.nan, inplace=True)
    df['Credit_Mix'].replace("_", np.nan, inplace=True)
    df['Profession'].replace("_", np.nan, inplace=True)
    df['Number'].replace("#F%$D@*&8", np.nan, inplace=True)
    df['Payment_Behaviour'].replace("!@9#%8", np.nan, inplace=True)

    # Impute missing values with mode for categorical columns
    for col in ['Payment_of_Min_Amount', 'Credit_Mix', 'Profession', 'Number', 'Payment_Behaviour']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Drop columns with more than 5% missing values
    threshold = df.shape[0] * 0.05
    df.dropna(thresh=threshold, axis=1, inplace=True)

    # Impute remaining null values with mean for numeric columns
    for column in df.select_dtypes(include=['float', 'int']).columns:
        df[column].fillna(df[column].mean(), inplace=True)

    return df, id_column

# Load data
df_train = pd.read_csv("input/credit-dset/train.csv")
df_test = pd.read_csv("input/credit-dset/test.csv")

# Preprocess both train and test datasets
df_train, _ = preprocess_credit_data(df_train)
df_test, test_ids = preprocess_credit_data(df_test)

# Map target variable
credit_score_map = {'Poor': 0, 'Standard': 1, 'Good': 2}
df_train['Credit_Score'] = df_train['Credit_Score'].map(credit_score_map)

# Separate features and target
X = df_train.drop(columns='Credit_Score')
y = df_train['Credit_Score']

# Define preprocessors and pipelines
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Define classifiers
xgb_model = XGBClassifier(random_state=4, eval_metric='mlogloss')
rf_model = RandomForestClassifier(random_state=42)

# Define pipelines
xgb_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', xgb_model)])
rf_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', rf_model)])

# Parameter grids for tuning
param_grid_xgb = {
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__max_depth': [6, 8],
    'classifier__n_estimators': [ 200, 300],
    'classifier__min_child_weight': [3, 5],
}

param_grid_rf = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [6, 8],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [2, 4],

}

# Use StratifiedKFold for more stable results with class imbalance
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Use RandomizedSearchCV to expand the parameter search space
grid_search_xgb = RandomizedSearchCV(xgb_pipeline, param_grid_xgb, scoring='accuracy', cv=stratified_cv, n_jobs=-1, verbose=1, n_iter=3)
grid_search_rf = RandomizedSearchCV(rf_pipeline, param_grid_rf, scoring='accuracy', cv=stratified_cv, n_jobs=-1, verbose=1, n_iter=3)

# Fit both models on the training data
grid_search_xgb.fit(X, y)
grid_search_rf.fit(X, y)

# Select the best model based on cross-validated score
best_pipeline = grid_search_xgb.best_estimator_ if grid_search_xgb.best_score_ > grid_search_rf.best_score_ else grid_search_rf.best_estimator_

# Predict on the test set
test_predictions = best_pipeline.predict(df_test)
test_predictions_labels = pd.Series(test_predictions).map({v: k for k, v in credit_score_map.items()})

# Prepare the submission file
submission = pd.DataFrame({'ID': test_ids, 'CreditScore': test_predictions_labels})
submission_file_path = 'submission.csv'
submission.to_csv(submission_file_path, index=False)

# Return values for verification
submission_file_path, best_pipeline, grid_search_xgb.best_score_, grid_search_rf.best_score_
