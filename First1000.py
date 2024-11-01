import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load training and test data
train_data = pd.read_csv('input/credit-dset/train.csv', low_memory=False)
test_data = pd.read_csv('input/credit-dset/test.csv', low_memory=False)
test_ids = pd.read_csv('input/credit-dset/test.csv', low_memory=False)['ID']

# Drop unnecessary columns
train_data.drop(columns=['Unnamed: 0', 'ID', 'Customer_ID', 'Month', 'Name', 'Number'], inplace=True, errors='ignore')
test_data.drop(columns=['ID', 'Customer_ID', 'Month', 'Name', 'Number'], inplace=True, errors='ignore')

# Convert relevant columns to numeric after removing any underscores
for col in ['Current_Debt_Outstanding', 'Income_Annual', 'Credit_Limit', 'Age']:
    train_data[col] = pd.to_numeric(train_data[col].astype(str).str.replace('_', '', regex=False), errors='coerce')
    test_data[col] = pd.to_numeric(test_data[col].astype(str).str.replace('_', '', regex=False), errors='coerce')

# Fill missing values with median in train data
train_data.fillna(train_data.median(numeric_only=True), inplace=True)

# Feature engineering in train data
train_data['Debt_Income_Ratio'] = train_data['Current_Debt_Outstanding'] / train_data['Income_Annual']
train_data['Income_Credit_Limit_Ratio'] = train_data['Income_Annual'] / train_data['Credit_Limit']
train_data['Debt_Credit_Limit_Ratio'] = train_data['Current_Debt_Outstanding'] / train_data['Credit_Limit']

# Replace infinity values in train data
train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
train_data.fillna(train_data.median(numeric_only=True), inplace=True)

# Label encode the target variable in train data
label_encoder = LabelEncoder()
train_data['Credit_Score'] = label_encoder.fit_transform(train_data['Credit_Score'])

# Prepare training features and labels
X_train = train_data.drop(columns='Credit_Score')
y_train = train_data['Credit_Score']

# Identify numerical and categorical columns in train data
numerical_features = X_train.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# Define preprocessing pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Create column transformer for preprocessing
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Define model pipeline
model = XGBClassifier(learning_rate=0.05, max_depth=6, n_estimators=300, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])

# Fit the pipeline on training data
pipeline.fit(X_train, y_train)

# Prepare test data
test_data.fillna(train_data.median(numeric_only=True), inplace=True)

# Feature engineering in test data
test_data['Debt_Income_Ratio'] = test_data['Current_Debt_Outstanding'] / test_data['Income_Annual']
test_data['Income_Credit_Limit_Ratio'] = test_data['Income_Annual'] / test_data['Credit_Limit']
test_data['Debt_Credit_Limit_Ratio'] = test_data['Current_Debt_Outstanding'] / test_data['Credit_Limit']

# Replace infinity values in test data
test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
test_data.fillna(train_data.median(numeric_only=True), inplace=True)

# Make predictions on test data
test_predictions = pipeline.predict(test_data)


# Convert predictions back to original labels
test_predictions_labels = label_encoder.inverse_transform(test_predictions)
# test_predictions_encoded = label_encoder.transform(test_predictions)

# Prepare the submission file
submission = pd.DataFrame({'ID': test_ids, 'Credit_Score': test_predictions_labels})
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
