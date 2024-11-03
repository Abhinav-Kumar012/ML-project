import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load datasets
train_df = pd.read_csv('input/credit-dset/clean_trained.csv')
test_df = pd.read_csv('input/credit-dset/test_cleaned_final.csv')

# Encode the target variable in the training data
label_encoder = LabelEncoder()
train_df['Credit_Score'] = label_encoder.fit_transform(train_df['Credit_Score'])

# Columns to label encode
label_encode_cols = ['Month', 'Profession', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']

# Combine train and test data for encoding
combined_df = pd.concat([train_df[label_encode_cols], test_df[label_encode_cols]])

# Apply label encoding to each specified column
for col in label_encode_cols:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Define features (X) and target (y)
X = train_df.drop(columns=['Credit_Score', 'Number'])  # Exclude target and unnecessary columns
y = train_df['Credit_Score']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify remaining categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Set up column transformer for one-hot encoding of remaining categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ], remainder='passthrough')

# Define parameter grids for RandomizedSearchCV
rf_param_grid = {
 
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

xgb_param_grid = {
    'classifier__n_estimators': np.arange(100, 1000, 100),
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7, 9, 11],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0]
}

# Define pipelines for both models
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                              ('classifier', RandomForestClassifier(random_state=42))])

xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                               ('classifier', XGBClassifier(eval_metric='mlogloss', random_state=42))])

# Perform RandomizedSearchCV for RandomForest
rf_search = RandomizedSearchCV(rf_pipeline, rf_param_grid, n_iter=10, scoring='accuracy', cv=3, random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)
best_rf_pipeline = rf_search.best_estimator_
rf_best_accuracy = accuracy_score(y_val, best_rf_pipeline.predict(X_val))
print(f"Tuned RandomForest Validation Accuracy: {rf_best_accuracy:.4f}")

# Perform RandomizedSearchCV for XGBoost
xgb_search = RandomizedSearchCV(xgb_pipeline, xgb_param_grid, n_iter=10, scoring='accuracy', cv=3, random_state=42, n_jobs=-1)
xgb_search.fit(X_train, y_train)
best_xgb_pipeline = xgb_search.best_estimator_
xgb_best_accuracy = accuracy_score(y_val, best_xgb_pipeline.predict(X_val))
print(f"Tuned XGBoost Validation Accuracy: {xgb_best_accuracy:.4f}")

# Choose the best model
best_model = best_xgb_pipeline if xgb_best_accuracy > rf_best_accuracy else best_rf_pipeline

# Prepare test data predictions for submission
test_features = test_df.drop(columns=['ID', 'Number'], errors='ignore')  # Exclude unnecessary columns
test_features_transformed = preprocessor.transform(test_features)
test_preds = best_model.predict(test_features_transformed)

# Create submission dataframe
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'Predicted': label_encoder.inverse_transform(test_preds)
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully.")
