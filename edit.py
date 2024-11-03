# %%
import numpy as np
import pandas as pd 

# %%
df_train = pd.read_csv("../input/credit-dset/train.csv")

# %%
# string -> no of months
def convert_to_months(age_str):
    if pd.isna(age_str):
        return np.nan
    else : 
        parts = age_str.split(' and ')
        years = int(parts[0].split()[0])
        months = int(parts[1].split()[0])
        total_months = (years * 12) + months
        return total_months

# %%
loan_type_col = df_train['Loan_Type']
# dropping columns
df_train = df_train.drop(['Name','Loan_Type','ID'], axis=1)
# base salary -> number
df_train['Base_Salary_PerMonth'] = pd.to_numeric(df_train['Base_Salary_PerMonth'],downcast = 'float',errors = 'coerce')
# Total delayed payments (removing underscores if present) -> number
df_train['Total_Delayed_Payments'] = df_train['Total_Delayed_Payments'].str.replace(r'[^-0-9]', '', regex=True)
df_train['Total_Delayed_Payments'] = pd.to_numeric(df_train['Total_Delayed_Payments'],downcast = 'float',errors = 'coerce')
# credit history age -> number (to no of months)
df_train['Credit_History_Age'] = df_train['Credit_History_Age'].apply(convert_to_months)
df_train['Credit_History_Age'] = pd.to_numeric(df_train['Credit_History_Age'],downcast = 'float',errors = 'coerce')
#age -> number
df_train['Age'] = df_train['Age'].str.replace(r'[^-0-9]', '', regex=True)
df_train['Age'] = pd.to_numeric(df_train['Age'],downcast = 'integer',errors = 'coerce')
#Income_annual -> number
df_train['Income_Annual'] = df_train['Income_Annual'].str.replace(r'[^-.0-9]', '', regex=True)
df_train['Income_Annual'] = pd.to_numeric(df_train['Income_Annual'],downcast = 'float',errors = 'coerce')
#Total_Current_Loans -> number
df_train['Total_Current_Loans'] = df_train['Total_Current_Loans'].str.replace(r'[^-0-9]', '', regex=True)
df_train['Total_Current_Loans'] = pd.to_numeric(df_train['Total_Current_Loans'],downcast = 'integer',errors = 'coerce')
#Current_Debt_Outstanding -> number
df_train['Current_Debt_Outstanding'] = df_train['Current_Debt_Outstanding'].str.replace(r'[^-.0-9]', '', regex=True)
df_train['Current_Debt_Outstanding'] = pd.to_numeric(df_train['Current_Debt_Outstanding'],downcast = 'float',errors = 'coerce')
#Credit_Limit -> number
df_train['Credit_Limit'] = pd.to_numeric(df_train['Credit_Limit'],downcast = 'float',errors = 'coerce')
#Monthly_Balance -> number
df_train['Monthly_Balance'] = pd.to_numeric(df_train['Monthly_Balance'],downcast = 'float',errors = 'coerce')

# %%
print(df_train.info(),end = "\n\n")
col = "Credit_Limit" #Monthly_Balance
print(df_train[pd.to_numeric(df_train[col],downcast = 'float', errors="coerce").isna()][col].value_counts())
#print(df_train[col].value_counts())

# %%
df_train.drop_duplicates(inplace=True)
print(df_train.isna().sum().to_string())
print(df_train.shape)

# %%
unknown_min_pay_repl = df_train['Payment_of_Min_Amount'].mode()[0]
df_train['Payment_of_Min_Amount'] = df_train.groupby("Customer_ID")['Payment_of_Min_Amount'].transform(
    lambda x: x.where(x != "NM", x[x != "NM"].mode().get(0, unknown_min_pay_repl))
)
df_train['Payment_of_Min_Amount'].value_counts()

# %%
unknown_mix_repl = df_train['Credit_Mix'].mode()[0]
df_train['Credit_Mix'] = df_train.groupby("Customer_ID")['Credit_Mix'].transform(
    lambda x: x.where(x != "_", x[x != "_"].mode().get(0, unknown_mix_repl))
)
df_train['Credit_Mix'].value_counts()

# %%
unknown_prof_repl = 'Lawyer'
df_train['Profession'] = df_train.groupby("Customer_ID")['Profession'].transform(
    lambda x: x.where(x != "_______", x[x != "_______"].mode().get(0, unknown_prof_repl))
)
df_train['Profession'].value_counts()

# %%
unknown_number_repl = '000-00-0000'
df_train['Number'] = df_train.groupby("Customer_ID")['Number'].transform(
    lambda x: x.where(x != "#F%$D@*&8", x[x != "#F%$D@*&8"].mode().get(0, unknown_number_repl))
)
df_train['Number'].value_counts()

# %%
unknown_behavior_repl = df_train['Payment_Behaviour'].mode()[0]
df_train['Payment_Behaviour'] = df_train.groupby("Customer_ID")['Payment_Behaviour'].transform(
    lambda x: x.where(x != "!@9#%8", x[x != "!@9#%8"].mode().get(0, unknown_behavior_repl))
)
df_train['Payment_Behaviour'].value_counts()

# %%
null_percentages=(df_train.isna().sum()/df_train.shape[0])*100
null_cols = null_percentages.loc[null_percentages > 0]
null_cols

# %%
rows_to_drop = null_cols.loc[null_cols < 5]
df_train.dropna(subset = rows_to_drop.keys(),inplace=True,how='any',axis=0)
print(df_train.isna().sum().to_string())
print(df_train.shape)

# %%
columns_to_drop = null_cols.loc[null_cols > 40]
df_train.drop(columns = columns_to_drop.keys(),inplace = True)
df_train.drop_duplicates(inplace=True)
print(df_train.shape)

# %%
null_percentages=(df_train.isna().sum()/df_train.shape[0])*100
null_cols = null_percentages.loc[null_percentages > 0]
print(null_cols,end = "\n\n")
col_impute = null_cols.loc[(null_cols >= 5) & (null_cols < 40)]
for column in col_impute.keys():
    central_tend = df_train[column].mean()
    df_train[column] = df_train[column].fillna(central_tend)

# %%
df_train['Monthly_Investment'] = df_train['Monthly_Investment'].str.replace(r'[^-.0-9]', '', regex=True)
df_train['Monthly_Investment'] = pd.to_numeric(df_train['Monthly_Investment'],downcast = 'float',errors = 'coerce')
df_train = df_train.drop(['Customer_ID'], axis=1)
print(df_train.info())

# %%
df_train.to_csv('clean_trained.csv',index = False)


