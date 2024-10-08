import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load data
data_path = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\credit-scoring-model-bati-bank\data\data.csv"
data = pd.read_csv(data_path)

# Check the structure of the data
print("Data Structure:")
print(data.info())
print("Available columns in DataFrame:")
print(data.columns)

# Feature Engineering

# 1. Handle Missing Values
# Impute missing values for categorical columns with mode
categorical_cols = data.select_dtypes(include=['object']).columns
imputer_cat = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])

# Impute missing values for numerical columns with mean
numerical_cols = data.select_dtypes(include=[np.number]).columns
imputer_num = SimpleImputer(strategy='mean')
data[numerical_cols] = imputer_num.fit_transform(data[numerical_cols])

# 2. Create Aggregate Features
data['Total_Transaction_Amount'] = data.groupby('CustomerId')['Amount'].transform('sum')
data['Average_Transaction_Amount'] = data.groupby('CustomerId')['Amount'].transform('mean')
data['Transaction_Count'] = data.groupby('CustomerId')['TransactionId'].transform('count')
data['Std_Transaction_Amount'] = data.groupby('CustomerId')['Amount'].transform('std')

# 3. Extract Features from Transaction Start Time
data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
data['Transaction_Hour'] = data['TransactionStartTime'].dt.hour
data['Transaction_Day'] = data['TransactionStartTime'].dt.day
data['Transaction_Month'] = data['TransactionStartTime'].dt.month
data['Transaction_Year'] = data['TransactionStartTime'].dt.year

# 4. Define Proxy Variable for Risk Categorization
# Using FraudResult as the default indicator (1 for fraud, 0 for no fraud)
data['Risk_Category'] = np.where(data['FraudResult'] == 1, 'High Risk', 'Low Risk')

# 5. Weight of Evidence (WoE) Binning for Risk Assessment
def woe_binning(data, feature, target):
    """Function to calculate Weight of Evidence for a given feature."""
    bins = pd.qcut(data[feature], q=10, duplicates='drop')
    woe_df = pd.DataFrame(data.groupby(bins, observed=False)[target].agg(['count', 'sum']))
    woe_df['non_event'] = woe_df['count'] - woe_df['sum']

    # Avoid division by zero and handle cases with zero events
    total_events = woe_df['sum'].sum()
    total_non_events = woe_df['non_event'].sum()

    # Safeguard against division by zero
    woe_df['event_rate'] = np.where(total_events > 0, woe_df['sum'] / total_events, 1e-9)
    woe_df['non_event_rate'] = np.where(total_non_events > 0, woe_df['non_event'] / total_non_events, 1e-9)

    # Add small epsilon to avoid log(0)
    woe_df['woe'] = np.log((woe_df['event_rate'] + 1e-9) / (woe_df['non_event_rate'] + 1e-9))

    # Handle NaN values that might arise
    woe_df['woe'] = woe_df['woe'].replace([np.inf, -np.inf], np.nan)

    return woe_df[['woe']]

# Apply WOE Binning on 'Amount' as an example
data['woe_Amount'] = woe_binning(data, 'Amount', 'FraudResult')['woe']

# 6. Normalize/Standardize Numerical Features
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Save the transformed data
output_path = r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\credit-scoring-model-bati-bank\data\transformed_data.csv"
data.to_csv(output_path, index=False)

print("Transformed data saved to:", output_path)
