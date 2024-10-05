# File: C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/credit-scoring-model-bati-bank/src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="whitegrid")

# File paths
data_path = r'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/credit-scoring-model-bati-bank/data/data.csv'
variable_def_path = r'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/credit-scoring-model-bati-bank/data/Xente_Variable_Definitions.csv'

# Load data
data = pd.read_csv(data_path)

# 1. Overview of Data
print("=== Overview of Data ===")
print(f"Data Shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(data.info())

# 2. Summary Statistics
summary_stats = data.describe(include='all')  # Include categorical features as well
print("\nSummary Statistics:")
print(summary_stats)

# 3. Distribution of Numerical Features
numerical_features = ['Amount', 'Value']  # Ensure these columns exist in the dataset

for feature in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[feature], kde=True, bins=30)  # Add bins for better visualization
    plt.title(f'Distribution of {feature}', fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y')
    plt.show()

# 4. Distribution of Categorical Features
categorical_features = ['CurrencyCode', 'CountryCode', 'ProductCategory', 'FraudResult']

for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, data=data, order=data[feature].value_counts().index)  # Order by frequency
    plt.title(f'Distribution of {feature}', fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.grid(axis='y')
    plt.show()

# 5. Correlation Analysis
# Only compute the correlation matrix for numerical columns
numerical_data = data[numerical_features]  # Select only numerical features
correlation_matrix = numerical_data.corr()  # Compute correlation matrix

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix', fontsize=16)
plt.show()

# 6. Missing Value Identification
missing_values = data.isnull().sum()
print("\nMissing Values by Column:")
print(missing_values[missing_values > 0])  # Only print columns with missing values

# Heatmap of Missing Values
plt.figure(figsize=(12, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap', fontsize=16)
plt.show()

# 7. Outlier Detection
for feature in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=data[feature])
    plt.title(f'Boxplot of {feature}', fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.grid(axis='y')
    plt.show()
