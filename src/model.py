# File: C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/credit-scoring-model-bati-bank/src/model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from utils import assign_credit_score, predict_loan_amount
import os  # Import os module to manage directory paths

# File paths
data_path = r'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/credit-scoring-model-bati-bank/data/data.csv'
output_path = r'C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/credit-scoring-model-bati-bank/results/final_output.csv'

# Ensure the output directory exists
output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)

# Load data
data = pd.read_csv(data_path)

# Define target variable
data['RiskCategory'] = data['FraudResult'].apply(lambda x: 'High Risk' if x == 1 else 'Low Risk')

# Select observable features (hypothetical selection)
selected_features = ['Amount', 'Value', 'ProductCategory', 'ChannelId', 'ProviderId']

# Splitting data into features and target variable
X = data[selected_features].copy()  # Make a copy to avoid SettingWithCopyWarning
y = data['RiskCategory']

# Check for non-numeric columns and convert them to numeric using Label Encoding
for column in X.select_dtypes(include=['object']).columns:  # Select categorical columns
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicting probabilities for test data
risk_probabilities = model.predict_proba(X_test)[:, 1]  # Probability of being High Risk

# Create a DataFrame for the test set predictions
test_results = pd.DataFrame({
    'RiskProbability': risk_probabilities,
    'RiskCategory': y_test.values  # Use the actual values from the test set
})

# Apply credit score based on probabilities for the test set
test_results['CreditScore'] = [assign_credit_score(prob) for prob in test_results['RiskProbability']]

# Predict loan amount and duration for the test set
test_results['LoanAmount'], test_results['LoanDuration'] = zip(*test_results['CreditScore'].apply(predict_loan_amount))

# Combine test results back into the original data DataFrame
data = data.merge(test_results[['RiskProbability', 'CreditScore', 'LoanAmount', 'LoanDuration']],
                  left_index=True, right_index=True, how='left')

# Save results
data.to_csv(output_path, index=False)

print("Model training and predictions completed successfully. Results saved to:", output_path)
