import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Set the paths to your CSV files
data_file_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\credit-scoring-model-bati-bank\data\data.csv'
variable_definitions_file_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\credit-scoring-model-bati-bank\data\Xente_Variable_Definitions.csv'

# Load data
try:
    data = pd.read_csv(data_file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{data_file_path}' was not found.")
    exit()

# Preview the data
print("Data Preview:")
print(data.head())

# Load variable definitions (optional)
try:
    variable_definitions = pd.read_csv(variable_definitions_file_path)
    print("Variable definitions loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{variable_definitions_file_path}' was not found.")
    variable_definitions = None

# Print the column names to check for 'FraudResult'
print("Column names in the dataset:")
print(data.columns)

# Step 1: Check if 'FraudResult' column exists
if 'FraudResult' not in data.columns:
    print("Error: The 'FraudResult' column is missing from the dataset.")
    exit()

# Step 1: Define a proxy variable for categorizing users
data['Risk_Category'] = data['FraudResult'].apply(lambda x: 'High Risk' if x == 1 else 'Low Risk')

# Step 2: Select observable features that correlate with the fraud result variable
# Convert categorical variables to numeric if necessary
data_numeric = pd.get_dummies(data.select_dtypes(include=['number']), drop_first=True)  # Keep only numeric columns and convert dummies

# Calculate the correlation matrix on numeric data only
correlation_matrix = data_numeric.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Selecting features with high correlation with the target variable 'FraudResult'
correlation_threshold = 0.3  # Adjust threshold as necessary
high_corr_features = correlation_matrix.index[abs(correlation_matrix['FraudResult']) > correlation_threshold].tolist()
print(f"High Correlation Features: {high_corr_features}")

# Define feature columns (using selected features)
X = data[high_corr_features].drop(columns=['FraudResult', 'Risk_Category'], errors='ignore')  # Exclude target and proxy variable
y = data['FraudResult']

# Encode categorical variables if necessary
X = pd.get_dummies(X)

# Step 1a: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 1b: Choose Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a directory to save models if it doesn't exist
model_dir = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\credit-scoring-model-bati-bank\saved_models'
os.makedirs(model_dir, exist_ok=True)

# Step 1c: Train the Models
best_models = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    best_models[model_name] = model

# Step 1d: Hyperparameter Tuning
# Define hyperparameter grids for tuning
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
}

# Hyperparameter tuning for Random Forest
print("Tuning Random Forest...")
grid_search_rf = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=5, scoring='roc_auc')
grid_search_rf.fit(X_train, y_train)
best_models['Random Forest'] = grid_search_rf.best_estimator_
print(f"Best parameters for Random Forest: {grid_search_rf.best_params_}")

# Hyperparameter tuning for Gradient Boosting
print("Tuning Gradient Boosting...")
grid_search_gb = GridSearchCV(GradientBoostingClassifier(), gb_param_grid, cv=5, scoring='roc_auc')
grid_search_gb.fit(X_train, y_train)
best_models['Gradient Boosting'] = grid_search_gb.best_estimator_
print(f"Best parameters for Gradient Boosting: {grid_search_gb.best_params_}")

# Step 2: Model Evaluation
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Save the model
    model_filename = os.path.join(model_dir, f"{model_name.replace(' ', '_')}.pkl")
    joblib.dump(model, model_filename)
    print(f"Model saved to: {model_filename}")

# Optional: Save the scaler for future use
scaler_filename = os.path.join(model_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved to: {scaler_filename}")
