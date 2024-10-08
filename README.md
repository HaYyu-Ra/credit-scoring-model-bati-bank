# Credit Scoring Model for Bati Bank

## Overview
As an Analytics Engineer at Bati Bank, this project aims to develop a Credit Scoring Model to enable a buy-now-pay-later service in partnership with a successful eCommerce company. This service allows customers to purchase products on credit, provided they qualify. The model will assess creditworthiness by predicting the likelihood of default based on historical transaction data.

## Business Need
Credit scoring assigns a quantitative measure to potential borrowers, estimating the likelihood of future default. Traditionally, creditors use statistical techniques to analyze past borrower information related to their loan performance. The resulting model evaluates new applicants based on similar data, providing either a score reflecting their creditworthiness or a prediction of potential default.

The primary objectives of this project include:

1. Defining a proxy variable to categorize users as high risk (bad) or low risk (good).
2. Selecting observable features that are strong predictors of the defined default variable.
3. Developing a model that assigns risk probabilities for new customers.
4. Creating a model to assign credit scores from risk probability estimates.
5. Predicting the optimal loan amount and duration.

## Data and Features
The dataset for this challenge is available at [Xente Challenge on Kaggle](https://www.kaggle.com/). The data fields include:

- **TransactionId**: Unique transaction identifier on the platform.
- **BatchId**: Unique number assigned to a batch of transactions for processing.
- **AccountId**: Unique number identifying the customer on the platform.
- **SubscriptionId**: Unique number identifying the customer subscription.
- **CustomerId**: Unique identifier attached to the account.
- **CurrencyCode**: Country currency.
- **CountryCode**: Numerical geographical code of the country.
- **ProviderId**: Source provider of the item bought.
- **ProductId**: Item name being bought.
- **ProductCategory**: Broader categories for ProductIds.
- **ChannelId**: Identifies customer channel (web, Android, iOS, etc.).
- **Amount**: Value of the transaction.
- **Value**: Absolute value of the amount.
- **TransactionStartTime**: Transaction start time.
- **PricingStrategy**: Category of pricing structure for merchants.
- **FraudResult**: Fraud status of transaction (1 for yes, 0 for no).

## Project Tasks

### Task 1: Understanding Credit Risk
- Focus on understanding the concept of credit risk and its implications for the bank.
- Key references for understanding credit risk modeling:
  - [Credit Scoring Approaches](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf)
  - [Alternative Credit Scoring](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)

### Task 2: Exploratory Data Analysis (EDA)
1. **Overview of the Data**: Understand dataset structure.
2. **Summary Statistics**: Analyze central tendency and dispersion.
3. **Distribution of Numerical Features**: Visualize distributions to identify patterns.
4. **Distribution of Categorical Features**: Analyze frequency and variability.
5. **Correlation Analysis**: Explore relationships between numerical features.
6. **Identifying Missing Values**: Determine missing data and imputation strategies.
7. **Outlier Detection**: Use box plots for outlier identification.

### Task 3: Feature Engineering
1. Create aggregate features (e.g., total transaction amount, average transaction amount).
2. Extract features from transaction timestamps (e.g., transaction hour, day, month, year).
3. Encode categorical variables using one-hot or label encoding.
4. Handle missing values through imputation or removal.
5. Normalize or standardize numerical features for consistent scaling.

### Task 4: Modeling
1. **Model Selection and Training**:
   - Split data into training and testing sets.
   - Choose at least two models (e.g., Logistic Regression, Random Forest).
   - Train models and perform hyperparameter tuning.
   
2. **Model Evaluation**:
   - Assess performance using metrics: accuracy, precision, recall, F1 score, and ROC-AUC.

### Task 5: Model Serving API Call
- Create a REST API for real-time predictions.
- Load the trained model and define API endpoints.
- Implement logic for preprocessing input data and returning predictions.
- Deploy the API on a web server or cloud platform.

## Conclusion
This Credit Scoring Model will provide Bati Bank with the necessary tools to assess customer creditworthiness effectively and facilitate a seamless buy-now-pay-later experience for customers.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
