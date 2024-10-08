# File: C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/credit-scoring-model-bati-bank/src/utils.py

def assign_credit_score(prob):
    """
    Assign credit score based on risk probability.
    """
    if prob >= 0.75:
        return 'Poor'
    elif prob >= 0.5:
        return 'Fair'
    elif prob >= 0.25:
        return 'Good'
    else:
        return 'Excellent'

def predict_loan_amount(score):
    """
    Predict loan amount and duration based on credit score.
    """
    if score == 'Poor':
        return 100, 6
    elif score == 'Fair':
        return 500, 12
    elif score == 'Good':
        return 1000, 24
    else:
        return 2000, 36
