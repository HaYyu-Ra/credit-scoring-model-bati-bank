from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Paths to models and scaler
model_paths = {
    "logistic_regression": r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\credit-scoring-model-bati-bank\saved_models\Logistic_Regression.pkl',
    "decision_tree": r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\credit-scoring-model-bati-bank\saved_models\Decision_Tree.pkl',
    "random_forest": r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\credit-scoring-model-bati-bank\saved_models\Random_Forest.pkl',
    "gradient_boosting": r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\credit-scoring-model-bati-bank\saved_models\Gradient_Boosting.pkl'
}

# Path to the scaler
scaler_path = r'C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\credit-scoring-model-bati-bank\saved_models\scaler.pkl'

# Load models and scaler
models = {name: joblib.load(path) for name, path in model_paths.items()}
scaler = joblib.load(scaler_path)

# Request body structure for input data
class InputData(BaseModel):
    Amount: float
    Value: float
    PricingStrategy: int

@app.post('/predict/{model_name}')
def predict(model_name: str, input_data: list[InputData]):
    """
    Predict the outcome using the specified model.
    
    - **model_name**: The name of the model to use for prediction.
    - **input_data**: A list of input data objects.
    """
    # Check if the model name is valid
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found. Available models: {', '.join(models.keys())}")

    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([item.dict() for item in input_data])
        
        # Update features based on what the model was trained on
        # If the model was not trained on 'PricingStrategy', exclude it
        features = ['Amount', 'Value']  # Adjust based on your training set

        # Extract and scale the features
        input_features = input_df[features]
        input_scaled = scaler.transform(input_features)
        
        # Get the selected model
        model = models[model_name]

        # Perform prediction
        predictions = model.predict(input_scaled)

        # Get predicted probabilities if the model supports it
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_scaled)[:, 1]  # Positive class probability
        else:
            probabilities = None

        # Return predictions and probabilities (if available)
        response = {'predictions': predictions.tolist()}
        if probabilities is not None:
            response['probabilities'] = probabilities.tolist()

        return response
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")

@app.post('/predict/all')
def predict_all(input_data: list[InputData]):
    """
    Predict the outcome using all available models.

    - **input_data**: A list of input data objects.
    """
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([item.dict() for item in input_data])
        
        # Update features based on what the models were trained on
        features = ['Amount', 'Value']  # Adjust based on your training set

        # Extract and scale the features
        input_features = input_df[features]
        input_scaled = scaler.transform(input_features)

        results = {}
        
        # Iterate over all models to make predictions
        for model_name, model in models.items():
            predictions = model.predict(input_scaled)

            # Get predicted probabilities if the model supports it
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_scaled)[:, 1]  # Positive class probability
            else:
                probabilities = None

            # Store predictions and probabilities in the results dictionary
            results[model_name] = {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist() if probabilities is not None else None
            }

        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
