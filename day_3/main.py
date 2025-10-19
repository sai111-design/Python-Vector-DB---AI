# Let's create a comprehensive example demonstrating the Day 3 task
# This will show the complete workflow for creating a /predict endpoint wrapping a dummy ML model

print("DAY 3 TASK DEMONSTRATION: CREATE /predict ENDPOINT WITH DUMMY ML MODEL")
print("=" * 70)

# Required imports
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.datasets import make_regression, make_classification
# from sklearn.model_selection import train_test_split
import joblib
import os

# Step 1: Create and train dummy ML models
print("\n1. CREATING AND TRAINING DUMMY ML MODELS")
print("-" * 50)

# Create dummy regression data
X_reg, y_reg = make_regression(n_samples=1000, n_features=4, noise=0.1, random_state=42)
feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']

print(f"Regression dataset created: {X_reg.shape}")
print(f"Features: {feature_names}")
print(f"Target range: {y_reg.min():.2f} to {y_reg.max():.2f}")

# Train dummy regressor
dummy_regressor = DummyRegressor(strategy='mean')
dummy_regressor.fit(X_reg, y_reg)

print(f"Dummy regressor trained - always predicts: {dummy_regressor.constant_[0]:.2f}")

# Create dummy classification data
X_cls, y_cls = make_classification(n_samples=1000, n_features=4, n_classes=3, 
                                   n_informative=3, random_state=42)
class_names = ['Class_A', 'Class_B', 'Class_C']

print(f"Classification dataset created: {X_cls.shape}")
print(f"Classes: {class_names}")
print(f"Class distribution: {np.bincount(y_cls)}")

# Train dummy classifier
dummy_classifier = DummyClassifier(strategy='most_frequent')
dummy_classifier.fit(X_cls, y_cls)

print(f"Dummy classifier trained - always predicts: Class_{dummy_classifier.classes_[np.argmax(dummy_classifier.class_prior_)]}")

# Save models
joblib.dump(dummy_regressor, 'dummy_regressor.pkl')
joblib.dump(dummy_classifier, 'dummy_classifier.pkl')

print("✓ Models saved as .pkl files")

# Step 2: Create sample FastAPI application structure
print("\n2. FASTAPI APPLICATION STRUCTURE")
print("-" * 50)

fastapi_code = '''
# main.py - FastAPI application with /predict endpoint

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import List, Dict, Any
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="ML Prediction API",
    description="A simple API for machine learning predictions using dummy models",
    version="1.0.0"
)

# Global variables to hold loaded models
dummy_regressor = None
dummy_classifier = None

# Pydantic models for request/response validation
class RegressionRequest(BaseModel):
    """Request model for regression predictions"""
    feature_1: float = Field(..., description="First feature value")
    feature_2: float = Field(..., description="Second feature value") 
    feature_3: float = Field(..., description="Third feature value")
    feature_4: float = Field(..., description="Fourth feature value")
    
    class Config:
        schema_extra = {
            "example": {
                "feature_1": 1.5,
                "feature_2": -0.8,
                "feature_3": 2.3,
                "feature_4": 0.1
            }
        }

class ClassificationRequest(BaseModel):
    """Request model for classification predictions"""
    feature_1: float = Field(..., description="First feature value")
    feature_2: float = Field(..., description="Second feature value")
    feature_3: float = Field(..., description="Third feature value") 
    feature_4: float = Field(..., description="Fourth feature value")
    
    class Config:
        schema_extra = {
            "example": {
                "feature_1": 0.5,
                "feature_2": -1.2,
                "feature_3": 1.8,
                "feature_4": -0.3
            }
        }

class RegressionResponse(BaseModel):
    """Response model for regression predictions"""
    prediction: float = Field(..., description="Predicted continuous value")
    model_type: str = Field(..., description="Type of model used")
    input_features: Dict[str, float] = Field(..., description="Input features used")

class ClassificationResponse(BaseModel):
    """Response model for classification predictions"""
    prediction: str = Field(..., description="Predicted class label")
    model_type: str = Field(..., description="Type of model used") 
    input_features: Dict[str, float] = Field(..., description="Input features used")
    
class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    models_loaded: bool = Field(..., description="Whether models are loaded")

# Startup event to load models
@app.on_event("startup")
async def load_models():
    """Load ML models on startup"""
    global dummy_regressor, dummy_classifier
    
    try:
        dummy_regressor = joblib.load('dummy_regressor.pkl')
        dummy_classifier = joblib.load('dummy_classifier.pkl')
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise

# Health check endpoint
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=dummy_regressor is not None and dummy_classifier is not None
    )

# Main prediction endpoints
@app.post("/predict/regression", response_model=RegressionResponse)
async def predict_regression(request: RegressionRequest):
    """
    Predict continuous values using dummy regressor
    
    This endpoint demonstrates how to wrap a simple ML model with FastAPI.
    The dummy regressor always predicts the mean of training data.
    """
    if dummy_regressor is None:
        raise HTTPException(status_code=503, detail="Regression model not loaded")
    
    try:
        # Convert request to numpy array
        features = np.array([[
            request.feature_1,
            request.feature_2, 
            request.feature_3,
            request.feature_4
        ]])
        
        # Make prediction
        prediction = dummy_regressor.predict(features)[0]
        
        return RegressionResponse(
            prediction=float(prediction),
            model_type="DummyRegressor",
            input_features={
                "feature_1": request.feature_1,
                "feature_2": request.feature_2,
                "feature_3": request.feature_3, 
                "feature_4": request.feature_4
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/classification", response_model=ClassificationResponse)
async def predict_classification(request: ClassificationRequest):
    """
    Predict class labels using dummy classifier
    
    This endpoint demonstrates classification with a dummy model.
    The dummy classifier always predicts the most frequent class.
    """
    if dummy_classifier is None:
        raise HTTPException(status_code=503, detail="Classification model not loaded")
    
    try:
        # Convert request to numpy array
        features = np.array([[
            request.feature_1,
            request.feature_2,
            request.feature_3,
            request.feature_4
        ]])
        
        # Make prediction
        prediction_idx = dummy_classifier.predict(features)[0]
        class_names = ['Class_A', 'Class_B', 'Class_C']
        prediction = class_names[prediction_idx]
        
        return ClassificationResponse(
            prediction=prediction,
            model_type="DummyClassifier",
            input_features={
                "feature_1": request.feature_1,
                "feature_2": request.feature_2, 
                "feature_3": request.feature_3,
                "feature_4": request.feature_4
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Additional utility endpoints
@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    if dummy_regressor is None or dummy_classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "regressor": {
            "type": "DummyRegressor",
            "strategy": dummy_regressor.strategy,
            "constant_prediction": float(dummy_regressor.constant_[0])
        },
        "classifier": {
            "type": "DummyClassifier", 
            "strategy": dummy_classifier.strategy,
            "classes": dummy_classifier.classes_.tolist(),
            "most_frequent_class": dummy_classifier.classes_[np.argmax(dummy_classifier.class_prior_)]
        }
    }

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
'''

# Save the FastAPI code
with open('main.py', 'w') as f:
    f.write(fastapi_code)

print("✓ FastAPI application code saved as 'main.py'")

# Step 3: Create requirements and run instructions
print("\n3. PROJECT SETUP FILES")
print("-" * 50)

requirements = '''fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
scikit-learn==1.3.2
joblib==1.3.2
numpy==1.25.2
pandas==2.1.4
'''

with open('requirements.txt', 'w') as f:
    f.write(requirements)

print("✓ Requirements file created")

# Create a simple test script
test_script = '''
# test_api.py - Simple test script for the FastAPI endpoints

import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
    # Test health check
    print("Testing health check...")
    response = requests.get(f"{base_url}/")
    print(f"Health check: {response.json()}")
    
    # Test regression prediction
    print("\\nTesting regression prediction...")
    reg_data = {
        "feature_1": 1.5,
        "feature_2": -0.8,
        "feature_3": 2.3,
        "feature_4": 0.1
    }
    
    response = requests.post(f"{base_url}/predict/regression", json=reg_data)
    print(f"Regression result: {response.json()}")
    
    # Test classification prediction
    print("\\nTesting classification prediction...")
    cls_data = {
        "feature_1": 0.5,
        "feature_2": -1.2,
        "feature_3": 1.8,
        "feature_4": -0.3
    }
    
    response = requests.post(f"{base_url}/predict/classification", json=cls_data)
    print(f"Classification result: {response.json()}")
    
    # Test model info
    print("\\nTesting model info...")
    response = requests.get(f"{base_url}/models/info")
    print(f"Model info: {response.json()}")

if __name__ == "__main__":
    test_api()
'''

with open('test_api.py', 'w') as f:
    f.write(test_script)

print("✓ Test script created as 'test_api.py'")

print("\n4. PROJECT STRUCTURE CREATED")
print("-" * 50)
print("📁 Project files:")
print("  ├── main.py                 # FastAPI application")
print("  ├── dummy_regressor.pkl     # Trained regression model") 
print("  ├── dummy_classifier.pkl    # Trained classification model")
print("  ├── requirements.txt        # Python dependencies")
print("  ├── test_api.py            # API testing script")
print("  └── README.md              # (to be created)")

print("\n5. HOW TO RUN THE APPLICATION")
print("-" * 50)
print("1. Install dependencies:")
print("   pip install -r requirements.txt")
print()
print("2. Start the FastAPI server:")
print("   uvicorn main:app --reload")
print("   # OR")
print("   python main.py")
print()
print("3. Test the API:")
print("   # In another terminal")
print("   python test_api.py")
print()
print("4. Access interactive docs:")
print("   http://localhost:8000/docs")

print("\n" + "=" * 70)
print("DAY 3 TASK SETUP COMPLETED! ✅")
print("=" * 70)