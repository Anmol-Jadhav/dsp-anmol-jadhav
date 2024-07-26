import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from typing import Dict, Any


def build_model(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Build and train the linear regression model.

    Parameters:
    data (pd.DataFrame): The input data containing features and target variable.

    Returns:
    Dict[str, Any]: A dictionary containing the root mean squared log error of the model.
    """
    X = data.drop(columns=["SalePrice"])  # Features
    y = data["SalePrice"]  # Target variable

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Continuous Features
    continuous_features = ["LotArea", "YearBuilt"]  
    # Categorical Features
    categorical_features = ["Neighborhood", "BldgType"]
    
    # Create and fit transformers
    scaler = StandardScaler()
    onehot = OneHotEncoder(handle_unknown='ignore')
    
    X_train_continuous_scaled = scaler.fit_transform(X_train[continuous_features])
    X_train_categorical_encoded = onehot.fit_transform(X_train[categorical_features])
    X_train_processed = np.concatenate([X_train_continuous_scaled, X_train_categorical_encoded.toarray()], axis=1)

    # Train the model
    model = LinearRegression()
    model.fit(X_train_processed, y_train)

    # Save transformers
    joblib.dump(scaler, 'C:/Users/ASUS/dsp-anmol-jadhav/models/scaler.joblib')
    joblib.dump(onehot, 'C:/Users/ASUS/dsp-anmol-jadhav/models/Encoder.joblib')
    joblib.dump(model, 'C:/Users/ASUS/dsp-anmol-jadhav/models/model.joblib')

    # Evaluate the model
    y_pred = model.predict(X_train_processed)
    rmsle = compute_rmsle(y_train, y_pred)
    
    return {'rmse': rmsle}