def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using the trained model.

    Parameters:
    input_data (pd.DataFrame): The input data containing features for prediction.

    Returns:
    np.ndarray: The predicted values.
    """
    # Load transformers
    scaler = joblib.load('C:/Users/ASUS/dsp-anmol-jadhav/models/scaler.joblib')
    onehot = joblib.load('C:/Users/ASUS/dsp-anmol-jadhav/models/encoder.joblib')
    
    # Transform input data
    continuous_features = ["LotArea", "YearBuilt"]
    categorical_features = ["Neighborhood", "BldgType"]
    X_test_continuous_scaled = scaler.transform(input_data[continuous_features])
    X_test_categorical_encoded = onehot.transform(input_data[categorical_features])
    X_test_processed = np.concatenate([X_test_continuous_scaled, X_test_categorical_encoded.toarray()], axis=1)
    
    # Load trained model
    loaded_model = joblib.load('C:/Users/ASUS/dsp-anmol-jadhav/models/model.joblib')

    # Make predictions
    predicted_prices = loaded_model.predict(X_test_processed)
    return predicted_prices
