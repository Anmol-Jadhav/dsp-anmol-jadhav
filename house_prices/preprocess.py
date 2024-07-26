def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    """
    Compute the Root Mean Squared Logarithmic Error (RMSLE).

    Parameters:
    y_test (np.ndarray): The actual values.
    y_pred (np.ndarray): The predicted values.
    precision (int): The precision for rounding the RMSLE.

    Returns:
    float: The computed RMSLE value.
    """
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


# Example usage:
if __name__ == "__main__":
    data = pd.read_csv('D:/SEM2/DSP/train.csv')
    model_performance = build_model(data)
    print("Model performance:", model_performance)
R