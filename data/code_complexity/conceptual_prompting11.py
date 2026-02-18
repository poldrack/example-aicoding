import numpy as np



def calculate_rmse(predicted_values, actual_values):

    """

    Calculate the Root Mean Square Error (RMSE) between the predicted and actual values.



    Args:

    predicted_values (list,1d-array): List or 1D array containing the predicted values.

    actual_values (list, 1d-array): List or 1D array containing the actual values.



    Returns:

    float: The root mean square error (RMSE) value.

    """

    if len(predicted_values) != len(actual_values):

        raise ValueError("Both predicted and actual values must have same length")



    squared_errors = [(pred - actual)**2 for pred, actual in zip(predicted_values, actual_values)]

    mean_squared_error = np.mean(squared_errors)

    rmse = np.sqrt(mean_squared_error)

    return rmse



# Example usage:

predicted = np.array([3.1, 5.0, 7.5, 9.7, 10.5])

actual = np.array([3, 5, 8, 10, 11])

rmse_value = calculate_rmse(predicted, actual)

print("RMSE:", rmse_value)
