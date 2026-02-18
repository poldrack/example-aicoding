def present_value(future_value, discount_rate, years):

    """

    Calculate the present value of a future cash flow.

    

    Parameters:

    future_value (float): The future cash amount

    discount_rate (float): The discount rate as a decimal (e.g., 0.05 for 5%)

    years (float): The number of years until the cash flow is received



    Returns:

    float: The present value of the future cash flow

    """

    return future_value / ((1 + discount_rate) ** years)



# Example usage

future_value = 10000  # A future cash flow of 10,000

discount_rate = 0.05  # A discount rate of 5%

years = 10  # The cash flow will be received in 10 years



present_value_result = present_value(future_value, discount_rate, years)

print(f"The present value of the future cash flow is: {present_value_result:.2f}")
