def interest_rate_parity(spot_exchange_rate, forward_exchange_rate, interest_rate_1, interest_rate_2):

    """

    Calculate the interest rate parity based on spot and forward exchange rates and interest rates of two countries.



    :param spot_exchange_rate: The spot exchange rate between the two countries

    :type spot_exchange_rate: float

    :param forward_exchange_rate: The forward exchange rate between the two countries

    :type forward_exchange_rate: float

    :param interest_rate_1: Interest rate of country 1

    :type interest_rate_1: float

    :param interest_rate_2: Interest rate of country 2

    :type interest_rate_2: float

    :return: A boolean, True if interest rate parity holds, False otherwise

    :rtype: bool

    """

    irp_equation = (1 + interest_rate_1) / (1 + interest_rate_2)

    exchange_rate_ratio = forward_exchange_rate / spot_exchange_rate



    return round(irp_equation, 6) == round(exchange_rate_ratio, 6)





# Example values

spot_exchange_rate = 1.20

forward_exchange_rate = 1.22

interest_rate_1 = 0.05

interest_rate_2 = 0.03



# Check if interest rate parity holds

result = interest_rate_parity(spot_exchange_rate, forward_exchange_rate, interest_rate_1, interest_rate_2)

print(result)
