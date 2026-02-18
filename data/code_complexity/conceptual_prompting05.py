def real_gdp_growth_rate(nominal_gdp_growth_rate, inflation_rate):

    """

    Calculate the real GDP growth rate.

    

    :param nominal_gdp_growth_rate: Nominal GDP growth rate as a percentage.

    :param inflation_rate: Inflation rate as a percentage.

    :return: Real GDP growth rate as a percentage.

    """

    real_growth_rate = ((1 + nominal_gdp_growth_rate / 100) / (1 + inflation_rate / 100)) - 1

    return real_growth_rate * 100



# Example values

nominal_gdp_growth_rate = 5  # 5% nominal GDP growth rate

inflation_rate = 2  # 2% inflation rate



# Calculate the real GDP growth rate

real_growth_rate = real_gdp_growth_rate(nominal_gdp_growth_rate, inflation_rate)



# Output the result

print(f"The real GDP growth rate is {real_growth_rate:.2f}%")
