def ecosystem_resilience(recovery_rate, disturbance_frequency):

    """

    Calculate the resilience of an ecosystem.



    Args:

        recovery_rate (float): The recovery rate of the ecosystem.

        disturbance_frequency (float): The disturbance frequency in the ecosystem.



    Returns:

        float: The calculated resilience of the ecosystem.

    """

    if recovery_rate < 0 or disturbance_frequency < 0:

        raise ValueError("Recovery rate and disturbance frequency must be positive values.")

    

    resilience = recovery_rate / (recovery_rate + disturbance_frequency)

    

    return resilience
