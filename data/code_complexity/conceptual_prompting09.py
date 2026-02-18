import math



def calculate_velocity(radius, time_period):

    """

    Calculate the velocity of an object in circular motion.



    Args:

    radius (float): The radius of the circular path (in meters)

    time_period (float): The time period of one complete revolution (in seconds)



    Returns:

    float: The velocity of the object (in meters per second)

    """

    circumference = 2 * math.pi * radius

    velocity = circumference / time_period

    return velocity



# Example usage:

radius = 5.0  # meters

time_period = 10.0  # seconds

velocity = calculate_velocity(radius, time_period)

print(f"The velocity of the object is {velocity} m/s")
