import math



def projectile_motion(initial_velocity, launch_angle, gravity=-9.81):

    """

    Calculate the motion of a projectile given the initial velocity,

    launch angle, and acceleration due to gravity.



    Args:

    initial_velocity (float): Initial velocity of the projectile (m/s).

    launch_angle (float): Launch angle of the projectile in degrees.

    gravity (float, optional): Acceleration due to gravity (m/s^2). Default value is -9.81.



    Returns:

    tuple: Contains maximum height (m), time of flight (s), and horizontal range (m).

    """



    # Convert launch angle to radians

    angle_radians = math.radians(launch_angle)



    # Calculate time of flight

    time_of_flight = (2 * initial_velocity * math.sin(angle_radians)) / gravity



    # Calculate maximum height

    max_height = (initial_velocity**2 * math.sin(angle_radians)**2) / (2 * -gravity)



    # Calculate horizontal range

    horizontal_range = initial_velocity * time_of_flight * math.cos(angle_radians)



    return max_height, time_of_flight, horizontal_range



# Test example

initial_velocity = 20  # m/s

launch_angle = 45  # degrees



max_height, time_of_flight, horizontal_range = projectile_motion(initial_velocity, launch_angle)

print("Max Height:", max_height)

print("Time of Flight:", time_of_flight)

print("Horizontal Range:", horizontal_range)
