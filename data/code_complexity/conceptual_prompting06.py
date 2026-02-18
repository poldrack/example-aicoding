import math



def escape_velocity(mass, radius, G=6.67430e-11):

    """

    Calculate the escape velocity from a celestial body, given its mass and radius.



    Args:

    mass (float): Mass of the celestial body in kg.

    radius (float): Radius of the celestial body in meters.



    Returns:

    float: Escape velocity in m/s.

    """

    return math.sqrt(2 * G * mass / radius)



if __name__ == "__main__":

    mass_earth = 5.972e24  # Earth mass in kg

    radius_earth = 6.371e6  # Earth radius in meters

    print(f"Escape velocity of Earth: {escape_velocity(mass_earth, radius_earth):.2f} m/s")
