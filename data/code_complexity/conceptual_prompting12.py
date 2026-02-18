import math



def gravitational_force(m1, m2, r):

    """

    Calculate the gravitational force between two objects.

    

    Parameters:

    m1 (float): Mass of first object in kilograms

    m2 (float): Mass of second object in kilograms

    r (float): Distance between the two objects in meters



    Returns:

    float: Gravitational force between the two objects in Newtons

    """

    G = 6.67430e-11  # Universal gravitational constant in m^3 kg^-1 s^-2

    force = G * m1 * m2 / (r ** 2)

    return force



if __name__ == "__main__":

    mass_earth = 5.972e24  # Mass of Earth in kg

    mass_moon = 7.342e22  # Mass of Moon in kg

    distance_earth_moon = 384400000  # Distance between Earth and Moon in m



    force = gravitational_force(mass_earth, mass_moon, distance_earth_moon)

    print(f"Gravitational force between Earth and Moon: {force} N")
