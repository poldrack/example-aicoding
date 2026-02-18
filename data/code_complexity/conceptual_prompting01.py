def angular_momentum(moment_of_inertia: float, angular_velocity: float) -> float:

    """

    Calculate the angular momentum of an object in rotational motion.



    :param moment_of_inertia: The moment of inertia of the object (kg*m^2).

    :param angular_velocity: The angular velocity of the object (rad/s).

    :return: The angular momentum of the object (kg*m^2/s).

    """

    return moment_of_inertia * angular_velocity





def main():

    # Test data

    moment_of_inertia = 2.5  # kg*m^2

    angular_velocity = 7.8  # rad/s



    # Calculate angular momentum

    result = angular_momentum(moment_of_inertia, angular_velocity)

    print(f"Angular momentum: {result:.2f} kg*m^2/s")





if __name__ == "__main__":

    main()
