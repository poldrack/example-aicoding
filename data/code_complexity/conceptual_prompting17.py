import numpy as np

import random



# Define species' temperature and precipitation tolerances

species_tolerances = {

    'species_1': {'temp': (5, 30), 'precip': (800, 1200)},

    'species_2': {'temp': (10, 25), 'precip': (500, 1000)},

    'species_3': {'temp': (0, 15), 'precip': (600, 1400)}

}



# Projected climate scenarios

projected_climate_data = [

    {'year': 2021, 'temp': 12, 'precip': 850},

    {'year': 2022, 'temp': 13, 'precip': 870},

    {'year': 2023, 'temp': 14, 'precip': 880},

    {'year': 2024, 'temp': 15, 'precip': 930}

]





def generate_species_distribution(species_name, climate_data):

    tolerance = species_tolerances[species_name]



    distribution_matrix = np.zeros((len(climate_data), 2))

    

    for i, year_data in enumerate(climate_data):

        suitable_temp = tolerance['temp'][0] <= year_data['temp'] <= tolerance['temp'][1]

        suitable_precip = tolerance['precip'][0] <= year_data['precip'] <= tolerance['precip'][1]

        suitable = suitable_temp and suitable_precip



        if suitable:

            distribution_matrix[i] = [year_data['year'], 1]

        else:

            distribution_matrix[i] = [year_data['year'], 0]



    return distribution_matrix





def print_species_distribution_matrix(species_list, reference_data):

    for species in species_list:

        print(f"Climate suitability for {species}:")

        matrix = generate_species_distribution(species, reference_data)

        print(matrix)

        print("\n")





print_species_distribution_matrix(species_tolerances.keys(), projected_climate_data)
