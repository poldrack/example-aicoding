import numpy as np



def create_resource_utilization_curve(num_resources, min_value=0, max_value=1):

    """

    Generate a random resource utilization curve.

    

    Params:

    -------

    num_resources: int

        Number of resources to be allocated

    min_value: float, optional (default: 0)

        Minimum value for resource allocation

    max_value: float, optional (default: 1)

        Maximum value for resource allocation

        

    Returns:

    --------

    np.array

        Array with resource allocations normalized to sum to 1

    """

    resources = np.random.uniform(min_value, max_value, num_resources)

    resources /= np.sum(resources)

    return resources



def calculate_ecological_niche_overlap(species1_resources, species2_resources):

    """

    Calculate the ecological niche overlap between two species.

    

    Params:

    -------

    species1_resources: np.array

        Array containing the resource utilization curve for species 1

    species2_resources: np.array

        Array containing the resource utilization curve for species 2

        

    Returns:

    --------

    float

        Ecological niche overlap value (ranges from 0 to 1)

    """

    if len(species1_resources) != len(species2_resources):

        raise ValueError("Resource utilization curves should have the same length")

    return np.sum(np.minimum(species1_resources, species2_resources))



# Example usage

species1_resources = create_resource_utilization_curve(10)

species2_resources = create_resource_utilization_curve(10)

overlap = calculate_ecological_niche_overlap(species1_resources, species2_resources)

print("Ecological niche overlap:", overlap)
