import numpy as np

import scipy.stats as stats



def create_contingency_table(a, b):

    """

    Create a contingency table given two categorical variables.



    :param a: A list of categories for the first variable

    :param b: A list of categories for the second variable

    :return: A numpy array representing the contingency table

    """

    a_categories = list(set(a))

    b_categories = list(set(b))

    table = np.zeros((len(a_categories), len(b_categories)))



    for i, a_categ in enumerate(a_categories):

        for j, b_categ in enumerate(b_categories):

            table[i][j] = sum((a == a_categ) & (b == b_categ))



    return table





def chi_squared_test(table):

    """

    Perform the chi-squared test on a contingency table.



    :param table: A numpy array representing the contingency table

    :return: The test statistic, p-value, degrees of freedom, and expected frequencies

    """

    return stats.chi2_contingency(table)





def main():

    # Sample data

    a_categories = ["A", "A", "B", "B", "A", "B", "A", "B", "A", "B"]

    b_categories = ["X", "Y", "X", "Y", "X", "X", "Y", "Y", "X", "Y"]



    table = create_contingency_table(a_categories, b_categories)



    chi2, p, df, expected = chi_squared_test(table)



    print("Chi-Square Statistic:", chi2)

    print("P-Value:", p)

    print("Degrees of Freedom:", df)

    print("Expected Frequencies:", expected)



if __name__ == "__main__":

    main()
