import numpy as np



def solow_growth_model(S, delta, n, f, K0, L0, T):

    """

    Solow Growth Model function.



    Parameters:

    - S (float): savings rate

    - delta (float): depreciation rate

    - n (float): population growth rate

    - f (function): production function (takes K and L as input, returns Y)

    - K0 (float): initial capital stock

    - L0 (float): initial labor force

    - T (int): time periods



    Returns:

    - K_path (list): path of capital over time

    - L_path (list): path of labor over time

    """



    # Initialize paths

    K_path = [K0]

    L_path = [L0]



    # Iterate for T periods

    for t in range(1, T + 1):

        Yt = f(K_path[-1], L_path[-1])

        It = S * Yt

        Kt_next = (1 - delta) * K_path[-1] + It

        Lt_next = L_path[-1] * (1 + n)



        K_path.append(Kt_next)

        L_path.append(Lt_next)



    return K_path, L_path



# Cobb-Douglas production function

def cobb_douglas(K, L, A=1, alpha=0.3, beta=0.7):

    return A * (K ** alpha) * (L ** beta)



# Example usage

S = 0.25

delta = 0.05

n = 0.01

K0 = 1000

L0 = 1000

T = 10



K_path, L_path = solow_growth_model(S, delta, n, lambda K, L: cobb_douglas(K, L), K0, L0, T)

print("Capital path:", K_path)

print("Labor path:", L_path)
