import numpy as np
import pandas as pd

HUNGRY_MINUTES = 45
MUST = 1
MUST_NOT = 2


def user_inputs_to_loss_function_inputs(diner1_inputs, diner2_inputs, diner3_inputs, rest_name, meal1, meal2, meal3):
    """
    Converts the user inputs to the loss function inputs.
    :params diner1_inputs, diner2_inputs, diner3_inputs: user input in format of: [0 - kosher, 1 - vegetarian, 2 - gluten free, 3 - alcohol free, 4 - prefer spicy, 5 - max price, 6 - min rating, 7 - hunger level, 8 - desired cuisines, 9 - day]
    :param rest_name: name of the restaurant
    :param meal1: name of 1st meal
    :param meal2: name of 2nd meal
    :param meal3: name of 3rd meal
    :return: a list of inputs for the loss function - [O, M, K, D, R, C, V1, V2, V3, G1, G2, G3, A1, A2, A3, S1, S2, S3, PH1, PH2, PH3, PS1, PS2, PS3]
    """

    rest_df = pd.read_csv("data/csv_wolt_restaurants_19-8-22.csv")
    meals_df = pd.read_csv("data/csv_wolt_menus_20-8-22.csv")
    rest = rest_df[rest_df["name"] == rest_name].reset_index(drop=True)
    meal1_df = meals_df[(meals_df['rest_name'] == rest_name) & (meals_df["name"] == meal1)].reset_index(drop=True)
    meal2_df = meals_df[(meals_df['rest_name'] == rest_name) & (meals_df["name"] == meal2)].reset_index(drop=True)
    meal3_df = meals_df[(meals_df['rest_name'] == rest_name) & (meals_df["name"] == meal3)].reset_index(drop=True)
    kosher1, vegetarian1, gluten_free1, alcohol_free1, spicy1, max_price1, rating1, hungry1, cuisines1, weekday = diner1_inputs
    kosher2, vegetarian2, gluten_free2, alcohol_free2, spicy2, max_price2, rating2, hungry2, cuisines2, weekday = diner2_inputs
    kosher3, vegetarian3, gluten_free3, alcohol_free3, spicy3, max_price3, rating3, hungry3, cuisines3, weekday = diner3_inputs

    # Group constraints:
    diners_kosher = False if (kosher1 == 0 and kosher2 == 0 and kosher3 == 0) else True
    diners_avg_rating = np.mean((rating1, rating2, rating3))
    hungry_diners = True if np.sum((hungry1, hungry2, hungry3)) >= 2 else False
    rest_cuisines = rest.at[0, 'food categories'].split('---')
    diner1_cui = 1 if len([meal for meal in cuisines1 if meal in rest_cuisines]) > 0 else 0
    diner2_cui = 1 if len([meal for meal in cuisines2 if meal in rest_cuisines]) > 0 else 0
    diner3_cui = 1 if len([meal for meal in cuisines3 if meal in rest_cuisines]) > 0 else 0

    O = 1 if weekday in rest.at[0, 'opening days'] else 0
    M = 1 if ((meal1_df.at[0, 'price'] + meal2_df.at[0, 'price'] + meal2_df.at[
        0, 'price']) >= 100) else 0  # TODO replace with order min
    K = 0 if diners_kosher and not rest['kosher'] else 1
    DT = rest.at[0, 'delivery estimation [minutes]'] + rest.at[0, 'prep estimation [minutes]']
    D = 0 if (hungry_diners and rest.at[0, 'delivery estimation [minutes]'] + rest.at[
        0, 'prep estimation [minutes]'] >= HUNGRY_MINUTES) else 1
    RD = rest.at[0, 'rating'] - diners_avg_rating
    R = 1 if diners_avg_rating <= rest.at[0, 'rating'] else 0
    C = diner1_cui + diner2_cui + diner3_cui

    # individual constraints:
    diner_delivery_cost = + rest.at[0, 'delivery price'] / 3
    price1, price2, price3 = meal1_df.at[0, 'price'], meal2_df.at[0, 'price'], meal3_df.at[0, 'price']

    V1 = 0 if vegetarian1 == MUST and not meal1_df.at[0, 'vegetarian'] else 1
    V2 = 0 if vegetarian2 == MUST and not meal2_df.at[0, 'vegetarian'] else 1
    V3 = 0 if vegetarian3 == MUST and not meal3_df.at[0, 'vegetarian'] else 1
    G1 = 0 if gluten_free1 == MUST and not meal1_df.at[0, 'GF'] else 1
    G2 = 0 if gluten_free2 == MUST and not meal2_df.at[0, 'GF'] else 1
    G3 = 0 if gluten_free3 == MUST and not meal3_df.at[0, 'GF'] else 1
    A1 = 0 if alcohol_free1 == MUST and meal1_df.at[0, 'alcohol_percentage'] > 0 else 1
    A2 = 0 if alcohol_free2 == MUST and meal2_df.at[0, 'alcohol_percentage'] > 0 else 1
    A3 = 0 if alcohol_free3 == MUST and meal3_df.at[0, 'alcohol_percentage'] > 0 else 1
    spicy_meal1, spicy_meal2, spicy_meal3 = meal1_df.at[0, 'spicy'], meal2_df.at[0, 'spicy'], meal3_df.at[0, 'spicy']
    S1 = 0 if (spicy1 == MUST and not spicy_meal1) or (spicy1 == MUST_NOT and spicy_meal1) else 1
    S2 = 0 if (spicy2 == MUST and not spicy_meal2) or (spicy2 == MUST_NOT and spicy_meal2) else 1
    S3 = 0 if (spicy3 == MUST and not spicy_meal3) or (spicy3 == MUST_NOT and spicy_meal3) else 1
    PH1 = 1 if price1 + diner_delivery_cost <= max_price1 else 0
    PH2 = 1 if price2 + diner_delivery_cost <= max_price2 else 0
    PH3 = 1 if price3 + diner_delivery_cost <= max_price3 else 0
    PS1 = max_price1 - price1 if PH1 == 1 else 0
    PS2 = max_price2 - price2 if PH2 == 1 else 0
    PS3 = max_price3 - price3 if PH3 == 1 else 0

    return [O, M, K, DT, D, RD, R, C, V1, V2, V3, G1, G2, G3, A1, A2, A3, S1, S2, S3, PH1, PH2, PH3, PS1, PS2, PS3]


def loss(O, M, K, DT, D, RD, R, C, V1, V2, V3, G1, G2, G3, A1, A2, A3, S1, S2, S3, PH1, PH2, PH3, PS1, PS2, PS3) -> float:
    """
    Loss function for the optimization problem.
    based on variables per restaurant (open, minimal order price, kosher, delivery time, rating, cuisines) and per diner (vegetarian, gluten free, alcohol_free, spicy, price).
    :param O: (open) - 1 if the restaurant open 0 otherwise
    :param M: (minimal order price) - 1 if the meal's combination surpasses the restaurant's minimal order price, 0 otherwise
    :param K: (kosher) - 1 if at least one diner eats kosher and the restaurant is kosher or none of the diners eat kosher, 0 otherwise
    :param DT: (delivery time) - delivery time + preparation time in minutes
    :param D: (delivery) - based on avg hunger level among the group. if hunger level is high - 1 if the meal is ready in less than 30 minutes, 0 otherwise. if hunger level is low - 1 if the meal is ready in less than 60 minutes, 0 otherwise.
    :param RD: (rating difference) - float on a scale of -9 to 9 - the difference between the restaurant's rating and the average rating of the diners
    :param R: (rating) - 1 if the restaurant is above avg desired minimal rating among the group or does not have a rating, 0 otherwise
    :param C: (cuisines) - 0-3 according to the amount of diners who prefer a cuisine that the restaurant offers.
    :param V1, V2, V3: (vegetarian) 1 if the meal matches the vegetarian desires of the diner, 0 otherwise
    :param G1, G2, G3: (gluten free) - 1 if the meal matches the gluten desires of the diner, 0 otherwise
    :param A1, A2, A3: (alcohol free) - 1 if the meal matches the alcohol desires of the diner, 0 otherwise
    :param S1, S2, S3: (spicy) - 1 if the meal matches the spiciness desires of the diner, 0 otherwise
    :param PH1, PH2, PH3: (price hard) - 1 if the meal is lower than the diner's desired maximal meal price, 0 otherwise
    :param PS1, PS2, PS3: (price soft) - difference between diner's maximal price and meals price, 0 if the meal's price is higher than the diner's desired maximal meal price
    :return: The loss value of the given inputs, according to the desired hard and soft constraints.
    """


    """
    hard constraints: 
    in order to be valid, a combination ob a restaurant and 3 meals must fulfill the following constraints:
    - the restaurant must be open (O)
    - the sum of prices of all 3 meals must surpass the restaurant's minimal order price (M)
    - the meal must match the vegetarian preferences of all 3 diners (V1, V2, V3)
    - the meal must match the gluten preferences of all 3 diners (G1, G2, G3)
    - the meal must match the alcohol preferences of all 3 diners (A1, A2, A3)    
    - the meal must be affordable (PH1, PH2, PH3)
    """

    hard_constraints = O * M * K * V1 * V2 * V3 * G1 * G2 * G3 * A1 * A2 * A3 * PH1 * PH2 * PH3
    if hard_constraints == 0: # if at least 1 hard constraint is not met, return a loss value of 0
        return 0

    """
    soft constraints:
    - delivery time in minuts (DT)
    - delivery time matches diners' hunger level (D)
    - rating difference (RD)
    - rating (R)
    - cuisines preferences (C)
    - spiciness preferences (S1, S2, S3)
    - price differences (PS1, PS2, PS3)
    """
    DELIVERY_W = -1
    RATING_W = 1
    CUISINE_W = 1
    SPICY_W = 1
    PRICE_W = 2

    loss = (D * DT) / 60 * DELIVERY_W
    loss += RD * RATING_W
    loss +=  ((PS1 + PS2 + PS3) / 3) / 10 * PRICE_W
    loss += C * CUISINE_W
    loss += (S1 + S2 + S3) * SPICY_W

    return loss

