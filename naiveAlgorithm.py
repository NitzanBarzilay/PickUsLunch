import sys
from itertools import combinations_with_replacement, product
import pandas as pd
pd.options.mode.chained_assignment = None
from tqdm import tqdm
import time
import datetime
import gainFunction

NUMBER_OF_EATERS = 3

# number of the first restaurant to run the search on (for testing purposes)
NUMBER_OF_RESTAURANTS_TO_RUN = None  # None means all the restaurants


# ---------------------------------------- Algorithm wrapper functions  ------------------------------------------------
def NaiveAlgorithm(restaurants_list, menus_list, d1, d2, d3):
    """
    Naive algorithm wrapper function. returns the solution that the algorithm chose (restaurant and 3 meals) and it's runtime.
    :param rest_df: restaurant dataframe
    :param meals_df: meals dataframe
    :param diner1: list of 1st diner preferences
    :param diner2: list of 2nd diner preferences
    :param diner3: list of 3rd diner preferences
    :return: chosen restaurant dataframe (single row), 3 chosen meals dataframes (single row each), runtime (float).
    """
    chosen_restaurant = None
    chosen_restaurant_score = None
    chosen_restaurant_meals = None

    ret_restaurant = None
    ret_meal1 = None
    ret_meal2 = None
    ret_meal3 = None
    ret_runtime = None

    num_of_restaurants = len(restaurants_list["name"][:NUMBER_OF_RESTAURANTS_TO_RUN])
    idx = 0
    num_of_per = 0

    # save the time of the first iteration
    start_time = time.time()
    # for each restaurant, create the product of all the combinations of the dishes
    for restaurant_name in tqdm(restaurants_list["name"][:NUMBER_OF_RESTAURANTS_TO_RUN], position=0,
                                desc=f'Restaurants', ncols=80):

        # save restaurant row from df
        restaurant_row = restaurants_list.loc[restaurants_list["name"] == restaurant_name]
        # extract the meals from the restaurant's row in the menus list
        meals_df = menus_list.loc[menus_list["rest_name"] == restaurant_name]
        # create a new column in the meals df with the and operation of vegetarian and GF
        meals_df["veg_and_gf"] = meals_df["vegetarian"] & meals_df["GF"]
        meals_df['veg_and_gf'] = meals_df['vegetarian'].values[:] & meals_df['GF'].values[:]
        # save only the meals in a list
        meals = meals_df["name"].tolist()
        # create product of meals of the specific restaurant
        meals_product = list(product(meals, repeat=NUMBER_OF_EATERS))
        num_of_per += meals_product.__len__()


        # calculate the score of each meal combination
        for permutation in tqdm(meals_product, position=1, desc=f'Restaurant - {idx}/{num_of_restaurants}', leave=False,
                                ncols=80):
            # params = LossFunc.user_inputs_to_loss_function_inputs(d1, d2, d3, restaurant_name, permutation[0], permutation[1], permutation[2])
            meal1 = meals_df.loc[(meals_df["name"] == permutation[0]) & (meals_df["rest_name"] == restaurant_name)]
            meal2 = meals_df.loc[(meals_df["name"] == permutation[1]) & (meals_df["rest_name"] == restaurant_name)]
            meal3 = meals_df.loc[(meals_df["name"] == permutation[2]) & (meals_df["rest_name"] == restaurant_name)]
            params = gainFunction.user_inputs_to_gain_function_inputs(d1, d2, d3, restaurant_row, meal1, meal2, meal3)

            # calculate the score of the permutation
            score = gainFunction.gain(*params)

            # if the score is better than the current best score, save the permutation and the score
            if chosen_restaurant_score is None or chosen_restaurant_score < score:

                chosen_restaurant_score = score
                ret_restaurant = restaurant_row
                ret_meal1 = meal1
                ret_meal2 = meal2
                ret_meal3 = meal3


        idx += 1

    # calculate the time of the last iteration
    time_elapsed = time.time() - start_time

    return ret_restaurant, ret_meal1, ret_meal2, ret_meal3, ret_runtime
