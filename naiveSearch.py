from itertools import combinations_with_replacement, product
import pandas as pd
from tqdm import tqdm
import time

CSV_RESTAURANTS     = "csv_wolt_restaurants_19-8-22.csv"
CSV_MENUS           = "csv_wolt_menus_20-8-22.csv"
NUMBER_OF_EATERS    = 3

def naive_search(csv_restaurants = CSV_RESTAURANTS, csv_menus = CSV_MENUS, number_of_eaters = NUMBER_OF_EATERS):
    # open the restaurants list csv in pandas df
    restaurants_list = pd.read_csv(CSV_RESTAURANTS, encoding="utf-8")
    menus_list = pd.read_csv(CSV_MENUS, encoding="utf-8")

    chosen_restaurant       = None
    chosen_restaurant_score = None
    chosen_restaurant_meals = None
    num_of_restaurants      = len(restaurants_list)
    idx                     = 0
    num_of_per              = 0

    # save the time of the first iteration
    start_time = time.time()
    #for each restaurant, create the product of all the combinations of the dishes
    for restaurant_name in tqdm(restaurants_list["name"], position=0, desc=f'Restaurants', ncols=80):
        # extract the meals from the restaurant's row in the menus list
        meals_df = menus_list.loc[menus_list["rest_name"] == restaurant_name]
        # save only the meals in a list
        meals = meals_df["name"].tolist()
        # create product of meals of the specific restaurant
        meals_product = list(product(meals, repeat=NUMBER_OF_EATERS))
        num_of_per += meals_product.__len__()

        # calculate the score of each meal combination
        for permutation in tqdm(meals_product, position=1, desc=f'Restaurant - {idx}/{num_of_restaurants}', leave=False, ncols=80):

            # calculate the score of the permutation
            score = 0 # TODO: update score function!
            # if the score is better than the current best score, save the permutation and the score
            if chosen_restaurant_score is None or chosen_restaurant_score < score:
                chosen_restaurant = restaurant_name
                chosen_restaurant_score = score
                chosen_restaurant_meals = permutation
            for i in range(10000):
                pass
        idx += 1

    # calculate the time of the last iteration
    time_elapsed = time.time() - start_time
    print(f'\nTime elapsed: {time_elapsed}')
    print(f'Number of permutations - {num_of_per}')
    print(f'Chosen restaurant - {chosen_restaurant}')
    print(f'Chosen restaurant score - {chosen_restaurant_score}')
    print(f'Chosen restaurant meals - {chosen_restaurant_meals}')

naive_search()