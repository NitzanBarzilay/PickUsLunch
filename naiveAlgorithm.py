import sys
from itertools import combinations_with_replacement, product
import pandas as pd
import vlc
from tqdm import tqdm
import time
import datetime
import gainFunction
import main


CSV_RESTAURANTS     = "data/restaurantsData.csv"
CSV_MENUS           = "data/mealsData.csv"
NUMBER_OF_EATERS    = 3

# number of the first restaurant to run the search on (for testing purposes)
NUMBER_OF_RESTAURANTS_TO_RUN = None # None means all the restaurants

def naive_search( d1, d2, d3, constrain_file_idx, csv_restaurants = CSV_RESTAURANTS, csv_menus = CSV_MENUS, number_of_eaters = NUMBER_OF_EATERS):
    # open the restaurants list csv in pandas df
    restaurants_list = pd.read_csv(CSV_RESTAURANTS, encoding="utf-8")
    menus_list = pd.read_csv(CSV_MENUS, encoding="utf-8")

    chosen_restaurant       = None
    chosen_restaurant_score = None
    chosen_restaurant_meals = None
    num_of_restaurants      = len(restaurants_list["name"][:NUMBER_OF_RESTAURANTS_TO_RUN])
    idx                     = 0
    num_of_per              = 0

    # save the time of the first iteration
    start_time = time.time()
    #for each restaurant, create the product of all the combinations of the dishes
    for restaurant_name in tqdm(restaurants_list["name"][:NUMBER_OF_RESTAURANTS_TO_RUN], position=0, desc=f'Restaurants', ncols=80):

        # save restaurant row from df
        restaurant_row = restaurants_list.loc[restaurants_list["name"] == restaurant_name]
        # extract the meals from the restaurant's row in the menus list
        meals_df = menus_list.loc[menus_list["rest_name"] == restaurant_name]
        # save only the meals in a list
        meals = meals_df["name"].tolist()
        # create product of meals of the specific restaurant
        meals_product = list(product(meals, repeat=NUMBER_OF_EATERS))
        num_of_per += meals_product.__len__()

        # calculate the score of each meal combination
        for permutation in tqdm(meals_product, position=1, desc=f'Restaurant - {idx}/{num_of_restaurants}', leave=False, ncols=80):
            # params = LossFunc.user_inputs_to_loss_function_inputs(d1, d2, d3, restaurant_name, permutation[0], permutation[1], permutation[2])
            meal1 = meals_df.loc[(meals_df["name"] == permutation[0]) & (meals_df["rest_name"] == restaurant_name)]
            meal2 = meals_df.loc[(meals_df["name"] == permutation[1]) & (meals_df["rest_name"] == restaurant_name)]
            meal3 = meals_df.loc[(meals_df["name"] == permutation[2]) & (meals_df["rest_name"] == restaurant_name)]
            params = LossFunc.user_inputs_to_gain_function_inputs(d1, d2, d3, restaurant_row, meal1, meal2, meal3)

            # calculate the score of the permutation
            score = LossFunc.gain(*params)
            if score > 0:
                with open(f"results/naive_search_results_input_constraints_{constrain_file_idx}.csv", "a", encoding="utf-8") as f:
                    f.write(f"{restaurant_name},{permutation[0]},{permutation[1]},{permutation[2]},{score}\n")
            # if the score is better than the current best score, save the permutation and the score
            if chosen_restaurant_score is None or chosen_restaurant_score < score:
                # open a csv file and save the permutation and the score
                chosen_restaurant = restaurant_name
                chosen_restaurant_score = score
                chosen_restaurant_meals = permutation
                # play successes music :)
                if score is not None and score > 0:
                    p = vlc.MediaPlayer("fun/AH.mp3")
                    p.play()
        idx += 1

    # calculate the time of the last iteration
    time_elapsed = time.time() - start_time
    print(f"\nConstraints file: {constrain_file_idx}")
    print(f'Time elapsed: {datetime.timedelta(seconds=time_elapsed)}')
    print(f'Number of permutations - {num_of_per}')
    print(f'Chosen restaurant - {chosen_restaurant}')
    print(f'Chosen restaurant score - {chosen_restaurant_score}')
    print(f'Chosen restaurant meals - {chosen_restaurant_meals}')



# example of usage
if __name__ == "__main__":
    constraints = ["example_preferences/input_constraints_1.txt",
                    "example_preferences/input_constraints_2.txt",
                    "example_preferences/input_constraints_3.txt",
                    "example_preferences/input_constraints_4.txt",
                    "example_preferences/input_constraints_5.txt"]
    for i, constraint in enumerate(constraints):
        d1, d2, d3 = main.get_diners_constraints(constraint)
        naive_search(d1, d2, d3, i+1)
        p = vlc.MediaPlayer("fun/who_is_next.mp3")
        p.play()
    p = vlc.MediaPlayer("fun/MW.mp3")
    p.play()