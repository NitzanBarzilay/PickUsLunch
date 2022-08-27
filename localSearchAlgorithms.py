import random
import sys
from random import shuffle
from timeit import default_timer as timer
from typing import List, Tuple

import numpy as np
from simpleai.search import (SearchProblem, breadth_first, depth_first,
                             uniform_cost, astar, greedy, hill_climbing_stochastic, hill_climbing,
                             hill_climbing_random_restarts, simulated_annealing)

import dataFrameParser
import gainFunction

# ---------------------------------------------- Constants -----------------------------------------------------------
MAX_COST_REST = 1000
MAX_COST_MEAL = 100

# ---------------------------------------------- Gain for climbing ---------------------------------------------------
HUNGRY_MINUTES = 45
MUST = 1
MUST_NOT = 2

count = 0


def user_inputs_to_gain_function_inputs(diner1_inputs, diner2_inputs, diner3_inputs, rest, meal1_df=None,
                                        meal2_df=None,
                                        meal3_df=None):
    """
    Converts the user inputs to the loss function inputs.
    :params diner1_inputs, diner2_inputs, diner3_inputs: user input in format of: [0 - kosher, 1 - vegetarian,
    2 - gluten free, 3 - alcohol free, 4 - prefer spicy, 5 - max price, 6 - min rating, 7 - hunger level, 8 - desired cuisines, 9 - day]
    :param rest: df with 1 row containing the restaurant
    :param meal1_df: df with 1 row containing 1st meal
    :param meal2_df: df with 1 row containing 2nd meal
    :param meal3_df: df with 1 row containing 3rd meal
    :return: a list of inputs for the loss function - [O, M, K, DT, D, RD, R, C, V1, V2, V3, G1, G2, G3, A1, A2, A3, S1, S2, S3, PH1, PH2, PH3, PS1, PS2, PS3]
    """
    kosher1, vegetarian1, gluten_free1, alcohol_free1, spicy1, max_price1, rating1, hungry1, cuisines1, weekday = \
        diner1_inputs
    kosher2, vegetarian2, gluten_free2, alcohol_free2, spicy2, max_price2, rating2, hungry2, cuisines2, weekday = \
        diner2_inputs
    kosher3, vegetarian3, gluten_free3, alcohol_free3, spicy3, max_price3, rating3, hungry3, cuisines3, weekday = \
        diner3_inputs

    # Group constraints:
    diners_kosher = False if (kosher1 == 0 and kosher2 == 0 and kosher3 == 0) else True
    diners_avg_rating = np.mean((rating1, rating2, rating3))
    hungry_diners = True if np.sum((hungry1, hungry2, hungry3)) >= 2 else False
    rest_cuisines = rest['food categories'].values[0].split('---')
    diner1_cui = 1 if len([meal for meal in cuisines1 if meal in rest_cuisines]) > 0 else 0
    diner2_cui = 1 if len([meal for meal in cuisines2 if meal in rest_cuisines]) > 0 else 0
    diner3_cui = 1 if len([meal for meal in cuisines3 if meal in rest_cuisines]) > 0 else 0

    O = 1 if weekday in rest['opening days'].values[0] else 0
    temp_price = 0
    number_of_meals = 0
    if meal1_df is not None:
        temp_price = meal1_df['price'].values[0]
        number_of_meals += 1
    if meal2_df is not None:
        temp_price += meal2_df['price'].values[0]
        number_of_meals += 1
    if meal3_df is not None:
        temp_price += meal3_df['price'].values[0]
        number_of_meals += 1
    M = 1 if (temp_price >= 100) else 0  # TODO replace with order min
    K = 0 if diners_kosher and not rest['kosher'].values[0] else 1
    DT = rest['delivery estimation [minutes]'].values[0] + rest['prep estimation [minutes]'].values[0]
    D = 0 if (hungry_diners and rest['delivery estimation [minutes]'].values[0] +
              rest['prep estimation [minutes]'].values[0] >= HUNGRY_MINUTES) else 1
    RD = rest['rating'].values[0] - diners_avg_rating
    R = 1 if diners_avg_rating <= rest['rating'].values[0] else 0
    C = diner1_cui + diner2_cui + diner3_cui

    # individual constraints:
    diner_delivery_cost = + rest['delivery price'].values[0] / 3
    price1, price2, price3 = 200, 200, 200
    if meal1_df is not None:
        price1 = meal1_df['price'].values[0]
    if meal2_df is not None:
        price2 = meal2_df['price'].values[0]
    if meal3_df is not None:
        price3 = meal3_df['price'].values[0]
    V1, V2, V3 = 1, 1, 1
    if meal1_df is not None and vegetarian1 == MUST and not meal1_df['vegetarian'].values[0]:
        V1 = 0
    if meal2_df is not None and vegetarian2 == MUST and not meal2_df['vegetarian'].values[0]:
        V2 = 0
    if meal3_df is not None and vegetarian3 == MUST and not meal3_df['vegetarian'].values[0]:
        V3 = 0

    G1, G2, G3 = 1, 1, 1
    if meal1_df is not None and gluten_free1 == MUST and not meal1_df['GF'].values[0]:
        G1 = 0
    if meal2_df is not None and gluten_free2 == MUST and not meal2_df['GF'].values[0]:
        G2 = 0
    if meal3_df is not None and gluten_free3 == MUST and not meal3_df['GF'].values[0]:
        G3 = 0

    A1, A2, A3 = 1, 1, 1
    if meal1_df is not None and alcohol_free1 == MUST and meal1_df['alcohol_percentage'].values[0] > 0:
        A1 = 0
    if meal2_df is not None and alcohol_free2 == MUST and meal1_df['alcohol_percentage'].values[0] > 0:
        A2 = 0
    if meal3_df is not None and alcohol_free3 == MUST and meal1_df['alcohol_percentage'].values[0] > 0:
        A3 = 0

    spicy_meal1, spicy_meal2, spicy_meal3 = meal1_df['spicy'].values[0] if meal1_df is not None else -1, \
                                            meal2_df['spicy'].values[0] if meal2_df is not None else -1, \
                                            meal3_df['spicy'].values[0] if meal3_df is not None else -1

    S1 = 0 if (spicy1 == MUST and not spicy_meal1) or (spicy1 == MUST_NOT and spicy_meal1) else 1
    S2 = 0 if (spicy2 == MUST and not spicy_meal2) or (spicy2 == MUST_NOT and spicy_meal2) else 1
    S3 = 0 if (spicy3 == MUST and not spicy_meal3) or (spicy3 == MUST_NOT and spicy_meal3) else 1
    PH1 = 1 if price1 + diner_delivery_cost <= max_price1 else 0
    PH2 = 1 if price2 + diner_delivery_cost <= max_price2 else 0
    PH3 = 1 if price3 + diner_delivery_cost <= max_price3 else 0
    PS1 = max_price1 - price1 if PH1 == 1 else 0
    PS2 = max_price2 - price2 if PH2 == 1 else 0
    PS3 = max_price3 - price3 if PH3 == 1 else 0

    return [O, M, K, DT, D, RD, R, C, V1, V2, V3, G1, G2, G3, A1, A2, A3, S1, S2, S3, PH1, PH2, PH3, PS1, PS2, PS3,
            number_of_meals]


def gain(O, M, K, DT, D, RD, R, C, V1, V2, V3, G1, G2, G3, A1, A2, A3, S1, S2, S3, PH1, PH2, PH3, PS1, PS2,
         PS3, number_of_meals) -> float:
    """
    Gain function that can be used to evaluate the fitness of a solution.
    based on variables per restaurant (open, minimal order price, kosher, delivery time, rating, cuisines) and per diner (vegetarian, gluten free, alcohol_free, spicy, price).
    :param O: (open) - 1 if the restaurant open 0 otherwise
    :param M: (minimal order price) - 1 if the meal's combination surpasses the restaurant's minimal order price, 0 otherwise
    :param K: (kosher) - 1 if at least one diner eats kosher and the restaurant is kosher or none of the diners eat kosher, 0 otherwise
    :param DT: (delivery time) - HUNGRY_MINUTES - (delivery time + preparation time in minutes)
    :param D: (delivery) - based on avg hunger level among the group. if hunger level is high - 1 if the meal is ready in less than 30 minutes, 0 otherwise.
     if hunger level is low - 1 if the meal is ready in less than 60 minutes, 0 otherwise.
    :param RD: (rating difference) - float on a scale of -9 to 9 - the difference between the restaurant's rating and the average rating of the diners
    :param R: (rating) - 1 if the restaurant is above avg desired minimal rating among the group or does not have a rating, 0 otherwise
    :param C: (cuisines) - int  0-3 according to the amount of diners who prefer a cuisine that the restaurant offers.
    :param V1, V2, V3: (vegetarian) 1 if the meal matches the vegetarian desires of the diner, 0 otherwise
    :param G1, G2, G3: (gluten free) - 1 if the meal matches the gluten desires of the diner, 0 otherwise
    :param A1, A2, A3: (alcohol free) - 1 if the meal matches the alcohol desires of the diner, 0 otherwise
    :param S1, S2, S3: (spicy) - 1 if the meal matches the spiciness desires of the diner, 0 otherwise
    :param PH1, PH2, PH3: (price hard) - 1 if the meal is lower than the diner's desired maximal meal price, 0 otherwise
    :param PS1, PS2, PS3: (price soft) - difference between diner's maximal price and meals price,
    0 if the meal's price is higher than the diner's desired maximal meal price
    :return: The gain value of the given inputs, according to the desired hard and soft constraints.
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
    gain_value = 0
    global count
    gain_value += O + M + K + V1 + V2 + V3 + G1 + G2 + G3 + A1 + A2 + A3 + PH1 + PH2 + PH3 + number_of_meals
    """
    soft constraints:
    - delivery time in minutes (DT)
    - delivery time matches diners' hunger level (D)
    - rating difference (RD)
    - rating (R)
    - cuisines preferences (C)
    - spiciness preferences (S1, S2, S3)
    - price differences (PS1, PS2, PS3)
    """
    DELIVERY_W = 10
    RATING_W = 10
    CUISINE_W = 10
    SPICY_W = 10 / 3
    PRICE_W = 10 / 6

    delivery_gain = (D * DT) / 60 * DELIVERY_W
    rating_gain = RD * RATING_W
    price_gain = ((PS1 + PS2 + PS3) / 3) / 10 * PRICE_W
    cuisine_gain = C * CUISINE_W
    spicy_gain = (S1 + S2 + S3) * SPICY_W
    gain_value += (delivery_gain + rating_gain + price_gain + cuisine_gain + spicy_gain)
    return gain_value


def check_hard_constraints(O, M, K, DT, D, RD, R, C, V1, V2, V3, G1, G2, G3, A1, A2, A3, S1, S2, S3, PH1, PH2, PH3,
                           PS1, PS2,
                           PS3):
    """Gain function that can be used to evaluate the fitness of a solution.
    based on variables per restaurant (open, minimal order price, kosher, delivery time, rating, cuisines) and per diner (vegetarian, gluten free, alcohol_free, spicy, price).
    :param O: (open) - 1 if the restaurant open 0 otherwise
    :param M: (minimal order price) - 1 if the meal's combination surpasses the restaurant's minimal order price, 0 otherwise
    :param K: (kosher) - 1 if at least one diner eats kosher and the restaurant is kosher or none of the diners eat kosher, 0 otherwise
    :param DT: (delivery time) - HUNGRY_MINUTES - (delivery time + preparation time in minutes)
    :param D: (delivery) - based on avg hunger level among the group. if hunger level is high - 1 if the meal is ready in less than 30 minutes, 0 otherwise.
     if hunger level is low - 1 if the meal is ready in less than 60 minutes, 0 otherwise.
    :param RD: (rating difference) - float on a scale of -9 to 9 - the difference between the restaurant's rating and the average rating of the diners
    :param R: (rating) - 1 if the restaurant is above avg desired minimal rating among the group or does not have a rating, 0 otherwise
    :param C: (cuisines) - int  0-3 according to the amount of diners who prefer a cuisine that the restaurant offers.
    :param V1, V2, V3: (vegetarian) 1 if the meal matches the vegetarian desires of the diner, 0 otherwise
    :param G1, G2, G3: (gluten free) - 1 if the meal matches the gluten desires of the diner, 0 otherwise
    :param A1, A2, A3: (alcohol free) - 1 if the meal matches the alcohol desires of the diner, 0 otherwise
    :param S1, S2, S3: (spicy) - 1 if the meal matches the spiciness desires of the diner, 0 otherwise
    :param PH1, PH2, PH3: (price hard) - 1 if the meal is lower than the diner's desired maximal meal price, 0 otherwise
    :param PS1, PS2, PS3: (price soft) - difference between diner's maximal price and meals price,
    0 if the meal's price is higher than the diner's desired maximal meal price
    :return 1 if hard constraints been completed and 0 if not."""
    hard_constraints = O * M * K * V1 * V2 * V3 * G1 * G2 * G3 * A1 * A2 * A3 * PH1 * PH2 * PH3
    if hard_constraints == 0:  # if at least 1 hard constraint is not met, return a loss value of 0
        return 0
    else:
        return 1


# ---------------------------------------------- Functions -----------------------------------------------------------
def get_diners_constraints(filename):
    """
    To optimize a meal order for a group of 3, the group must provide a formatted file
    (see format instructions at the end of the example file) that contains provide
    10 details about each diner's preferences.
    This function takes a such formatted input file and returns 3 constraint list (one for each diner).
    :param filename: the name of the file containing the diners constraints.
    :return: 3 lists, one10-item list  for each diner that follows the following format:
    0 - kosher (int - 1 for kosher / 0 for doesn't matter)
    1 - vegetarian (int - 1 for vegetarian / 0 for doesn't matter)
    2 - gluten free (int - 1 for GF / 0 for doesn't matter)
    3 - alcohol free (int - 1 for alcohol free / 0 for doesn't matter)
    4 - prefer spicy (int - 2 for not spicy / 1 for spicy / 0 for doesn't matter)
    5 - max price (int - in ILS)
    6 - min rating (int - range from 1 to 10)
    7 - hunger level (int - 1 for very hungry / 0 for not so hungry)
    8 - desired cuisines (list(str) - list of strings out of a predefined list)
    9 - weekday (str - lowercase string from sunday to saturday)
    """
    diner1, diner2, diner3 = [], [], []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:9]:
            diner1.append(int(line.strip().split(" ")[-1]))
        diner1.append(list(lines[9].split("[")[-1][:-2].split(" ")))
        diner1.append((lines[10].split(" ")[-1].strip()))
        for line in lines[13:21]:
            diner2.append(int(line.strip().split(" ")[-1]))
        diner2.append(list(lines[21].split("[")[-1][:-2].split(" ")))
        diner2.append((lines[22].strip().split(" ")[-1].strip()))
        for line in lines[25:33]:
            diner3.append(int(line.strip().split(" ")[-1]))
        diner3.append(list(lines[33].split("[")[-1][:-2].split(" ")))
        diner3.append((lines[34].strip().split(" ")[-1].strip()))

    return diner1, diner2, diner3


def init_problem(rest_df, meals_df, diner1, diner2, diner3):
    restaurants = get_rest_lst(rest_df)
    meals = get_menus_meals(meals_df, restaurants)
    history = Data(restaurants, meals)
    action_obj = Action(history)
    constraints = diner1, diner2, diner3
    init_state = State(rest=None, meals=[])
    return WoltProblem(
        history, action_obj, init_state, constraints, data_rests, data_menu)


def run_algorithm(algo, input, rest_df, meals_df, diner1, diner2, diner3):
    problem = init_problem(rest_df, meals_df, diner1, diner2, diner3)
    start = timer()
    result = algo(problem, input)
    end = timer()
    if result is None:
        print("goal not found")
        return
    else:
        args = gainFunction.user_inputs_to_gain_function_inputs(
            constraints[0],
            constraints[1],
            constraints[2],
            data_rests[data_rests["name"] == result.state.restaurant],
            data_menu[data_menu["name"] == result.state.meals[0]],
            # and data_menu["rest_name"] == state.restaurant],
            data_menu[data_menu["name"] == result.state.meals[1]],
            # and data_menu["rest_name"] == state.restaurant],
            data_menu[data_menu["name"] == result.state.meals[2]]
        )
        # delete when done
        print(
            f"----------------{end - start} sec, ---------- Loss = {gainFunction.gain(*args)}---- cost = {result.cost}--\n "
            f'{result.state.restaurant} - {data_rests[data_rests["name"] == result.state.restaurant].values}\n'
            f"-------------------------------------------\n"
            f'{result.state.meals[0]} - {data_menu[(data_menu["rest_name"] == result.state.restaurant) & (data_menu["name"] == result.state.meals[0])].values}\n'
            f"-------------------------------------------\n"
            f'{result.state.meals[1]} - {data_menu[(data_menu["rest_name"] == result.state.restaurant) & (data_menu["name"] == result.state.meals[1])].values}\n'
            f"-------------------------------------------\n"
            f'{result.state.meals[2]} - {data_menu[(data_menu["rest_name"] == result.state.restaurant) & (data_menu["name"] == result.state.meals[2])].values}\n'
            f"-------------------------------------------\n"
            f"{constraints}"
        )
    return result.state.restaurant, result.state.meals, end - start


def DFSAlgorithm(rest_df, meals_df, diner1, diner2, diner3):
    """
        DFS algorithm wrapper function. returns the solution that the algorithm chose (restaurant and 3 meals) and it's runtime.
        :param rest_df: restaurant dataframe
        :param meals_df: meals dataframe
        :param diner1: list of 1st diner preferences
        :param diner2: list of 2nd diner preferences
        :param diner3: list of 3rd diner preferences
        :return: chosen restaurant dataframe (single row), 3 chosen meals dataframes (single row each), runtime (float).
    """
    return run_algorithm(depth_first, True, rest_df, meals_df, diner1, diner2, diner3)


def UCSAlgorithm(rest_df, meals_df, diner1, diner2, diner3):
    """
        UCS algorithm wrapper function. returns the solution that the algorithm chose (restaurant and 3 meals) and it's runtime.
        :param rest_df: restaurant dataframe
        :param meals_df: meals dataframe
        :param diner1: list of 1st diner preferences
        :param diner2: list of 2nd diner preferences
        :param diner3: list of 3rd diner preferences
        :return: chosen restaurant dataframe (single row), 3 chosen meals dataframes (single row each), runtime (float).
        """
    return run_algorithm(uniform_cost, True, rest_df, meals_df, diner1, diner2, diner3)


def AstarAlgorithm(rest_df, meals_df, diner1, diner2, diner3):
    """
        A star algorithm wrapper function. returns the solution that the algorithm chose (restaurant and 3 meals) and it's runtime.
        :param rest_df: restaurant dataframe
        :param meals_df: meals dataframe
        :param diner1: list of 1st diner preferences
        :param diner2: list of 2nd diner preferences
        :param diner3: list of 3rd diner preferences
        :return: chosen restaurant dataframe (single row), 3 chosen meals dataframes (single row each), runtime (float).
        """
    return run_algorithm(astar, True, rest_df, meals_df, diner1, diner2, diner3)


def HillClimbingAlgorithm(rest_df, meals_df, diner1, diner2, diner3):
    """
        Hill climbing algorithm wrapper function. returns the solution that the algorithm chose (restaurant and 3 meals) and it's runtime.
        :param rest_df: restaurant dataframe
        :param meals_df: meals dataframe
        :param diner1: list of 1st diner preferences
        :param diner2: list of 2nd diner preferences
        :param diner3: list of 3rd diner preferences
        :return: chosen restaurant dataframe (single row), 3 chosen meals dataframes (single row each), runtime (float).
        """
    return run_algorithm(hill_climbing, 10000, rest_df, meals_df, diner1, diner2, diner3)


def StochasticHillClimbingAlgorithm(rest_df, meals_df, diner1, diner2, diner3):
    """
        Stochastic hill climbing algorithm wrapper function. returns the solution that the algorithm chose (restaurant and 3 meals) and it's runtime.
        :param rest_df: restaurant dataframe
        :param meals_df: meals dataframe
        :param diner1: list of 1st diner preferences
        :param diner2: list of 2nd diner preferences
        :param diner3: list of 3rd diner preferences
        :return: chosen restaurant dataframe (single row), 3 chosen meals dataframes (single row each), runtime (float).
        """
    return run_algorithm(hill_climbing_stochastic, 100000, rest_df, meals_df, diner1, diner2, diner3)


def SimulatedAnnealingAlgorithm(rest_df, meals_df, diner1, diner2, diner3):
    """
        simulated annealing algorithm wrapper function. returns the solution that the algorithm chose (restaurant and 3 meals) and it's runtime.
        :param rest_df: restaurant dataframe
        :param meals_df: meals dataframe
        :param diner1: list of 1st diner preferences
        :param diner2: list of 2nd diner preferences
        :param diner3: list of 3rd diner preferences
        :return: chosen restaurant dataframe (single row), 3 chosen meals dataframes (single row each), runtime (float).
        """
    return run_algorithm(simulated_annealing, 1000, rest_df, meals_df, diner1, diner2, diner3)


# ----------------------------------------------  State --------------------------------------------------------------
class State:
    def __init__(self, rest: str, meals: List):
        self.restaurant = rest
        self.meals = meals

    def __eq__(self, other):
        if self.restaurant is None:
            if other.restaurant is None:
                return True
            else:
                return False
        elif self.restaurant is not None:
            if other.restaurant is None:
                return False
            if str(self)== str(other):
                return True
        else:
            return True

    def __str__(self):
        if self.restaurant is None:
            return ""
        return self.restaurant + " " + (" ").join(self.meals)

    def __hash__(self):
        return hash(str(self))


class Action:
    CHANGE_REST = 0
    ADD_MEAL = 1
    ALL_RESTS = None
    ALL_MEALS = None

    def __init__(self, data):
        self.data = data
        self.ALL_RESTS = [(self.CHANGE_REST, i) for i in range(len(data.restaurants))]
        self.generate_change_meal()

    def generate_change_meal(self):
        self.ALL_MEALS = {}
        for k, rest in enumerate(self.data.restaurants):
            self.ALL_MEALS[rest] = [
                (self.ADD_MEAL, i) for i in range(len(self.data.meals[k]))
            ]

    def get_actions(self, state, users=3):
        if not state.restaurant and not state.meals:
            return self.ALL_RESTS
        elif len(state.meals) < users:
            return self.ALL_MEALS[state.restaurant]
        else:
            return []


# ---------------------------------------------- Data ----------------------------------------------------------------


class Data:
    def __init__(self, restaurants, meals):
        self.history_states = set()
        self.restaurants = restaurants
        self.meals = meals
        self.rest_dict = {rest: i for i, rest in enumerate(restaurants)}

    def check_state(self, state_tuple: Tuple[str]):
        return state_tuple in self.history_states

    def add_state(self, state_tuple: Tuple[str]):
        self.history_states.add(state_tuple)

    def check_rest(self, rest):
        for key in self.history_states:
            if key[0] == rest:
                return False
        return True


def get_rest_lst(data_frame):
    rests = list(data_frame["name"])
    return rests


def get_menus_meals(data_frame, rests):
    our_meals = []
    for rest in rests:
        our_meals.append(list(data_frame["name"][data_frame["rest_name"] == rest]))
    return our_meals


# ---------------------------------------------- Main class problem ---------------------------------------------------


class WoltProblem(SearchProblem):
    def __init__(
            self, data, action_obj, init_state, constraints, data_rests, data_menu
    ):
        super().__init__(init_state)
        self.data = data
        self.action_obj = action_obj
        self.constraints = constraints
        self.data_rests, self.data_menu = data_rests, data_menu
        diner1_inputs, diner2_inputs, diner3_inputs = constraints
        (
            self.kosher1,
            self.vegetarian1,
            self.gluten_free1,
            self.alcohol_free1,
            self.spicy1,
            self.max_price1,
            self.rating1,
            self.hungry1,
            self.cuisines1,
            self.weekday,
        ) = diner1_inputs
        (
            self.kosher2,
            self.vegetarian2,
            self.gluten_free2,
            self.alcohol_free2,
            self.spicy2,
            self.max_price2,
            self.rating2,
            self.hungry2,
            self.cuisines2,
            self.weekday,
        ) = diner2_inputs
        (
            self.kosher3,
            self.vegetarian3,
            self.gluten_free3,
            self.alcohol_free3,
            self.spicy3,
            self.max_price3,
            self.rating3,
            self.hungry3,
            self.cuisines3,
            self.weekday,
        ) = diner3_inputs

        self.diners_kosher = (
            False
            if (self.kosher1 == 0 and self.kosher2 == 0 and self.kosher3 == 0)
            else True
        )
        self.diners_avg_rating = np.mean((self.rating1, self.rating2, self.rating3))
        self.hungry_diners = (
            True if np.sum((self.hungry1, self.hungry2, self.hungry3)) >= 2 else False
        )

    def result(self, state, action):
        if action[0] == Action.ADD_MEAL:
            return self.change_meal(state, action[1])
        if action[0] == Action.CHANGE_REST:
            return self.change_rest_state(action[1])

    def is_goal(self, state):
        print(f"is goal func state : rest = {state.restaurant}, meals = {state.meals}")
        if len(state.meals) < 3:
            return False
        if self.check_constraints(state):
            return False
        return True

    def check_constraints(self, state):
        args = gainFunction.user_inputs_to_gain_function_inputs(
            self.constraints[0],
            self.constraints[1],
            self.constraints[2],
            self.data_rests[self.data_rests["name"] == state.restaurant],
            self.data_menu[self.data_menu["name"] == state.meals[0]],
            # and self.data_menu["rest_name"] == state.restaurant],
            self.data_menu[self.data_menu["name"] == state.meals[1]],
            # and self.data_menu["rest_name"] == state.restaurant],
            self.data_menu[self.data_menu["name"] == state.meals[2]],
            # and self.data_menu["rest_name"] == state.restaurant],
        )
        return gainFunction.gain(*args) == 0

    def value(self, state):
        if state.restaurant is None:
            return 0
        dfs = [None, None, None]
        for i, meal in enumerate(state.meals):
            dfs[i] = (self.data_menu[self.data_menu["name"] == meal])
        args = user_inputs_to_gain_function_inputs(
            self.constraints[0],
            self.constraints[1],
            self.constraints[2],
            self.data_rests[self.data_rests["name"] == state.restaurant],
            *dfs
        )
        return gain(*args)

        # val = 0
        # if state.restaurant is None:
        #     val += 4000
        # if len(state.meals) < 3:
        #     val += 2300 * (3 - len(state.meals))
        # if state.restaurant:
        #     val += self.restaurant_value(state)
        # for meal in state.meals:
        #     val += self.meal_value(
        #         state
        #     )
        # return 1 / val

    def meal_value(self, state):
        return self.meal_cost(state.meals[-1], state.restaurant, len(state.meals) - 1) - 10

    def restaurant_value(self, state):
        return self.restaurant_cost(state.restaurant) - 10

    def crossover(self, state1, state2):
        pass

    def mutate(self, state):
        pass

    def generate_random_state(self, users=3):
        rest_index = random.randint(0, len(self.data.restaurants) - 1)
        length_meals = len(self.data.meals[rest_index])
        if length_meals == 1:
            rest_index = random.randint(0, len(self.data.restaurants) - 1)
        num_of_meals = random.randint(0, users)
        return State(
            self.data.restaurants[rest_index],
            [
                self.data.meals[rest_index][random.randint(0, length_meals - 1)]
                for j in range(num_of_meals)
            ],
        )

    def cost(self, state, action, state2):
        if action[0] == Action.ADD_MEAL:
            return self.meal_cost(
                state2.meals[-1], state2.restaurant, len(state2.meals) - 1
            )
        if action[0] == Action.CHANGE_REST:
            return self.restaurant_cost(state2.restaurant)

    def actions(self, state, random=True):
        action = self.action_obj.get_actions(state).copy()
        if random:
            shuffle(action)
        return action

    def heuristic(self, state):
        h_cost = 0
        if state.restaurant is None:
            return h_cost
        h_cost += self.rest_heuristic(state)
        # for index, meal in state.meals:
        #     h_cost += self.meal_heuristic(meal, index)
        return h_cost

    def rest_heuristic(self, state):
        rest_cost = 0
        all_rest_meals_df = self.data_menu[self.data_menu["rest_name"] == state.restaurant]
        index_diner = len(state.meals)

        gluten_free = len(all_rest_meals_df[all_rest_meals_df["GF"] == True])
        not_gluten_meals = len(all_rest_meals_df) - gluten_free
        vegi = len(all_rest_meals_df[all_rest_meals_df["vegetarian"] == True])
        not_vegi = len(all_rest_meals_df) - vegi
        spicy = len(all_rest_meals_df[all_rest_meals_df["spicy"] == True])
        not_spicy = len(all_rest_meals_df) - spicy
        alcohol = len(all_rest_meals_df[all_rest_meals_df["alcohol_percentage"] > 0])
        no_alcohol = len(all_rest_meals_df) - alcohol
        categories = self.data_rests[self.data_rests["name"] == state.restaurant]["food categories"].values[0]

        if index_diner == 0:
            rest_cost += (not_gluten_meals if bool(self.gluten_free1) else gluten_free) / 10
            rest_cost += (not_vegi if bool(self.vegetarian1) else vegi) / 10
            rest_cost += (alcohol if bool(self.alcohol_free1) else no_alcohol) / 100
            rest_cost += (not_spicy if bool(self.spicy1) else spicy) / 20
            rest_cost += len(all_rest_meals_df[all_rest_meals_df["price"] > self.max_price1]) / 20
            cat_count = 0
            for cat in self.cuisines1:
                if cat not in categories:
                    cat_count += 1
            if cat_count == 1:
                rest_cost += 10
            else:
                rest_cost -= 10
        elif index_diner == 1:
            rest_cost += (not_gluten_meals if bool(self.gluten_free2) else gluten_free) / 10
            rest_cost += (not_vegi if bool(self.vegetarian2) else vegi) / 10
            rest_cost += (alcohol if bool(self.alcohol_free2) else no_alcohol) / 100
            rest_cost += (not_spicy if bool(self.spicy2) else spicy) / 20
            rest_cost += len(all_rest_meals_df[all_rest_meals_df["price"] > self.max_price2]) / 20
            cat_count = 0
            for cat in self.cuisines2:
                if cat not in categories:
                    cat_count += 1
            if cat_count == 1:
                rest_cost += 10
            else:
                rest_cost -= 10

        elif index_diner == 2:
            rest_cost += (not_gluten_meals if bool(self.gluten_free3) else gluten_free) / 10
            rest_cost += (not_vegi if bool(self.vegetarian3) else vegi) / 10
            rest_cost += (alcohol if bool(self.alcohol_free3) else no_alcohol) / 100
            rest_cost += (not_spicy if bool(self.spicy3) else spicy) / 20
            rest_cost += len(all_rest_meals_df[all_rest_meals_df["price"] > self.max_price3]) / 20
            cat_count = 0
            for cat in self.cuisines3:
                if cat not in categories:
                    cat_count += 1
            if cat_count == 1:
                rest_cost += 10
            else:
                rest_cost -= 10
        return rest_cost

    def meal_heuristic(self, meal, diner_index):
        pass

    def change_rest_state(self, i) -> State:
        return State(self.data.restaurants[i], [])

    def change_meal(self, state, index_meal) -> State:
        if state.meals:
            meals = [
                *state.meals[:],
                self.data.meals[self.data.rest_dict[state.restaurant]][index_meal],
            ]
        else:
            meals = [self.data.meals[self.data.rest_dict[state.restaurant]][index_meal]]
        new_state = State(state.restaurant, meals)
        return new_state

    def meal_cost(self, meal_name, restaurant_name, meal_index):
        meal_df = self.data_menu[
            (self.data_menu["rest_name"] == restaurant_name)
            & (self.data_menu["name"] == meal_name)
            ].reset_index(drop=True)
        return (
                10 * self.check_veg(meal_index, meal_df)
                + 10 * self.check_gf(meal_index, meal_df)
                + self.check_price(meal_index, meal_df)
                + self.check_spicy(meal_index, meal_df)
                + self.check_alcohol(meal_index, meal_df)
        )

    def restaurant_cost(self, restaurant_name):
        rest = self.data_rests[self.data_rests["name"] == restaurant_name].reset_index(
            drop=True
        )
        cost = 0
        un_satisfied = self.food_category_cost(rest)
        # rating hungry
        if un_satisfied > 2:
            cost += MAX_COST_REST
        else:
            cost += 50 * un_satisfied
        cost += self.rating_cost(rest)
        cost += self.hungry_loss(rest)
        cost += self.kosher(rest)
        return cost

    def food_category_cost(self, rest_df):
        rest_cuisines = rest_df["food categories"].values[0].split("---")
        diner1_cui = (
            0
            if len([meal for meal in self.cuisines1 if meal[1:-1] in rest_cuisines]) > 0
            else 1
        )
        diner2_cui = (
            0
            if len([meal for meal in self.cuisines2 if meal[1:-1] in rest_cuisines]) > 0
            else 1
        )
        diner3_cui = (
            0
            if len([meal for meal in self.cuisines3 if meal[1:-1] in rest_cuisines]) > 0
            else 1
        )
        gain = diner1_cui + diner2_cui + diner3_cui
        return gain

    def check_veg(self, meal_index, meal_df):
        if meal_index == 0:
            return (
                MAX_COST_MEAL
                if bool(self.vegetarian1) != meal_df["vegetarian"].values[0]
                else 0
            )
        elif meal_index == 1:
            return (
                MAX_COST_MEAL
                if bool(self.vegetarian2) != meal_df["vegetarian"].values[0]
                else 0
            )
        elif meal_index == 2:
            return (
                MAX_COST_MEAL
                if bool(self.vegetarian3) != meal_df["vegetarian"].values[0]
                else 0
            )

    def check_gf(self, meal_index, meal_df):
        if meal_index == 0:
            return (
                MAX_COST_MEAL
                if bool(self.gluten_free1) != meal_df["GF"].values[0]
                else 0
            )
        elif meal_index == 1:
            return (
                MAX_COST_MEAL
                if bool(self.gluten_free2) != meal_df["GF"].values[0]
                else 0
            )
        elif meal_index == 2:
            return (
                MAX_COST_MEAL
                if bool(self.gluten_free3) != meal_df["GF"].values[0]
                else 0
            )

    def check_spicy(self, meal_index, meal_df):
        if meal_index == 0:
            return (
                MAX_COST_MEAL / 2
                if bool(self.spicy1) != meal_df["spicy"].values[0]
                else 0
            )
        elif meal_index == 1:
            return (
                MAX_COST_MEAL / 2
                if bool(self.spicy2) != meal_df["spicy"].values[0]
                else 0
            )
        elif meal_index == 2:
            return (
                MAX_COST_MEAL / 2
                if bool(self.spicy3) != meal_df["spicy"].values[0]
                else 0
            )

    def check_price(self, meal_index, meal_df):
        if meal_index == 0:
            return (
                meal_df["price"].values[0] - self.max_price1
                if self.max_price1 < meal_df["price"].values[0]
                else 0
            )
        elif meal_index == 1:
            return (
                meal_df["price"].values[0] - self.max_price2
                if self.max_price2 < meal_df["price"].values[0]
                else 0
            )
        elif meal_index == 2:
            return (
                meal_df["price"].values[0] - self.max_price3
                if self.max_price3 < meal_df["price"].values[0]
                else 0
            )

    def hungry_loss(self, rest):
        if np.sum((self.hungry1, self.hungry2, self.hungry3)) >= 2:
            time_prep = (
                    rest["delivery estimation [minutes]"].values[0]
                    + rest["prep estimation [minutes]"].values[0]
            )
            if time_prep > gainFunction.HUNGRY_MINUTES:
                return time_prep - gainFunction.HUNGRY_MINUTES
        return 0

    def rating_cost(self, rest):
        rating = [self.rating1, self.rating2, self.rating3]
        rest_rating = rest["rating"].values[0]
        diners_avg_rating = np.mean((self.rating1, self.rating2, self.rating3))
        if len([1 for rate in rating if rate > rest_rating]) >= 2:
            return MAX_COST_REST
        elif diners_avg_rating > rest["rating"].values[0]:
            return diners_avg_rating - rest["rating"].values[0]
        return 0

    def check_alcohol(self, meal_index, meal_df):
        if meal_index == 0:
            return (
                MAX_COST_MEAL / 2
                if bool(self.alcohol_free1) and meal_df["alcohol_percentage"].values[0] != 0
                else 0
            )
        elif meal_index == 1:
            return (
                MAX_COST_MEAL / 2
                if bool(self.alcohol_free2) and meal_df["alcohol_percentage"].values[0] != 0
                else 0
            )
        elif meal_index == 2:
            return (
                MAX_COST_MEAL / 2
                if bool(self.alcohol_free3) and meal_df["alcohol_percentage"].values[0] != 0
                else 0
            )

    def kosher(self, rest):
        diners_kosher = (
            False
            if (self.kosher1 == 0 and self.kosher2 == 0 and self.kosher3 == 0)
            else True
        )
        if diners_kosher and not rest["kosher"].values[0]:
            return MAX_COST_REST
        return 0


# ---------------------------------------------- main  ---------------------------------------------------------------
if __name__ == "__main__":
    df = dataFrameParser.WoltParser([], init_files=False)
    df.get_dfs()
    data_rests = df.general_df
    data_menu = df.menus_df
    constraints = [*get_diners_constraints(sys.argv[1])]
    StochasticHillClimbingAlgorithm(data_rests, data_menu, *constraints)
