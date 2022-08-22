import random
import sys
from typing import List, Tuple

import numpy as np
from simpleai.search import SearchProblem, breadth_first, depth_first, uniform_cost

import dataFrameParser
import LossFunc
from main import get_diners_constraints


# ----------------------------------------------  State --------------------------------------------------------------
class State:
    def __init__(self, rest: str, meals: List):
        self.restaurant = rest
        self.meals = meals


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
            return True
        return False

    def check_constraints(self, state):
        args \
            = LossFunc.user_inputs_to_loss_function_inputs(self.constraints[0],
                                                           self.constraints[1], self.constraints[2],
                                                           state.restaurant, state.meals[0], state.meals[1],
                                                           state.meals[2])
        return LossFunc.loss(*args) != 0

    def value(self, state):
        pass

    def crossover(self, state1, state2):
        pass

    def mutate(self, state):
        pass

    def generate_random_state(self, users=3):
        rest_index = random.randint(0, len(self.data.restaurants) - 1)
        length_meals = len(self.data.meals[rest_index])
        num_of_meals = random.randint(0, users)
        return State(
            self.data.restaurants[rest_index],
            [
                self.data.meals[rest_index][random.randint(0, length_meals)]
                for j in range(num_of_meals)
            ]
        )

    def cost(self, state, action, state2):
        if action[0] == Action.ADD_MEAL:
            return self.meal_cost(state2.meals[-1], state2.restaurant, len(state2.meals) - 1)
        if action[0] == Action.CHANGE_REST:
            return self.restaurant_cost(state2.restaurant)

    def actions(self, state):
        return self.action_obj.get_actions(state)

    def heuristic(self, state):
        pass

    def change_rest_state(self, i) -> State:
        return State(self.data.restaurants[i], [])

    def change_meal(self, state, index_meal) -> State:
        if state.meals:
            meals = [*state.meals[:], self.data.meals[self.data.rest_dict[state.restaurant]][index_meal]]
        else:
            meals = [self.data.meals[self.data.rest_dict[state.restaurant]][index_meal]]
        new_state = State(state.restaurant, meals)
        return new_state

    def meal_cost(self, meal_name, restaurant_name, meal_index):
        meal_df = self.data_menu[
            (self.data_menu["rest_name"] == restaurant_name)
            & (self.data_menu["name"] == meal_name)
            ].reset_index(drop=True)
        return self.check_veg(meal_index, meal_df) + self.check_gf(meal_index, meal_df) + \
            self.check_price(meal_index, meal_df) + self.check_spicy(meal_index, meal_df)

    def restaurant_cost(self, restaurant_name):
        rest = self.data_rests[self.data_rests["name"] == restaurant_name].reset_index(
            drop=True
        )
        return self.food_category_cost(rest) * 10

    def food_category_cost(self, rest_df):

        rest_cuisines = rest_df["food categories"]
        diner1_cui = 0 if len([meal for meal in self.cuisines1 if meal in rest_cuisines]) > 0 else 1
        diner2_cui = 0 if len([meal for meal in self.cuisines2 if meal in rest_cuisines]) > 0 else 1
        diner3_cui = 0 if len([meal for meal in self.cuisines3 if meal in rest_cuisines]) > 0 else 1
        gain = diner1_cui + diner2_cui + diner3_cui
        return gain

    def check_veg(self, meal_index, meal_df):
        if meal_index == 0:
            return 1 if bool(self.vegetarian1) == meal_df["vegetarian"].values[0] else 0
        elif meal_index == 1:
            return 1 if bool(self.vegetarian2) == meal_df["vegetarian"].values[0] else 0
        elif meal_index == 2:
            return 1 if bool(self.vegetarian3) == meal_df["vegetarian"].values[0] else 0

    def check_gf(self, meal_index, meal_df):
        if meal_index == 0:
            return 1 if bool(self.gluten_free1) == meal_df["GF"].values[0] else 0
        elif meal_index == 1:
            return 1 if bool(self.gluten_free2) == meal_df["GF"].values[0] else 0
        elif meal_index == 2:
            return 1 if bool(self.gluten_free3) == meal_df["GF"].values[0] else 0

    def check_spicy(self, meal_index, meal_df):
        if meal_index == 0:
            return 1 if bool(self.spicy1) == meal_df["spicy"].values[0] else 0
        elif meal_index == 1:
            return 1 if bool(self.spicy2) == meal_df["spicy"].values[0] else 0
        elif meal_index == 2:
            return 1 if bool(self.spicy3) == meal_df["spicy"].values[0] else 0

    def check_price(self, meal_index, meal_df):
        if meal_index == 0:
            return 1 if self.max_price1 < meal_df["price"].values[0] else 0
        elif meal_index == 1:
            return 1 if self.max_price2 < meal_df["price"].values[0] else 0
        elif meal_index == 2:
            return 1 if self.max_price3 < meal_df["price"].values[0] else 0


# ---------------------------------------------- main  ---------------------------------------------------------------
if __name__ == "__main__":
    df = dataFrameParser.WoltParser([], init_files=False)
    df.get_dfs()
    data_rests = df.general_df[:2]
    data_menu = df.menus_df
    restaurants = get_rest_lst(data_rests)
    meals = get_menus_meals(data_menu, restaurants)
    history = Data(restaurants, meals)
    action_obj = Action(history)
    print(restaurants)
    print(meals)
    constraints = [*get_diners_constraints(sys.argv[1])]
    init_state = State(rest=None, meals=[])
    problem = WoltProblem(
        history, action_obj, init_state, constraints, data_rests, data_menu
    )
    result = depth_first(problem, True)
    print(f'-------------------------------------------\n{result.state.restaurant}, \n{result.state.meals[0]},\n{ result.state.meals[1]}, '
          f'\n{result.state.meals[2]}, \n{constraints}')
