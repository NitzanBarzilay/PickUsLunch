import random
import sys
from random import shuffle
from timeit import default_timer as timer
from typing import List, Tuple

import numpy as np
from simpleai.search import (SearchProblem, breadth_first, depth_first,
                             uniform_cost, astar)

import dataFrameParser
import gainFunction
from main import get_diners_constraints

MAX_COST_REST = 1000
MAX_COST_MEAL = 100


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
            if self.restaurant == other.restaurant and self.meals == self.meals:
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
        args = LossFunc.user_inputs_to_gain_function_inputs(
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
        return LossFunc.gain(*args) == 0

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
            if time_prep > LossFunc.HUNGRY_MINUTES:
                return time_prep - LossFunc.HUNGRY_MINUTES
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
    start = timer()

    result = astar(problem, graph_search=True)
    if result is None:
        print("goal not found")
    else:
        args = LossFunc.user_inputs_to_gain_function_inputs(
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
        end = timer()
        print(
            f"----------------{end - start} sec, ---------- Loss = {LossFunc.gain(*args)}---- cost = {result.cost}--\n "
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
