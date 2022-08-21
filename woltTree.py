import random
import dataFrameParser
from simpleai.search import SearchProblem
from typing import List, Tuple
import LossFunc
from simpleai.search import breadth_first
from main import get_diners_constraints
import sys


# ---------------------------------------------- Helper classes -------------------------------------------------------
class State:
    def __init__(self, rest: str, meals: List):
        self.restaurant = rest
        self.meals = meals


class Action:
    CHANGE_REST = 0
    CHANGE_MEAL = 1

    def __init__(self, history):
        self.history = history

    def get_actions(self, state, users=3):
        actions = []
        index = self.history.restaurants.index(state.restaurant)
        for i in range(len(self.history.restaurants)):
            actions.append((self.CHANGE_REST, i))
        for j in range(len(self.history.meals[index])):
            for k in range(users):
                actions.append((self.CHANGE_MEAL, j, k))
        return actions


class History:
    def __init__(self, restaurants, meals):
        self.history_states = set()
        self.restaurants = restaurants
        self.meals = meals

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

    def __init__(self, history, action_obj, init_state, constraints):
        super().__init__(init_state)
        self.history = history
        self.action_obj = action_obj
        self.constraints = constraints

    def result(self, state, action):
        if action[0] == Action.CHANGE_MEAL:
            return self.change_meal(state, action[1], action[2])
        if action[0] == Action.CHANGE_REST:
            return self.change_rest_state(action[1], len(self.history.meals[action[1]]))

    def is_goal(self, state):
        return self.value(state) != 0

    def value(self, state):
        return LossFunc.loss(*LossFunc.user_inputs_to_loss_function_inputs(self.constraints[0], self.constraints[1],
                                                                           self.constraints[2], state.restaurant,
                                                                           state.meals[0],
                                                                           state.meals[1], state.meals[2]))

    def crossover(self, state1, state2):
        pass

    def mutate(self, state):
        pass

    def generate_random_state(self, users=3):
        rest_index = random.randint(0, len(self.history.restaurants))
        length_meals = len(self.history.meals[rest_index])
        return State(self.history.restaurants[rest_index],
                     [self.history.meals[rest_index][random.randint(0, length_meals)] for j in
                      range(users)])

    def cost(self, state, action, state2):
        return self.value(state2) - self.value(state)

    def actions(self, state):
        return self.action_obj.get_actions(state)

    def heuristic(self, state):
        pass

    def change_rest_state(self, i, length_meals, users=3) -> State:
        return State(self.history.restaurants[i], [self.history.meals[i][random.randint(0, length_meals - 1)] for j in
                                                   range(users)])

    def change_meal(self, state, index_meal, index_diner) -> State:
        rest_index = self.history.restaurants.index(state.restaurant)
        meal = random.randint(0, len(self.history.meals[rest_index]) - 1)
        while meal == index_meal:
            meal = random.randint(0, len(self.history.meals[rest_index]) - 1)
        state.meals[index_diner] = \
            self.history.meals[rest_index][meal]
        return state


# ---------------------------------------------- main  ---------------------------------------------------------------
if __name__ == '__main__':
    df = dataFrameParser.WoltParser([], init_files=False)
    df.get_dfs()
    data_rests = df.df
    data_menu = df.df_menus
    restaurants = get_rest_lst(data_rests)
    meals = get_menus_meals(data_menu, restaurants)
    history = History(restaurants, meals)
    action_obj = Action(history)
    print(restaurants)
    print(meals)
    constraints = [*get_diners_constraints(sys.argv[1])]
    init_state = State(rest="Doron's Jachnun | Tel Aviv",
                       meals=['ג׳חנון תימני עצבני', 'ג׳חנון תימני עצבני', 'ג׳חנון תימני עצבני'])
    problem = WoltProblem(history, action_obj, init_state, constraints)
    result = breadth_first(problem)
    print(result)
