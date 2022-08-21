import random
import dataFrameParser
from simpleai.search import SearchProblem
from typing import List, Tuple
import LossFunc
from simpleai.search import breadth_first
from main import get_diners_constraints
import sys


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

    def generate_change_meal(self):
        self.ALL_MEALS = {}
        for k, rest in enumerate(self.data.restaurants):
            self.ALL_MEALS[rest] = [(self.ADD_MEAL, i) for i in range(len(self.data.meals[k]))]

    def get_actions(self, state, users=3):
        if not state.restaurant and not state.meals:
            return self.ALL_RESTS
        if len(state.meals) < users:
            return self.ALL_MEALS[state.restaurant]
        if len(state.meals) == users:
            return []


# ---------------------------------------------- History ---------------------------------------------------

class Data:
    def __init__(self, restaurants, meals):
        self.history_states = set()
        self.restaurants = restaurants
        self.meals = meals
        self.rest_dict = {rest:i for i, rest in enumerate(restaurants)}

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

    def __init__(self, data, action_obj, init_state, constraints):
        super().__init__(init_state)
        self.data = data
        self.action_obj = action_obj
        self.constraints = constraints

    def result(self, state, action):
        if action[0] == Action.ADD_MEAL:
            return self.change_meal(state, action[1])
        if action[0] == Action.CHANGE_REST:
            return self.change_rest_state(action[1])

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
        rest_index = random.randint(0, len(self.data.restaurants))
        length_meals = len(self.data.meals[rest_index])
        return State(self.data.restaurants[rest_index],
                     [self.data.meals[rest_index][random.randint(0, length_meals)] for j in
                      range(users)])

    def cost(self, state, action, state2):

        return self.value(state2) - self.value(state)

    def actions(self, state):
        return self.action_obj.get_actions(state)

    def heuristic(self, state):
        pass

    def change_rest_state(self, i) -> State:
        return State(self.data.restaurants[i], [])

    def change_meal(self, state, index_meal) -> State:
        state.meals.append(self.data.meals[self.data.rest_dict[state.restaurant]][index_meal])
        return state


# ---------------------------------------------- main  ---------------------------------------------------------------
if __name__ == '__main__':
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
    init_state = State(rest="Doron's Jachnun | Tel Aviv",
                       meals=['ג׳חנון תימני עצבני', 'ג׳חנון תימני עצבני', 'ג׳חנון תימני עצבני'])
    problem = WoltProblem(history, action_obj, init_state, constraints)
    result = breadth_first(problem)
    print(result)
