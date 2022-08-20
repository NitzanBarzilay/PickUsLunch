import random
import dataFrameParser
from simpleai.search import SearchProblem
from typing import List, Tuple


# ---------------------------------------------- Helper classes -----------------------------------------------------
class State:
    def __init__(self, rest: str, meals: List):
        self.restaurant = rest
        self.meals = meals


class Action:
    CHANGE_REST = 0
    CHANGE_MEAL = 1

    def __init__(self, history):
        self.history = history

    def get_actions(self):
        actions = []
        for i in range(self.history.restaurants):
            actions.append((self.CHANGE_REST, i))
        for i in range(len(self.history.meals)):
            actions.append((self.CHANGE_MEAL, i))


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
            return self.change_meal(state, action[1])
        if action[0] == Action.CHANGE_REST:
            return self.change_rest_state(action[1], len(self.history.meals[action[1]]))

    def is_goal(self, state):
        pass

    def value(self, state):
        pass

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
        pass

    def actions(self, state):
        return self.action_obj.get_actions()

    def heuristic(self, state):
        pass

    def change_rest_state(self, i, length_meals, users=3) -> State:
        return State(self.history.restaurants[i], [self.history.meals[i][random.randint(0, length_meals)] for j in
                                                   range(users)])

    def change_meal(self, state, index) -> State:
        rest_index = self.history.restaurants.index(state.restaurant)
        state.meals[index] = \
            self.history.meals[rest_index] \
                [random.randint(0, (self.history.meals[rest_index]))]
        return state


# ---------------------------------------------- main  -----------------------------------------------------
if __name__ == '__main__':
    df = dataFrameParser.WoltParser([], init_files=False)
    df.read_df()
    data_rests = df.df
    data_menu = df.df_menus
    restaurants = get_rest_lst(data_rests)
    meals = get_menus_meals(data_menu, restaurants)
    history = History(restaurants, meals)
    action_obj = Action(history)
    print(restaurants)
    print(meals)
