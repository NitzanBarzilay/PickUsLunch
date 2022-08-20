from simpleai.search import SearchProblem
from typing import List, Tuple
from enum import Enum


class WoltProblem(SearchProblem):

    def result(self, state, action):
        if action[0] == Action.CHANGE_MEAL:
            return change_meal(state, action[1])
        if action[0] == Action.CHANGE_REST:
            return change_rest_state()

    def is_goal(self, state):
        pass

    def value(self, state):
        pass

    def crossover(self, state1, state2):
        pass

    def mutate(self, state):
        pass

    def generate_random_state(self):
        pass

    def cost(self, state, action, state2):
        # TODO: read the right lines from the csv
        pass

    def actions(self, state):
        pass

    def heuristic(self, state):
        pass


class State:
    def __init__(self, rest, meals: List):
        self.restaurant = rest
        self.meals = meals


class Action:
    CHANGE_REST = 0
    CHANGE_MEAL = 1

    def get_actions(self, length_meals):
        actions = []
        for i in range(length_meals):
            actions.append((self.CHANGE_REST, -1))
            actions.append((self.CHANGE_MEAL, i))


class History:
    def __init__(self):
        self.history_states = set()

    def check_state(self, state_tuple: Tuple[str]):
        return state_tuple in self.history_states

    def add_state(self, state_tuple: Tuple[str]):
        self.history_states.add(state_tuple)


def change_rest_state() -> State:
    pass


def change_meal(state, index) -> State:
    pass
