import time
import pygad
import random
import numpy as np
import pandas as pd
from copy import deepcopy

database = pd.DataFrame()
res = []
user_input = dict()


def _get_diners_constraints(filename):
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


def get_user_input(diner1, diner2, diner3):
    diners = [diner1, diner2, diner3]

    return dict(kosher=any([diner[0] for diner in diners]),
                vegetarian=[diner[1] for diner in diners],
                GF=[diner[2] for diner in diners],
                alcohol_percentage=[0 for _ in diners],
                spicy=[diner[4] == 1 for diner in diners],
                price=[diner[5] for diner in diners],
                min_rating=max([diner[6] for diner in diners]),
                cuisines=[diner[8] for diner in diners],
                weekday=diner1[9],
                people=len(diners),
                )


def fitness_function(solution, solution_idx):
    solution = [database.iloc[i] for i in solution]

    price_const = [(dish['price'] <= inp) for (dish, inp) in zip(solution, user_input['price'])]

    vegetarian_const = [dish['vegetarian'] == inp for (dish, inp) in zip(solution, user_input['vegetarian'])]
    GF_const = [dish['GF'] == inp for (dish, inp) in zip(solution, user_input['GF'])]
    spicy_const = [dish['spicy'] == inp for (dish, inp) in zip(solution, user_input['spicy'])]
    alcohol_percentage_const = [(dish['alcohol_percentage'] > 0) == inp for (dish, inp) in
                                zip(solution, user_input['alcohol_percentage'])]

    hard_const = all([all(price_const),
                      all(vegetarian_const),
                      all(GF_const),
                      all(spicy_const),
                      all(alcohol_percentage_const)])

    soft_price = np.mean(
        [s * (p['price'] / ip) for (s, (p, ip)) in zip(price_const, zip(solution, user_input["price"]))])

    soft_const = np.mean([soft_price,
                          sum(vegetarian_const) / user_input['people'],
                          sum(GF_const) / user_input['people'],
                          sum(spicy_const) / user_input['people'],
                          sum(alcohol_percentage_const) / user_input['people'],
                          ])
    fitness = np.mean([int(hard_const), soft_const])
    res.append(fitness)

    return fitness


def print_dish(solution, const, i):
    dish = solution[i]

    print('-' * 50)
    print(f'{dish["name"]} - {dish["price"]} ₪')

    print(f'price: {"✅" if dish["price"] <= const["price"][i] else f"❌"}', end=' ')
    print(f'vegetarian: {"✅" if dish["vegetarian"] == const["vegetarian"][i] else f"❌"}', end=' ')
    print(f'GF: {f"✅" if dish["GF"] == const["GF"][i] else f"❌"}', end=' ')
    print(f'spicy: {f"✅" if dish["spicy"] == const["spicy"][i] else f"❌"}', end=' ')
    print(f'alcohol: {f"✅" if (dish["alcohol_percentage"] > 0) == const["alcohol_percentage"][i] else f"❌"}')

    print('-' * 50)

# ---------------------------------------- Algorithm wrapper functions  ------------------------------------------------

def GeneticAlgorithm(_rest_df, meals_df, diner_1, diner_2, diner_3):
    """
    Genetic algorithm wrapper function. returns the solution that the algorithm chose (restaurant and 3 meals) and it's runtime.
    :param rest_df: restaurant dataframe
    :param meals_df: meals dataframe
    :param diner1: list of 1st diner preferences
    :param diner2: list of 2nd diner preferences
    :param diner3: list of 3rd diner preferences
    :return: chosen restaurant dataframe (single row), 3 chosen meals dataframes (single row each), runtime (float).
    """
    global user_input, database, res

    # initialize:
    rest_df = _rest_df.copy().set_index('name')
    groups = [g for g in meals_df.groupby('rest_name') if g[0] in list(rest_df.index)]
    base_user_input = get_user_input(diner_1, diner_2, diner_3)

    best_fitness = -np.inf
    best_solution = np.nan
    best_res = []
    best_rest_input = pd.DataFrame()
    best_database = np.nan

    found = False

    combinations = 0
    for _, df in groups:
        combinations += (df.shape[0] ** 3)

    print(f'------- Searching space of size {combinations} -------\n')

    start = time.time()
    random.shuffle(groups)

    for restaurant, database in groups:
        # print('...')
        rest_input = rest_df.loc[restaurant]
        delivery_per_person = rest_input["delivery price"] / 3
        user_input = deepcopy(base_user_input)
        user_input["price"] = [p - delivery_per_person for p in user_input["price"]]

        if rest_input.kosher != user_input['kosher']:
            continue
        if rest_input.rating < user_input["min_rating"]:
            continue

        database.reset_index(inplace=True, drop=True)
        res = []

        # choose run params:
        ga_instance = pygad.GA(num_generations=100,
                               num_parents_mating=5,
                               sol_per_pop=10,
                               num_genes=user_input['people'],

                               init_range_low=0,
                               init_range_high=len(database),

                               random_mutation_min_val=0,
                               random_mutation_max_val=len(database),

                               mutation_by_replacement=True,
                               mutation_num_genes=1,
                               mutation_probability=0.1,
                               crossover_type='scattered',
                               gene_type=int,
                               fitness_func=fitness_function,
                               )

        # run:
        ga_instance.run()

        # print solution:
        solution, fitness, i = ga_instance.best_solution()

        if fitness > best_fitness:
            best_fitness = fitness
            best_res = res.copy()
            best_rest_input = rest_input
            best_solution = solution
            best_database = database.copy()
            print(f'\nUpdate: best fitness updated to {round(best_fitness, 3)}', end='')
            if (not found) and (best_fitness > 0.5):
                initial = time.time()
                print(f"---> a solution was found ! 🏆 ({round(initial - start, 3)}s)")
                found = True
            else:
                print()

    end = time.time()
    runtime = start - end
    best_solution = [best_database.iloc[i] for i in best_solution]
    if __name__ == '__main__':
        [print(inp) for inp in base_user_input.items()]

        print(f'total time {round(end - start, 3)}s')
        print(f'\n----------------- Best solution ------------------\n')
        print(f'Fitness:          {best_fitness}')
        print(f'Hard constraints: {"🏆" if best_fitness > 0.5 else "🛑"}')
        print(f'Total price:      {sum([dish["price"] for dish in best_solution])} ₪ (limit was {user_input["price"]} ₪)')
        print()
        [print_dish(best_solution, user_input, i) for i in range(len(best_solution))]

        print()
        print('Evolution Plot:')
        pd.Series(best_res).plot()
        pd.Series(best_res).rolling(window=100).mean().plot()
        pd.Series(np.ones(len(best_res)) * 0.5).plot(style='--')

    return best_rest_input, best_solution[0], best_solution[1], best_solution[2], runtime


if __name__ == '__main__':
    # load database:
    meals_df = pd.read_csv('./data/mealsData.csv')
    rest_df = pd.read_csv('./data/restaurantsData.csv')
    diner1, diner2, diner3 = _get_diners_constraints('./example_preferences/input_constraints_4.txt')
    rest, meal1, meal2, meal3, runtime = GeneticAlgorithm(rest_df, meals_df, diner1, diner2, diner3)
