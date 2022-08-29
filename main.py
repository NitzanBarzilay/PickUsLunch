import random
import requests
import json
import pandas as pd
from collections import namedtuple
from time import sleep
import tqdm
import sys
from dataFrameParser import WoltParser
from gainFunction import user_inputs_to_gain_function_inputs, gain
from naiveAlgorithm import NaiveAlgorithm
from geneticAlgorithm import GeneticAlgorithm
from localSearchAlgorithms import DFSAlgorithm, UCSAlgorithm, AstarAlgorithm, HillClimbingAlgorithm, \
    StochasticHillClimbingAlgorithm, SimulatedAnnealingAlgorithm

INPUT_FILE = 1
OUTPUT_FILE = 2
ALGORITHM = 3

class Wolt:
    HEADERS = {
        'user-agent':
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.68 Safari/537.36'
    }

    PLACES_URL = 'https://restaurant-api.wolt.com/v1/google/places/autocomplete/json'
    GOOGLE_GEOCODE_URL = 'https://restaurant-api.wolt.com/v1/google/geocode/json'
    DELIVERY_URL = 'https://restaurant-api.wolt.com/v1/pages/delivery'

    def __init__(self):
        self.__wolt_api_url = "https://restaurant-api.wolt.com"

    def get_matching_cities(self, street):
        """
        :param street: a street name
        :return: a list of dictionaries containing the all the cities that has the passed street name and its id
        """
        params = {'input': street}
        response = json.loads(requests.get(self.PLACES_URL, headers=self.HEADERS, params=params).text)
        predictions = response['predictions']

        return [
            {
                'street': result['description'],
                'place_id': result['place_id']} for result in predictions] if response else None

    def get_lat_lon(self, city_id):
        """
        :param city_id: the ID of the city you want to get the latitude and longitude of
        :return: dictionary containing the lat and lon of the
        """
        params = {'place_id': city_id}
        response = json.loads(requests.get(self.GOOGLE_GEOCODE_URL, headers=self.HEADERS, params=params).text)
        lat_lon = response['results'][0]['geometry']['location']
        return lat_lon

    def serach_restaurant(self, name='', lat=None, lon=None, city=None, limit=50):
        if lat and lon:
            __request__ = requests.get(f"{self.__wolt_api_url}/v1/search?q={name}&lat={lat}&lon={lon}&limit={limit}")
        else:
            __request__ = requests.get(f"{self.__wolt_api_url}/v1/search?q={name}&limit={limit}")
        if __request__.status_code == 200:
            return __request__.json().get("results")

        return None

    def get_nearby_restaurants(self, lat, lon):
        """
        :param lat: latitude of the street
        :param lon: longitude of the street
        :return: list of dictionaries containing all the available information about nearby restaurant
        """
        params = {'lat': lat, 'lon': lon}
        response = requests.get(self.DELIVERY_URL, headers=self.HEADERS, params=params).text
        restaurants = json.loads(response)['sections'][0]['items']
        return restaurants

    def get_restaurant_menu(self, oid):
        __request__ = requests.get(f"{self.__wolt_api_url}/v3/menus/{oid}")
        if __request__.status_code == 200:
            return Wolt_Resterant(__request__.json().get("results").pop())


class Wolt_Resterant:
    def __init__(self, __data__):
        self.oid = __data__['_id']['$oid']
        self.categories = [Wolt_Categorie(x) for x in __data__['categories']]
        self.meals = [Wolt_Meals(x) for x in __data__['items']]


class Wolt_Categorie:
    def __init__(self, __data__):
        self.o_id = __data__["_id"]['$oid']
        self.name = __data__['name'][0]['value']
        self.description = __data__['description']


class Wolt_Meals:
    def __init__(self, __data__):
        self.o_id = __data__['_id']['$oid']
        self.alcohol_percentage = __data__['alcohol_percentage']
        self.allowed_delivery_methods = __data__['allowed_delivery_methods']
        self.price = __data__['baseprice']
        self.name = __data__['name'][0]['value']
        try:
            self.image = __data__['image']
        except KeyError:
            self.image = None
        self.days = __data__['times'][0]['visible_days_of_week']


def create_restaurant_df(rest_dict: dict) -> pd.DataFrame:
    """
    :param rest_dict: a dictionary containing all the information about the restaurant
    :return: a pandas dataframe containing all the information about the restaurant
    """
    df = pd.DataFrame(rest_dict)
    df.set_index('oid', inplace=True)
    return df


class Restaurant:
    '''
    A class that represents a parsed restaurant object
    '''
    MEAL_MIN_PRICE = 30

    def __init__(self, name: str, wolt: Wolt, lat_lon: dict):
        self.is_valid = True
        restaurant = wolt.serach_restaurant(name=name, lat=lat_lon['lat'], lon=lat_lon['lng'])
        if not restaurant or restaurant[0]['value']['product_line'] != 'restaurant' \
                or 'homedelivery' not in restaurant[0]['value']['delivery_methods']:
            self.is_valid = False
            return
        self.name = name
        restaurant = restaurant[0]['value']

        # general info about the restaurant
        self.is_active = restaurant['online']
        self.id = restaurant['id']['$oid']
        self.location = restaurant['location']['coordinates']
        self.address = restaurant['address']
        self.city = restaurant['city']
        try:
            self.rating = restaurant['rating']['score']
        except KeyError:
            self.rating = None

        # food
        self.menu = []
        self.__fill_restaurant_menu(wolt, restaurant)
        self.food_categories = restaurant['food_tags']
        self.kosher = ("kosher" in self.food_categories) or ('Kosher L’mehadrin' in self.food_categories)

        # delivery
        self.delivery_estimation = restaurant['estimates']['delivery']['mean']
        self.prep_estimation = restaurant['estimates']['preparation']['mean']
        self.delivery_price = restaurant['delivery_specs']['delivery_pricing']['base_price'] / 100

        # opening hours
        self.opening_days = list(restaurant['opening_times'].keys())

    def __fill_restaurant_menu(self, wolt: Wolt, restaurant: dict) -> None:
        """
        Fills the menu of the restaurant with parsed meal namedtuples that includes
        info like if the meal is vegetarian, spicy or gluten free.
        Only keeps meals that are available for delivery and cost more than 20 ILS.
        :param wolt: Wolt object
        :param restaurant: restaurant dictionary
        :return: None
        """
        Meal = namedtuple("Meal", "name price alcohol_percentage vegetarian GF spicy image days")
        temp_menu = wolt.get_restaurant_menu(restaurant['active_menu']['$oid']).meals
        for item in temp_menu:
            if "homedelivery" in item.allowed_delivery_methods and (item.price / 100) > self.MEAL_MIN_PRICE:
                is_veg = "vegan" in item.name or "vegetarian" in item.name or "וג'י" in item.name \
                         or 'טבעוני' in item.name or 'צמחוני' in item.name or "🌱" in item.name
                is_spicy = "spicy" in item.name or 'חריף' in item.name or 'חריפה' in item.name \
                           or 'ספייסי' in item.name or '🌶' in item.name
                gluten_free = "ללא גלוטן" in item.name or "נטול גלוטן" in item.name or "GF" in item.name \
                              or "🌾" in item.name
                self.menu.append(Meal(item.name, item.price / 100, item.alcohol_percentage / 10.0,
                                      is_veg, gluten_free, is_spicy, item.image, item.days))


def get_restaurant_list(number_of_rests=None, file_parser: WoltParser = None, save_to_file: bool = False):
    wolt = Wolt()

    # Get the matching streets
    cities = wolt.get_matching_cities('Tel Aviv')

    # Select the first place (Allenby, Tel-Aviv Yafo) and get the lat and long of it
    city = cities[0]['place_id']
    lat_lon = wolt.get_lat_lon(city)
    restaurants = []
    potential_restaurants = wolt.get_nearby_restaurants(lat_lon['lat'], lat_lon['lng'])
    print(len(potential_restaurants))
    if number_of_rests:
        potential_restaurants = potential_restaurants[:number_of_rests]
    for restaurant in tqdm.tqdm(potential_restaurants):
        rest_obj = Restaurant(restaurant['title'], wolt, lat_lon)
        sleep(random.uniform(0, 0.1))
        if rest_obj.is_valid:
            if file_parser:
                try:
                    file_parser.write_line(rest_obj)
                    file_parser.write_line_menu(rest_obj)
                except Exception as e:
                    print(f"got {e} while loading {rest_obj.name} restaurant")
            restaurants.append(rest_obj)
    return restaurants


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


def get_restaurant_results(K, O, R, D, C):
    """
    :return: a string that represents which constraints of the diners were matched by the restaurant.
    """
    results_output = f'    open: {"✅" if O == 1 else f"❌"}\n'
    results_output += f'    kosher: {"✅" if K == 1 else f"❌"}\n'
    results_output += f'    rating: {"✅" if R == 1 else f"❌"}\n'
    results_output += f'    delivery matches hunger level: {"✅" if D == 1 else f"❌"}\n'
    results_output += f'    cuisines: {"✅" if C == 1 else f"❌"}\n\n'
    return results_output


def get_meals_result(V, G, A, S, PH, PS) -> str:
    """
    :return: a string that represents which constraints of the diner were matched by the meal.
    """
    result = f'    price: {"✅" if PH == 1 else f"❌"}, {PS} ILS cheaper than max meal price\n'
    result += f'    vegetarian: {"✅" if V == 1 else f"❌"}\n'
    result += f'    gluten: {"✅" if G == 1 else f"❌"}\n'
    result += f'    alcohol: {"✅" if A == 1 else f"❌"}\n'
    result += f'    spicy: {"✅" if S == 1 else f"❌"}\n\n'
    return result


def save_results(results, filename, diner1, diner2, diner3, algo_name):
    """
    prints formatted results and saves them to the given filename.
    :param results: a list of 4 dataframes (one for the restaurant and one for each of
    the 3 diners) and a float representing the running time
    :param filename: output filename
    :param diner1: the diner1 constraints
    :param diner2: the diner2 constraints
    :param diner3: the diner3 constraints
    :param algo_name: the name of the algorithm that was used to find the results
    :return: None
    """
    rest, meal1, meal2, meal3, runtime = results
    gain_params = user_inputs_to_gain_function_inputs(diner1, diner2, diner3, rest, meal1, meal2, meal3)
    O, M, K, DT, D, RD, R, C, V1, V2, V3, G1, G2, G3, A1, A2, A3, S1, S2, S3, PH1, PH2, PH3, PS1, PS2, PS3 = gain_params
    results = f'{algo_name} algorithm\n'
    results += "----------------- CHOSEN SOLUTION -----------------\n"
    results += f"restaurant: {rest.iloc[0]['name']}\n"
    results += get_restaurant_results(K, O, R, D, C)
    results += f"Meal for 1st diner: {meal1['name'].values[0]}\n"
    results += get_meals_result(V1, G1, A1, S1, PH1, PS1)
    results += f"Meal for 2nd diner: {meal2['name'].values[0]}\n"
    results += get_meals_result(V2, G2, A2, S2, PH2, PS2)
    results += f"Meal for 3rd diner: {meal3['name'].values[0]}\n"
    results += get_meals_result(V3, G3, A3, S3, PH3, PS3)
    results += "\n----------------- RESULTS -----------------\n"
    results += f'Gain score: {gain(*gain_params)}\n'
    results += f'Total price: {sum([meal["price"].values[0] for meal in [meal1, meal2, meal3]])}\n'
    # TODO compare percentiles
    results += f"Runtime: {runtime}\n"
    with open(filename, 'w', encoding="utf-8") as f:
        f.write(str(results))
    print(f"DONE! Results saved to {filename}. Showing results:")
    print(results)


def choose_algorithm(algorithm: str):
    """
    Returns an algorithm fucntion based on the algorithm name.
    :param algorithm: algorithm name
    :return: algorithm function
    """
    if algorithm == "naive":
        return NaiveAlgorithm
    elif algorithm == "dfs":
        return DFSAlgorithm
    elif algorithm == "ucs":
        return UCSAlgorithm
    elif algorithm == "astar":
        return AstarAlgorithm
    elif algorithm == "hill_climbing":
        return HillClimbingAlgorithm
    elif algorithm == "stochastic_hill_climbing":
        return StochasticHillClimbingAlgorithm
    elif algorithm == "simulated_annealing":
        return SimulatedAnnealingAlgorithm
    elif algorithm == "genetic":
        return GeneticAlgorithm
    else:
        raise ValueError(f"Algorithm {algorithm} not recognized.")


if __name__ == '__main__':
    if len(sys.argv) not in [3, 4]:
        print("Usage: python3 main.py <preference_file_path> <output_file_path> <algorithm> (algorithm optional)")
        exit(1)
    diner1, diner2, diner3 = get_diners_constraints(sys.argv[INPUT_FILE])
    rest_df = pd.read_csv("data/restaurantsData.csv")
    meals_df = pd.read_csv("data/mealsData.csv")
    if len(sys.argv) == 4:  # specified algorithm
        chosen_algorithm = sys.argv[ALGORITHM]
    else:  # choose default algorithm
        chosen_algorithm = "genetic"  # TODO decide on default algorithm
    algorithm = choose_algorithm(chosen_algorithm)
    results = algorithm(rest_df, meals_df, diner1, diner2, diner3)
    save_results(results, sys.argv[OUTPUT_FILE], diner1, diner2, diner3, chosen_algorithm)

    # df_manager = WoltParser([])
    # restaurants = get_restaurant_list(file_parser=df_manager)
    # df_manager.crate_restaurants_df()
