import requests
import json
import pandas as pd
from collections import namedtuple


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

def create_restaurant_df(rest_dict: dict) -> pd.DataFrame:
    """
    :param rest_dict: a dictionary containing all the information about the restaurant
    :return: a pandas dataframe containing all the information about the restaurant
    """
    df = pd.DataFrame(rest_dict)
    df.set_index('oid', inplace=True)
    return df

class Restaurant:
    def __init__(self, name: str, wolt: Wolt, lat_lon: dict):
        self.is_valid = True
        restaurant = wolt.serach_restaurant(name=name, lat=lat_lon['lat'], lon=lat_lon['lng'])
        if not restaurant:
            self.is_valid = False
            return
        self.name = name
        restaurant = restaurant[0]['value']
        # general info about the restaurant
        self.is_active = restaurant['alive']
        self.id = restaurant['id']['$oid']
        self.location = restaurant['location']['coordinates']
        self.address = restaurant['address']
        self.city = restaurant['city']
        self.rating = restaurant['rating']['score']

        # food
        self.menu = []
        MenuItem = namedtuple("MenuItem", "name price alcohol_percentage")
        temp_menu = wolt.get_restaurant_menu(restaurant['active_menu']['$oid']).meals
        for item in temp_menu:
            if "homedelivery" in item.allowed_delivery_methods:
                self.menu.append(MenuItem(item.name, item.price, item.alcohol_percentage))
        self.food_categories = restaurant['food_tags']

        # delivery
        self.delivery_methods = restaurant['delivery_methods']
        self.delivery_estimation = restaurant['estimates']['delivery']['mean']
        self.prep_estimation = restaurant['estimates']['preparation']['mean']
        self.delivery_price = restaurant['delivery_specs']['delivery_pricing']['base_price']

        # opening hours
        self.open_sunday = restaurant['opening_times']['sunday']
        self.open_monday = restaurant['opening_times']['monday']
        self.open_tuesday = restaurant['opening_times']['tuesday']
        self.open_wednesday = restaurant['opening_times']['wednesday']
        self.open_thursday = restaurant['opening_times']['thursday']
        self.open_friday = restaurant['opening_times']['friday']
        self.open_saturday = restaurant['opening_times']['saturday']

def get_restaurant_list():
    wolt = Wolt()

    # Get the matching streets
    cities = wolt.get_matching_cities('Allenby')

    # Select the first place (Allenby, Tel-Aviv Yafo) and get the lat and long of it
    city = cities[0]['place_id']
    lat_lon = wolt.get_lat_lon(city)
    restaurants = []
    for restaurant in wolt.get_nearby_restaurants(lat_lon['lat'], lat_lon['lng']):
        rest_obj = Restaurant(restaurant['title'], wolt, lat_lon)
        if rest_obj.is_valid and rest_obj.is_active:
            restaurants.append(Restaurant(restaurant['title'], wolt, lat_lon))
    return restaurants


if __name__ == '__main__':
    restaurants = get_restaurant_list()
    print(restaurants[0].name)