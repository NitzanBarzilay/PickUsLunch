import requests
import json


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
        self.baseprice = __data__['baseprice']
        self.image = __data__.get("image")
        self.name = __data__['name'][0]['value']



if __name__ == '__main__':
    wolt = Wolt()

    # Get the matching streets
    cities = wolt.get_matching_cities('Allenby')

    # Select the first place (Allenby, Tel-Aviv Yafo)
    # and get the latitude and longitude of it
    city = cities[0]['place_id']
    lat_lon = wolt.get_lat_lon(city)
    print(lat_lon)

    # Pass the latitude and longitude to get all nearby restaurants
    restaurants = wolt.get_nearby_restaurants(lat_lon['lat'], lat_lon['lng'])
    print(restaurants[8])
    name = restaurants[8]['title']
    restaurant = wolt.serach_restaurant(name=name, lat=lat_lon['lat'], lon=lat_lon['lng'])
    print(restaurant)

    oid = restaurant[0]['value']['active_menu']['$oid']

    # find restaurant menu by her oid (wold id) the search return wolt rest obeject
    menu = wolt.get_restaurant_menu(oid)  # the oid can be find at the last function u can find him;
    print(menu)