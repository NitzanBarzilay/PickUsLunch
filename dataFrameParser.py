from main2 import Restaurant, get_restaurant_list
from typing import List
import csv
import pickle


class WoltParser:
    def __init__(self, restaurants: List[Restaurant]):
        self.restaurants = restaurants
        self.create_general_df()

    def create_general_df(self):
        headers = [
            "name", "address", "city", "delivery estimation [minutes]", "delivery price", "food categories",
            "is active", "is valid", "kosher", "location lat long", "menu", "opening days", "prep estimation [minutes]"
            , "rating"
        ]
        with open("csv_wolt_19-8-22.csv", "w", newline="", encoding='utf-8') as curr_file:
            dw = csv.DictWriter(curr_file, delimiter=",", fieldnames=headers)
            dw.writeheader()
            for rest in self.restaurants:
                line_dict = {"name": rest.name, "address": rest.address, "city": rest.city,
                             "delivery estimation [minutes]": rest.delivery_estimation,
                             "delivery price": rest.delivery_price, "food categories": " ".join(rest.food_categories),
                             "is active": rest.is_active, "is valid": rest.is_valid, "kosher": rest.kosher,
                             "location lat long": tuple(rest.location), "opening days": " ".join(rest.opening_days),
                             "prep estimation [minutes]": rest.prep_estimation, "rating": rest.rating, "menu": []}
                meal_lst = []
                for meal in rest.menu:
                    meal_lst.append(str(meal))
                line_dict["menu"] = " ".join(meal_lst)
                dw.writerow(line_dict)


if __name__ == '__main__':
    restaurants = get_restaurant_list(10)
    # with open('rests.pickle', 'wb') as handle:
    #     pickle.dump(restaurants, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('rests.pickle', 'rb') as handle:
    #     restaurants = pickle.load(handle)
    file_creator = WoltParser(restaurants)
