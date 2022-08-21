from main import Restaurant, get_restaurant_list
from typing import List, Tuple
import csv
import pickle
import pandas as pd


class WoltParser:
    def __init__(self, restaurants: List[Restaurant], general_file_name:str = "csv_wolt_restaurants_21-8-22.csv",
                 menu_file_name: str = "csv_wolt_menus_20-8-22.csv", init_files: bool = True):
        self.menus_df = None
        self.general_df = None
        self.restaurants = restaurants
        if init_files:
            self.create_general_df()
        self.file_name = f"data/{general_file_name}"
        self.file_name_menus = f"data/{menu_file_name}"

    def crate_restaurants_df(self):
        headers = ["rest_name", "name", "price", "alcohol_percentage", "vegetarian", "GF", "image", "days", "spicy"]
        with open(self.file_name_menus, "w", newline="", encoding='utf-8') as curr_file:
            dw = csv.DictWriter(curr_file, delimiter=",", fieldnames=headers)
            dw.writeheader()
            for rest in self.restaurants:
                for dish in rest.menu:
                    line_dict = dish._asdict()
                    line_dict["rest_name"] = rest.name
                    dw.writerow(line_dict)

    def create_general_df(self):
        headers = [
            "name", "address", "city", "delivery estimation [minutes]", "delivery price", "food categories",
            "is active", "is valid", "kosher", "location lat long", "menu", "opening days", "prep estimation [minutes]"
            , "rating"
        ]
        with open(self.file_name, "w", newline="", encoding='utf-8') as curr_file:
            dw = csv.DictWriter(curr_file, delimiter=",", fieldnames=headers)
            dw.writeheader()
    @staticmethod
    def write_line(rest, headers=None):
        if headers is None:
            headers = [
                "name", "address", "city", "delivery estimation [minutes]", "delivery price", "food categories",
                "is active", "is valid", "kosher", "location lat long", "menu", "opening days",
                "prep estimation [minutes]"
                , "rating"
            ]
        with open("csv_wolt_restaurants_19-8-22.csv", "w", newline="", encoding='utf-8') as curr_file:
            dw = csv.DictWriter(curr_file, delimiter=",", fieldnames=headers)
            line_dict = {"name": rest.name, "address": rest.address, "city": rest.city,
                         "delivery estimation [minutes]": rest.delivery_estimation,
                         "delivery price": rest.delivery_price,
                         "food categories": "---".join(rest.food_categories),
                         "is active": rest.is_active, "is valid": rest.is_valid, "kosher": rest.kosher,
                         "location lat long": tuple(rest.location), "opening days": " ".join(rest.opening_days),
                         "prep estimation [minutes]": rest.prep_estimation, "rating": rest.rating, "menu": []}
            meal_lst = []
            for meal in rest.menu:
                meal_lst.append(str(meal))
            line_dict["menu"] = "---".join(meal_lst)
            dw.writerow(line_dict)

    def get_dfs(self)-> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.general_df:
            self.general_df = pd.read_csv(self.file_name, encoding="utf-8")
        if not self.menus_df:
            self.menus_df = pd.read_csv(self.file_name_menus, encoding="utf-8")

        return self.general_df, self.menus_df


if __name__ == '__main__':
    load = False
    restaurants = get_restaurant_list(10)

    # if not load:
    #     with open('rests.pickle', 'wb') as handle:
    #
    #         pickle.dump(restaurants, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # else:
    #     with open('rests.pickle', 'rb') as handle:
    #         restaurants = pickle.load(handle)
    file_creator = WoltParser(restaurants, True)
    file_creator.get_dfs()
    print(file_creator.general_df)
