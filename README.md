![image](https://user-images.githubusercontent.com/36603609/187044513-49cbe2fe-aaeb-4bd2-aafc-7fa452b285ed.png)
# PickUsLunch - AI smart assistant for group meal orders
We present PickUsLunch - a smart group meal order AI assistant, that takes into concideration each group member's needs and preferences, and provides you with a recommendation for a single restaurant that matches everybody's needs!
## Abilities and limitations
The assistant will need 10 simple preferences from each diner in the group, and in return it will provide you with a restaurant and list on meals for it's menu that best matches everybody's needs and likes ouf of Wolt's restaurants variaty. 
This is a first demo version (which we plan to further extend in the future), which currently has the following limitations:
* The assistant works only on groups of exactly 3 diners
* The assistant will choose a restaurant that were availably via Wolt _in Tel Aviv_ during august 2022.


For information regarding the algorithms used and their performance, see [the project's review](https://github.com/NitzanBarzilay/PickUsLunch/blob/main/PickUsLunch%20-%20Project%20review%20(Hebrew).pdf) (only available in Hebrew). 

## How to use
To use the assistant, clone the repo and make sure you have the required packages installed on your environment. 
Then, create a txt file that represents the constraints and preferences of each diner (example files including instructions are found [here](https://github.com/NitzanBarzilay/PickUsLunch/tree/main/example_preferences)). After creating the preferences file, you can run the PickUsLunch AI assistant using the one of the following commands:
```
python3 main.py <preference_file_path> <output_file_path>
python3 main.py <preference_file_path> <output_file_path> <algorithm>
```
with the following inputs:
1. preference_file_path - path to the preference file you have created / one of the example prefrences files provided
2. output_file_path - path to save results to
3. algorithm (optional) - if not specified, the default algorithm (hill climbing algorithm) will be ran. If you want, you can choose a specific algorithm from the following list:
   - naive
   - dfs
   - ucs
   - a_star
   - hill_climbing
   - simulated_anealing
   - genetic
