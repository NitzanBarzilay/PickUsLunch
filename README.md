![image](https://user-images.githubusercontent.com/36603609/187044513-49cbe2fe-aaeb-4bd2-aafc-7fa452b285ed.png)
# PickUsLunch - AI smart assistant for group meal orders
text here
## Abilities and limitations
text here
## Algorithms used
text here
## How to use
To use the assistant, clone the repo. Then, create a txt file that represents the constraints and preferences of each diner (example files including instructions are found [here](https://github.com/NitzanBarzilay/PickUsLunch/tree/main/example_preferences)). After creating the preferences file, you can run the PickUsLunch AI assistant using the one of the following commands:
```
python3 main.py <preference_file_path> <output_file_path>
python3 main.py <preference_file_path> <output_file_path> <algorithm>
```
with the following inputs:
1. preference_file_path - path to the preference file you have created / one of the example prefrences files provided
2. output_file_path - path to save results to
3. algorithm (optional) - if not specified, the default algorithm will be ran. If you want, you can choose a specific algorithm from the following list:
   - naive
   - dfs
   - ucs
   - hill_climbing
   - stochastic_hill_climbing
   - simulated_anealing
   - genetic
