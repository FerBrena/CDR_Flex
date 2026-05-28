# CDR_Flex

## Description
CDR optimization model

1. Clean_data.py: Loads input excel file and cleans it.

2. Functions_params.py: Creates all variables (e.g. costs, demand, capacities, etc.)

3. Scenarios.json: Change parameter level of each scenario (high, medium, low)

4. Run.py: Where the model is built. Pyomo and the solver Gurobi are used.

## How to run
Step 1. pip install -r requirements.txt

Step 2. Make sure you have Gurobi license credentials

Step 3. ```python Run.py -i <input_file_name> -s <scenario_input(low,mod,high)> -o <output_file_name>```