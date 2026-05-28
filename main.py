import logging as logger
from Run import run
import argparse
import json

logger.basicConfig(level=logger.INFO, filename= "main.log")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pathway Script")
    parser.add_argument("-i", "--input", help="Input file", required=True)
    parser.add_argument("-o", "--output_folder", help="Output folder", required=True)
    parser.add_argument("-s", "--scenario", help="Scenario choice", required=True)
    args = parser.parse_args()
    input = args.input
    output_folder = args.output_folder
    scenario_choice = args.scenario

    logger.info(f"Starting run for {input}, scenario: {scenario_choice}, {output_folder} ")

    # CONSTANT ---------------------------------------------------------------------
    technologies = ["DACCS", "BECCS", "OAE", "AR", "EW", "BC"]
    storage_based_technologies = ["DACCS", "BECCS"]
    discount_rate = 0.01
    penalty_cdr = 500000000 #$/MtCO2 = $170 Millions/Mt
    penalty_storage = 500000000 #$/MtCO2 = $170 Millions/Mt

    # Define custom colors for each technology
    technology_colors = {
        'DACCS': 'maroon',   
        'BECCS': 'darkgreen',      
        'OAE': 'cornflowerblue',              
        'BC': 'purple',
        'AR': 'limegreen',         
        'EW': 'slategray'}
    
    # Define the **desired order** of technologies in the stacked plot and legend
    custom_order = ['AR', 'BC' , 'EW','BECCS', 'OAE', 'DACCS']
    
    # ARGS ------------------------------------------------------------------------
    with open("scenarios_high.json", "r") as f:
        scenarios = json.load(f)

    if scenario_choice in scenarios:
        scenario = scenarios[scenario_choice]
    else: 
        raise

    file_path = f"./Input/{input}"

    # DYNAMIC --------------------------------------------------------------------
    foresights =  [5, 10, 15, 20, 25, 30]
    dps = [5, 10, 15, 20, 25, 30] 
    growth_rates = [1.025, 1.05, 1.1]

    for sight in foresights:
        logger.info(f"Optimizing foresight {sight}")
        for dp in dps:
            logger.info(f"Optimizing decision period {dp}")
            for rate in growth_rates:
                logger.info(f"Optimizing rate {rate}")
                if rate == 0.025:
                    output_name = f"I{int(rate*1000-1000)}F{sight}D{dp}"
                else:
                    output_name = f"I{int(rate*100-100)}F{sight}D{dp}"
                run(
                    file_path = file_path,
                    scenario = scenario,
                    technologies = technologies,
                    storage_based_technologies = storage_based_technologies,
                    foresight = sight,
                    decision_period = dp,
                    discount_rate = discount_rate,
                    max_growth_rate = rate,
                    min_decline_rate = 2 - rate,
                    penalty_cdr = penalty_cdr,
                    penalty_storage = penalty_storage,
                    technology_colors = technology_colors,
                    custom_order = custom_order,
                    output = output_name,
                    output_folder = output_folder
                )

# Run with:
# python main.py -i <input_file_name> -s <scenario_input(low,mod,high)> -o <output_folder>