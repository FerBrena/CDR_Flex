import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from clean_data import process_scenario_data
from functions_params import make_functions, Params
import argparse
import numpy as np
from pyomo.core.expr.visitor import expression_to_string

def extrapolate_missing_years(solved_values, technologies, missing_years, df_limits, max_growth, max_decline):

    extrapolated = {tech: {} for tech in technologies}

    for tech in technologies:
        years_known = sorted(solved_values[tech])

        if len(years_known) < 2:
            continue

        y1, y2 = years_known[-2], years_known[-1]
        v1, v2 = solved_values[tech][y1], solved_values[tech][y2]

        slope = (v2 - v1) / (y2 - y1)

        first_year = df_limits["Year"].min()

        first_limit_row = df_limits.loc[(df_limits["Year"] == first_year) & (df_limits["Technology"] == tech)]

        if not first_limit_row.empty:
            first_year_limit = first_limit_row["Limit"].iloc[0]
            min_threshold = 0.01 * first_year_limit
        else:
            min_threshold = 0

        for y in missing_years:
            if y > y2:
                projected = v2 + slope * (y - y2)
                prev_val = extrapolated[tech].get(y-1, v2)

                lower = prev_val * max_decline
                mult_upper = prev_val * max_growth

                activated = (v1 == 0 and v2 > 0)
                if activated:
                    linear_increment = v2 - v1
                    linear_upper = prev_val + linear_increment
                    upper = min(linear_upper, mult_upper)
                else:
                    upper = mult_upper

                value = min(max(projected, lower), upper)

                limit_row = df_limits.loc[(df_limits["Year"] == y) & (df_limits["Technology"] == tech)]
                if not limit_row.empty:
                    tech_limit = limit_row["Limit"].iloc[0]
                    value = min(max(value, 0), tech_limit)

                if value < min_threshold and slope <= 0:
                    value = 0.0

                extrapolated[tech][y] = value

    return extrapolated

def last_known_unit_cost(detailed_costs, tech, year):
    years = [
        d["Year"] for d in detailed_costs
        if d["Technology"] == tech and d["Year"] <= year
    ]
    if not years:
        return 0
    last_year = max(years)
    for d in reversed(detailed_costs):
        if d["Technology"] == tech and d["Year"] == last_year:
            deployment = d["Optimized Deployment (MtCO₂)"]
            if deployment > 0:
                return d["Total Cost ($M)"] * 1e6 / deployment
    return 0

def extrapolate_penalty(df, penalty_cdr, penalty_storage, storage_techs):
    # Demand penalty: total deployment < demand
    year_stats = (df.sort_values("Year").groupby("Year", as_index=True)
        .agg(
            Demand=("Demand", "first"),                              # DO NOT sum demand
            Deployment=("Optimized Deployment (MtCO₂)", "sum"),      # sum deployment
            Storage=("Storage Capacity", "first")))                  # DO NOT sum storage capacity

    year_stats["Demand Penalty"] = np.where(
        year_stats["Deployment"] < year_stats["Demand"],
        (year_stats["Demand"] - year_stats["Deployment"]) * penalty_cdr/1e6, 0.0)
    
    # Storage penalty: total storage < deployment
    storage_deployment = (df[df["Technology"].isin(storage_techs)].groupby("Year")["Optimized Deployment (MtCO₂)"].sum())

    year_stats["Storage Penalty"] = np.where(
        storage_deployment.reindex(year_stats.index, fill_value=0) > year_stats["Storage"],
        (storage_deployment.reindex(year_stats.index, fill_value=0) - year_stats["Storage"]) * penalty_storage / 1e6, 0.0)

    tol = 1e-6  # adjust as needed
    year_stats["Demand Penalty"] = np.where(year_stats["Demand Penalty"].abs() < tol,0.0,year_stats["Demand Penalty"])
    year_stats["Storage Penalty"] = np.where(year_stats["Storage Penalty"].abs() < tol, 0.0, year_stats["Storage Penalty"])

    idx = (df[df["Source"] == "Extrapolated"].groupby("Year").head(1).index)

    df.loc[idx, "Penalty Demand"] = year_stats.loc[df.loc[idx, "Year"], "Demand Penalty"].values
    df.loc[idx, "Penalty Storage"] = year_stats.loc[df.loc[idx, "Year"], "Storage Penalty"].values
    return df

final_year = 2100

def run(*, file_path, 
        scenario, 
        technologies, 
        storage_based_technologies,
        foresight,
        decision_period,
        discount_rate,
        max_growth_rate,
        min_decline_rate,
        penalty_cdr,
        penalty_storage,
        technology_colors,
        custom_order,output, 
        output_folder):

    df = process_scenario_data(file_path, scenario)
    df.to_excel(f"./Output/{output_folder}/{output}_Processed_CDR.xlsx", index=False)

    df_limits = pd.read_excel(f"./Output/{output_folder}/{output}_Processed_CDR.xlsx")
    limit_cols = [c for c in df_limits.columns if c.startswith("Limit_")]
    df_long = df_limits.melt(id_vars= "Year", value_vars=limit_cols, var_name="Technology", value_name="Limit")
    df_long["Technology"] = df_long["Technology"].str.replace("Limit_", "", regex=False)

    years = sorted(df["Year"].tolist())
    
    # === Initialize Results Storage ===
    results = []
    #previous_decision = {i: {t: 0 for t in years} for i in technologies}  # Set default to 0 instead of 1e-3
    previous_decision = {i: {} for i in technologies}

    solver = pyo.SolverFactory("gurobi", solver_io='python') 

    start_years = list(range(min(years), max(years) + 1, decision_period))
    infeasible_periods = []
    detailed_costs = []

    for start_year in start_years:
        # Ensure the last window covers all remaining years
        end_year = min(start_year + foresight - 1, max(years))
        if end_year == 2099:
            end_year = 2100
        if end_year != start_year:
            print(f"\n🔵 Optimizing for {start_year} - {end_year} 🔵")

            # === Params object ===
            params = Params(dataframe=df, 
                            years=years,
                            foresight=foresight,
                            decision_period=decision_period,  
                            discount_rate=discount_rate,
                            max_growth_rate=max_growth_rate,
                            min_decline_rate=min_decline_rate,
                            penalty_cdr=penalty_cdr,
                            penalty_storage=penalty_storage,
                            technologies=technologies,
                            storage_based_technologies=storage_based_technologies,
                            previous_decision=previous_decision)
            
            # ✅ Create a New Model for Each Planning Window ===
            model = pyo.ConcreteModel()
            model.I = pyo.Set(initialize=technologies)
            model.T = pyo.Set(initialize=range(start_year, end_year + 1))

            # ✅ Define Decision Variables ===
            model.x = pyo.Var(model.I, model.T, domain=pyo.NonNegativeReals)  # CDR deployment
            model.excess_storage = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # Excess storage usage
            model.unmet_cdr = pyo.Var(model.T, domain=pyo.NonNegativeReals) #Unmet CDR demand

            # ✅ Binary activation variables for constraints
            model.growth_active = pyo.Var(model.I, model.T, domain=pyo.Binary)
            model.is_deployed = pyo.Var(model.I, model.T, domain=pyo.Binary)

            # ✅ Objective Function ===
            fns = make_functions(params)
            model.obj = pyo.Objective(rule=fns["total_cost"], sense=pyo.minimize)

            iis_mapping = {}
            gurobi_mapping = {}
            # ✅ Constraint 1: Growth Constraints (Declared Before Solving)
            model.growth_activation_lower = pyo.ConstraintList()
            iis_mapping['growth_activation_lower'] = {}
            counter = 1
            for i in model.I:
                for t in model.T:
                    expr = fns["growth_activation_lower"](model, i, t)
                    if expr is not pyo.Constraint.Skip:
                        model.growth_activation_lower.add(expr = expr)
                        iis_mapping['growth_activation_lower'][counter] = (i, t)
                        gurobi_mapping[counter] = ("growth_activation_lower", (i, t))
                        counter += 1

            model.growth_activation_upper = pyo.ConstraintList()
            iis_mapping['growth_activation_upper'] = {}
            counter = 1
            for i in model.I:
                for t in model.T:
                    expr = fns["growth_activation_upper"](model, i, t)
                    if expr is not pyo.Constraint.Skip:
                        model.growth_activation_upper.add(expr = expr)
                        iis_mapping['growth_activation_upper'][counter] = (i, t)
                        gurobi_mapping[counter] = ("growth_activation_upper", (i, t))
                        counter += 1

            model.enforce_growth = pyo.ConstraintList()
            iis_mapping['enforce_growth'] = {}
            counter = 1            
            for i in model.I:
                for t in model.T:
                    expr = fns["enforce_growth"](model, i, t)
                    if expr is not pyo.Constraint.Skip:
                        model.enforce_growth.add(expr = expr)
                        iis_mapping['enforce_growth'][counter] = (i, t)
                        gurobi_mapping[counter] = ("enforce_growth", (i, t))
                        counter += 1

            # ✅ Constraint 4: Decline Constraint
            model.enforce_decline = pyo.ConstraintList()
            iis_mapping['enforce_decline'] = {}
            counter = 1            
            for i in model.I:
                for t in model.T:
                    expr = fns["enforce_decline"](model, i, t)
                    if expr is not pyo.Constraint.Skip:
                        model.enforce_decline.add(expr = expr)
                        iis_mapping['enforce_decline'][counter] = (i, t)
                        gurobi_mapping[counter] = ("enforce_decline", (i, t))
                        counter += 1                        

            # ✅  Constraint 5: Avoiding excess storage
            model.excess_storage_constraint = pyo.ConstraintList()
            iis_mapping['excess_storage_constraint'] = {}  
            counter = 1   
            for t in model.T:
                expr = fns["excess_storage_constraint"](model, t)
                if expr is not pyo.Constraint.Skip:
                    model.excess_storage_constraint.add(expr = expr)
                    iis_mapping['excess_storage_constraint'][counter] = t
                    gurobi_mapping[counter] = ("excess_storage_constraint", t)
                    counter += 1            


            # ✅  Constraint 6: Meeting CDR Demand
            model.cdr_demand_constraint = pyo.ConstraintList()
            iis_mapping['cdr_demand_constraint'] = {}  
            counter = 1   
            for t in model.T:
                expr = fns["cdr_demand_constraint"](model, t)
                if expr is not pyo.Constraint.Skip:
                    model.cdr_demand_constraint.add(expr = expr)
                    iis_mapping['cdr_demand_constraint'][counter] = t
                    gurobi_mapping[counter] = ("cdr_demand_constraint", t)
                    counter += 1
                    
            # ✅ Constraint 7: Technology Limit Constraint 
            model.tech_limit_constraint = pyo.ConstraintList()
            iis_mapping['tech_limit_constraint'] = {}  
            counter = 1               
            for i in model.I:
                for t in model.T:
                    expr = fns["tech_limit_constraint"](model, i, t)
                    if expr is not pyo.Constraint.Skip:
                        model.tech_limit_constraint.add(expr = expr)            
                        iis_mapping['tech_limit_constraint'][counter] = (i, t)
                        gurobi_mapping[counter] = ("tech_limit_constraint", (i, t))
                        counter += 1

            model.min_deploy_lower = pyo.ConstraintList()
            iis_mapping['min_deploy_lower'] = {}  
            counter = 1               
            for i in model.I:
                for t in model.T:
                    expr = fns["min_deploy_lower"](model, i, t)
                    if expr is not pyo.Constraint.Skip:
                        model.min_deploy_lower.add(expr = expr)            
                        iis_mapping['min_deploy_lower'][counter] = (i, t)
                        gurobi_mapping[counter] = ("min_deploy_lower", (i, t))
                        counter += 1

            model.min_deploy_upper = pyo.ConstraintList()
            iis_mapping['min_deploy_upper'] = {}  
            counter = 1               
            for i in model.I:
                for t in model.T:
                    expr = fns["min_deploy_upper"](model, i, t)
                    if expr is not pyo.Constraint.Skip:
                        model.min_deploy_upper.add(expr = expr)            
                        iis_mapping['min_deploy_upper'][counter] = (i, t)
                        gurobi_mapping[counter] = ("min_deploy_upper", (i, t))
                        counter += 1

            model.deploy_growth = pyo.ConstraintList()
            iis_mapping['deploy_growth'] = {}  
            counter = 1               
            for i in model.I:
                for t in model.T:
                    expr = fns["deploy_growth"](model, i, t)
                    if expr is not pyo.Constraint.Skip:
                        model.deploy_growth.add(expr = expr)            
                        iis_mapping['deploy_growth'][counter] = (i, t)
                        gurobi_mapping[counter] = ("deploy_growth", (i, t))
                        counter += 1

            # === Solve Optimization ===

            solver.options['IterationLimit'] = 20000   # Allow more iterations
            solver.options['OptimalityTol'] = 1e-8       # Require very high precision

            result = solver.solve(model)

            if result.solver.termination_condition == pyo.TerminationCondition.infeasible:
                print(f"❌ Infeasibility detected in {start_year} - {end_year}. Skipping this period.")

                gurobi_model = solver._solver_model  # This is the low-level Gurobi model
                gurobi_model.computeIIS()

                gurobi_to_pyomo = {}

                for pyomo_con, gurobi_con in solver._pyomo_con_to_solver_con_map.items():
                    gurobi_to_pyomo[gurobi_con] = pyomo_con

                with open(f"./Output/{output_folder}/{output}_IIS_CDR.txt", "a") as f:
                    f.write(f"\n=== Window {start_year}-{end_year} ===\n")
                    for c in gurobi_model.getConstrs():
                        if c.IISConstr and c in gurobi_to_pyomo:
                            pcon = gurobi_to_pyomo[c]
                            expr = expression_to_string(pcon.body)
                            lb = pcon.lower
                            ub = pcon.upper

                            if lb is not None and ub is not None and lb == ub:
                                eq = f"{expr} == {lb}"
                            elif ub is not None:
                                eq = f"{expr} <= {ub}"
                            else:
                                eq = f"{expr} >= {lb}"

                            f.write(f"{pcon.name}:\n  {eq}\n\n")

                """If cdr_demand_constraint in IIS:
                    for t in infeasible_years:
                        x(t-1)*max_growth_rate"""

                infeasible_periods.append((start_year, end_year))
                continue  # Skip storing results for this period
                
            # === Store Results ===
            seen_entries = set()

            penalty_demand = []
            penalty_estorage = []

            if decision_period > foresight:
                commit_year = end_year + 1
            else:
                if start_year + decision_period >= 2100:
                    commit_year = start_year + decision_period + 1
                else:
                    commit_year = start_year + decision_period

            for i in technologies:
                for t in model.T:
                    optimized_value = max(0, pyo.value(model.x[i, t]))

                    # ✅ Store only committed values (used for growth/decline continuity)
                    if i not in previous_decision:
                        previous_decision[i] = {}
                    if start_year <= t < commit_year:
                        previous_decision[i][t] = optimized_value

                    # ✅ Save only the years within the current decision period

                    if start_year <= t < commit_year:

                        if (t, i) not in seen_entries:
                            results.append({
                                "Year": t,
                                "Technology": i,
                                "Optimized Deployment": optimized_value,
                                "Source": "Optimized"
                            })
                            seen_entries.add((t, i))
            

                        # ✅ ✅ ✅ Add this block here (INSIDE the loop) 
                        penalty_demand_cost = (params.penalty_cdr * pyo.value(model.unmet_cdr[t]))/ 1e6 # $/MtCO2 * MtCO2/MtCO2 = $/MtCO2
                        penalty_storage_cost = (params.penalty_storage * pyo.value(model.excess_storage[t]))/ 1e6 # $/MtCO2 * MtCO2/MtCO2 = $/MtCO2
                        penalty_cost = penalty_demand_cost + penalty_storage_cost
                        total_cost = (params.cost.get((i, t),0) * optimized_value/ 1e6)  # $/MtCO₂ × MtCO₂ (1/MtCO2) = $/MtCO2

                        # save result
                        detailed_costs.append({
                            "Year": t,
                            "Technology": i,
                            "Demand": params.cdr_demand[t],
                            "Storage Capacity": params.storage_capacity[t],
                            "Limits": params.limits[i, t],
                            "Optimized Deployment (MtCO₂)": optimized_value,
                            "Total Cost ($M)": total_cost,
                            "Penalty Demand": penalty_demand_cost if t not in penalty_demand else 0,
                            "Penalty Storage": penalty_storage_cost if t not in penalty_estorage else 0,
                            "Source": "Optimized"
                        })

                        if t not in penalty_demand:
                            penalty_demand.append(t)
                        if t not in penalty_estorage:
                            penalty_estorage.append(t)

            # === EXTRAPOLATE MISSING YEARS HERE ===

            solved_years = sorted(previous_decision[next(iter(technologies))].keys())

            last_known_year = solved_years[-1]
            next_start = start_year + decision_period
            if next_start == 2100:
                next_start = 2101

            gap_years = list(range(last_known_year + 1, min(next_start, final_year + 1)))
            print(gap_years)
            if gap_years:
                extrapolated = extrapolate_missing_years(
                    solved_values=previous_decision,
                    technologies=technologies,
                    missing_years=gap_years, df_limits=df_long,
                    max_growth= max_growth_rate, max_decline= min_decline_rate
                )

                for i in technologies:
                    for y in gap_years:
                        previous_decision[i][y] = extrapolated[i][y]
                for i in technologies:
                    for y in gap_years:
                        results.append({
                            "Year": y,
                            "Technology": i,
                            "Optimized Deployment": extrapolated[i][y],
                            "Source": "Extrapolated"
                        })
            
            for i in technologies:
                for y in gap_years:
                    extrap_deployment = extrapolated[i][y]

                    unit_cost = last_known_unit_cost(
                        detailed_costs=detailed_costs,
                        tech=i,
                        year=y
                    )

                    extrap_total_cost = unit_cost * extrap_deployment / 1e6

                    detailed_costs.append({
                        "Year": y,
                        "Technology": i,
                        "Demand": params.cdr_demand[y],
                        "Storage Capacity": params.storage_capacity[y],
                        "Limits": params.limits[i, y],
                        "Optimized Deployment (MtCO₂)": extrap_deployment,
                        "Total Cost ($M)": extrap_total_cost,
                        "Penalty Demand": 0,
                        "Penalty Storage": 0,
                        "Source": "Extrapolated"
                    })

        # ✅ Convert and Save
        optimized_results = pd.DataFrame(results)
            
        if not optimized_results.empty:
            #optimized_results.to_excel(f"./Output/{output_folder}/{output}_Optimized_CDR.xlsx", index=False)
            print(f"✅ Results saved successfully! {len(optimized_results)} entries recorded.")
        else:
            print("⚠️ No valid results to save (model was infeasible).")
            
    print(optimized_results.head())  # Check for duplicate Years/Technologies
    
    # Check for duplicates
    dupes = optimized_results.duplicated(subset=["Year", "Technology"], keep=False)
    if dupes.any():
        print("⚠️ Duplicate (Year, Technology) entries found! Fixing...")
        optimized_results = optimized_results.drop_duplicates(subset=["Year", "Technology"])
        
        # === Print summary of feasibility ===
    if infeasible_periods:
        print("\n🚫 Infeasible periods detected in the following ranges:")
        for start, end in infeasible_periods:
            print(f" - {start} to {end}")
    else:
        print("\n✅ All optimization periods were feasible.")
        
    detailed_df = pd.DataFrame(detailed_costs)
    detailed_df = extrapolate_penalty(detailed_df, penalty_cdr, penalty_storage, storage_based_technologies)
    detailed_df.to_excel(f"./Output/{output_folder}/{output}_Costs_CDR.xlsx", index=False)

    # === Visualization ===
    pivot_df = (optimized_results.pivot(index="Year", columns="Technology", values="Optimized Deployment")
                .reindex(columns=custom_order).sort_index())
    pivot_df = pivot_df.reindex(columns=custom_order)

    # Identify gaps in years
    years = pivot_df.index.to_numpy()
    gaps = np.where(np.diff(years) > 1)[0]
    feasible_years = pivot_df.index.to_numpy()
    all_years = np.arange(feasible_years.min(), feasible_years.max() + 1)
    infeasible_years = np.setdiff1d(all_years, feasible_years)
    total_deployment = pivot_df.sum(axis=1)

    # Insert NaN rows after gaps
    blocks = []
    for i, year in enumerate(pivot_df.index):
        blocks.append(pivot_df.loc[[year]])  # ← keep as DataFrame
        if i in gaps:
            nan_block = pd.DataFrame(np.nan, index=[year + 0.5], columns=pivot_df.columns)
            blocks.append(nan_block)

    pivot_df_gap = pd.concat(blocks)

    plt.figure(figsize=(12, 6))
    plt.stackplot(
        pivot_df_gap.index, 
        pivot_df_gap.T,  
        labels=custom_order,  
        colors=[technology_colors[tech] for tech in custom_order]
    )

        # Shade infeasible years
    for y in infeasible_years:
        prev_years = feasible_years[feasible_years < y]
        if len(prev_years) == 0:
            continue

        y_prev = prev_years.max()
        height = total_deployment.loc[y_prev]
        plt.fill_between([y - 0.5, y + 0.5], 0, height, facecolor="none",
                           hatch="///", edgecolor="grey", linewidth=0, zorder = 5)
        
    plt.xlabel("Year")
    plt.ylabel("Optimized Deployment (MtCO₂/year)")
    plt.title(f"Optimized CDR Deployment with {params.foresight}-Year Foresight, Decision Interval of {params.decision_period} years and {round((params.max_growth_rate-1)*100,0)}% growth")

    handles, labels = plt.gca().get_legend_handles_labels()
    ordered_handles_labels = [(handles[custom_order.index(lbl)], lbl) for lbl in labels]
    ordered_handles, ordered_labels = zip(*ordered_handles_labels)
    plt.legend(ordered_handles, ordered_labels, title="CDR Technology", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./Output/{output_folder}/{output}")
    #plt.show()
    plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pathway Script")
    parser.add_argument("-i", "--input", help="Input file", required=True)
    parser.add_argument("-o", "--output", help="Output Scenario", required=True)
    parser.add_argument("-s", "--scenario", help="Scenario choice", required=True)
    parser.add_argument("--switch", help="Select if switch", required=False, default="yes")
    args = parser.parse_args()
    input = args.input
    output = args.output
    scenario_choice = args.scenario
 
    if args.switch == "no":
        output_folder = "High"
    else:
        output_folder = "Mod" 

    import json
    with open("scenarios_high.json", "r") as f:
        scenarios = json.load(f)

    if scenario_choice in scenarios:
        scenario = scenarios[scenario_choice]
    else: 
        raise

    file_path = f"./Input/{input}"
    
    technologies = ["DACCS", "BECCS", "OAE", "AR", "EW", "BC"]
    storage_based_technologies = ["DACCS", "BECCS"]
    
    max_growth_rate = 1.1
    min_decline_rate = 0.9
    foresight = 70
    decision_period=35

    discount_rate = 0.01
    penalty_cdr = 500000000 #$/MtCO2 = $500 Millions/Mt
    penalty_storage = 500000000 #$/MtCO2 = $500 Millions/Mt
    
    # Define custom colors for each technology
    technology_colors = {
        'DACCS': 'maroon',   
        'BECCS': 'darkgreen',      
        'OAE': 'cornflowerblue',              
        'BC': 'purple',
        'AR': 'limegreen',         
        'EW': 'slategray'         
    }
    
    # Define the **desired order** of technologies in the stacked plot and legend
    custom_order = ['AR', 'BC' , 'EW','BECCS', 'OAE', 'DACCS']
    
    run(
        file_path = file_path,
        scenario = scenario,
        technologies = technologies,
        storage_based_technologies = storage_based_technologies,
        foresight = foresight,
        decision_period = decision_period,
        discount_rate = discount_rate,
        max_growth_rate = max_growth_rate,
        min_decline_rate = min_decline_rate,
        penalty_cdr = penalty_cdr,
        penalty_storage = penalty_storage,
        technology_colors = technology_colors,
        custom_order = custom_order,
        output = output, 
        output_folder = output_folder
    )

# Run with:
# python Run.py -i <input_file_name> -s <scenario_input(low,mod,high)> -o <output_file_name> --switch <no: if scenarios>