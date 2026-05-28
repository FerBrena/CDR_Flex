import pyomo.environ as pyo

def get_param(df, base_name, years, technologies, tech_specific=False, default=0):
    param_dict = {}
    if tech_specific:
        for tech in technologies:
            col_name = f"{base_name}_{tech}"
            if col_name in df.columns:
                for year in years:
                    param_dict[(tech, year)] = df.loc[df["Year"] == year, col_name].values[0]
            else:
                pass
    else:
        col_name = base_name
        if col_name in df.columns:
            param_dict = {year: df.loc[df["Year"] == year, col_name].values[0] for year in years}
        else:
            param_dict = {year: default for year in years}
    return param_dict

class Params:
    def __init__(self, *, 
                 dataframe, 
                 years, 
                 foresight, 
                 decision_period,
                 discount_rate, 
                 max_growth_rate,
                 min_decline_rate,
                 penalty_cdr,
                 penalty_storage,
                 technologies,
                 storage_based_technologies,
                 previous_decision = None):
        df = dataframe
        self.cost = get_param(df, "Cost", years, technologies, tech_specific=True)
        self.subsidy = get_param(df, "Subsidy", years, technologies, tech_specific=True)
        self.cdr_demand = get_param(df, "CDR_Demand", years, technologies)
        self.limits = get_param(df, "Limit", years, technologies, tech_specific=True)
        self.storage_capacity = get_param(df, "Storage_Capacity", years, technologies)      
        self.years = years       
        self.foresight = foresight
        self.decision_period=decision_period
        self.discount_rate = discount_rate
        self.max_growth_rate = max_growth_rate
        self.min_decline_rate = min_decline_rate
        self.penalty_cdr = penalty_cdr
        self.penalty_storage = penalty_storage
        self.technologies = technologies
        self.storage_based_technologies = storage_based_technologies
        if previous_decision is None:
            previous_decision = {}
        self.previous_decision = previous_decision

def make_functions(params: Params):
    def excess_storage_constraint(model, t):
        return (model.excess_storage[t] >= 
                sum(model.x[i, t] for i in 
                    params.storage_based_technologies if i in model.I) 
                - params.storage_capacity[t])
 
    def total_cost(model):
        base_year = min(model.T)
        discounted_costs = sum(
            sum(
                (max(params.cost.get((i, t), 0) - params.subsidy.get((i, t), 0), 0) * model.x[i, t])
                for i in model.I
            ) / ((1 + params.discount_rate) ** (t - base_year))
            for t in model.T
        )
        undiscounted_penalties = sum(
            params.penalty_storage * model.excess_storage[t] + params.penalty_cdr * model.unmet_cdr[t]
            for t in model.T
        )
        return discounted_costs + undiscounted_penalties

    def cdr_demand_constraint(model, t):
        return sum(model.x[i, t] for i in model.I) + model.unmet_cdr[t] >= params.cdr_demand[t]
    
    def tech_limit_constraint(model, i, t):
        return model.x[i, t] <= params.limits.get((i, t), 1e12)  # Ensure correct indexing
    
    def get_prev_value(model, i, t):
        """Previous window: Return numeric prev_value (float)
        Same window: Return pyomo Var model.x"""
        prev_year = t - 1
        if t == min(model.T) and t != min(params.years):
            # numeric previous value from earlier solved window
            return params.previous_decision[i].get(prev_year, 0)
        else:
            # py variable from the current window
            return model.x[i, prev_year]
    
    def growth_activation_lower(model, i, t):
        """Ensure growth constraints activate only when deployment is significant."""
        if t == min(params.years):  
            return pyo.Constraint.Skip  # Skip for the first year

        prev_value = get_prev_value(model, i, t)
    
        max_limit_first_year = params.limits.get((i, min(params.years)), 1)
        threshold = .01 * max_limit_first_year
        BIG_M = params.limits.get((i, max(params.years)))

        # if growth active = 1 then prev value >= threshold (big deploy)
        # if growth active = 0 then prev value >= -M (no lower bound) Always satisfied, can jump (small deployment)
        return prev_value - threshold >= -BIG_M * (1 - model.growth_active[i, t])

    
    def growth_activation_upper(model, i, t):
        """Ensure technology deployment respects the growth limit."""
        if t == min(params.years):  # Skip first year of simulation
            return pyo.Constraint.Skip
    
        prev_value = get_prev_value(model, i, t)
    
        max_limit_first_year = params.limits.get((i, min(params.years)), 1)
        threshold = .01 * max_limit_first_year
        BIG_M = params.limits.get((i, max(params.years)))
        epsilon = 1e-9

        # if growth active = 1 then prev value <= threshold + M Always satisfied (can be big deployment)
        # if growth active = 0 then prev value <= threshold (must be small)
        return prev_value - (threshold - epsilon) <= BIG_M * model.growth_active[i, t]
    
    def enforce_growth(model, i, t):
        """Ensure growth constraints apply across planning horizons."""
        if t == min(params.years):  # Skip first year of full simulation
            return pyo.Constraint.Skip
    
        prev_value = get_prev_value(model, i, t)
    
        max_limit_first_year = params.limits.get((i, min(params.years)), 1)
        threshold = 0.01 * max_limit_first_year
    
        if isinstance(prev_value, (int, float)):
            # For numeric values (prev window)
            if prev_value < threshold:
                return model.x[i, t] <= max_limit_first_year #if deployment is small it can jump
            else:
                return model.x[i, t] <= params.max_growth_rate * prev_value #if it's big it follows growth rate
        else:
            # The growth_active binary will control whether it's enforced
            # growth active = 1 then we use growth rate 
            # growth active = 0 then we use limits (small deployment)
            return model.x[i, t] <= (params.max_growth_rate * prev_value * model.growth_active[i, t] + 
                                          max_limit_first_year * (1 - model.growth_active[i, t]))

    def enforce_decline(model, i, t):
        """Allow phase-out to zero if previous year was 'tiny' (below threshold).
        Enforce decline rate when previous year was 'active' (above threshold)."""
        if t == min(params.years):  # Skip first year of full simulation
            return pyo.Constraint.Skip
    
        prev_value = get_prev_value(model, i, t)

        max_limit_first_year = params.limits.get((i, min(params.years)), 1)
        threshold = 0.01 * max_limit_first_year
    
        if isinstance(prev_value, (int, float)):
            if prev_value <= threshold:
                # previous year was "tiny" -> allow x to be 0 (no decline constraint)
                return pyo.Constraint.Skip
            else:
                return model.x[i, t] >= params.min_decline_rate * prev_value
        else:
            # prev_value is a pyomo Var: use growth_active (which signals prev_value >= threshold)
            # growth_active == 1 -> enforce decline; 
            # growth_active == 0 -> allow to go to 0
            return model.x[i, t] >= params.min_decline_rate * prev_value * model.growth_active[i, t]    

    
    def min_deploy_lower(model, i, t):
        """If deployed, x must be at least % of first-year capacity."""
        max_limit_first_year = params.limits.get((i, min(params.years)), 1)
        min_threshold = 0.01 * max_limit_first_year #<----------------------Change floor if needed
        return model.x[i, t] >= min_threshold * model.is_deployed[i, t]

    def min_deploy_upper(model, i, t):
        """Link is_deployed binary to actual deployment via Big-M."""
        BIG_M = params.limits.get((i, max(params.years)), 1e12)
        return model.x[i, t] <= BIG_M * model.is_deployed[i, t] 

    def deploy_growth(model, i, t):
        """is_deployed must be on if growth_active is on."""
        if t == min(model.T):  # skip first year
            return pyo.Constraint.Skip
        return model.growth_active[i, t] == model.is_deployed[i, t-1]  

    func_dict = {
        "excess_storage_constraint": excess_storage_constraint,
        "total_cost": total_cost,
        "cdr_demand_constraint": cdr_demand_constraint,
        "tech_limit_constraint": tech_limit_constraint,   
        "growth_activation_upper": growth_activation_upper,  
        "growth_activation_lower":growth_activation_lower,
        "enforce_growth": enforce_growth,  
        "enforce_decline":enforce_decline,
        "min_deploy_lower": min_deploy_lower,
        "min_deploy_upper": min_deploy_upper,
        "deploy_growth": deploy_growth,
        }
    return func_dict