import pandas as pd
import re

# === Step 1: Load Excel File ===
def load_excel(file_path):
    """Loads an Excel file and returns a dictionary of dataframes for each sheet."""
    xls = pd.ExcelFile(file_path)
    return {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}

# === Step 2: Define Scenario Selection (General & Technology-Specific Variables) ===
def select_scenario(sheets_data, scenario):
    """
    Extracts relevant columns based on the given scenario.
    - General variables (e.g., CDR Demand, Storage Capacity) are selected once.
    - Technology-specific variables (e.g., Costs, Limits) are selected per technology.
    """
    scenario_data = {}
    
    for sheet, selection in scenario.items():
        if sheet in sheets_data:
            df = sheets_data[sheet]
            selected_columns = ["Year"]
            
            # If the selection is a single string (not a dictionary), it's a general variable
            if isinstance(selection, str):  
                selected_columns += [col for col in df.columns if selection in col]
            else:  # Otherwise, it's technology-specific
                for tech, level in selection.items():
                    selected_columns += [col for col in df.columns if tech in col and level in col]
            
            scenario_data[sheet] = df[selected_columns]
    
    return scenario_data

# === Step 3: Clean, Convert, and Interpolate Data ===

def clean_convert_interpolate(df):
    """Cleans non-numeric values, removes duplicates, and interpolates missing years."""

    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])

    def clean_numeric(value):
        if isinstance(value, str):
            return float(re.sub(r"[^\d.]", "", value)) if re.search(r"\d", value) else None
        return value

    for col in df.columns[1:]:
        df[col] = df[col].astype(str).map(clean_numeric)

    # Set Year as index for easier manipulation
    df = df.set_index("Year")

    # Reindex for interpolation and sort
    df_interpolated = df.reindex(range(2030, 2101)).sort_index().interpolate(method="linear").reset_index()

    return df_interpolated


def clean_column_name(col):
    """
    Cleans column names by:
    - Removing 'Low', 'Medium', or 'High' if they appear at the end.
    - Removing units enclosed in parentheses (e.g., '(tCO₂)', '(GJ)', etc.).
    - Removing trailing underscores.
    """
    col = re.sub(r"\s*(Low|Medium|High)$", "", col)  # Remove scenario suffixes
    col = re.sub(r"\s*\(.*?\)", "", col)  # Remove units in parentheses
    col = re.sub(r"_+$", "", col)  # Remove trailing underscores
    return col

def rename_columns_with_sheet(sheet_name, df):
    """
    Renames columns to include their sheet name for better tracking 
    and ensures column names are cleaned.
    """
    
    return df.rename(columns={col: f"{sheet_name}_{clean_column_name(col)}" for col in df.columns if col != "Year"})

def scale_selected_columns(df, scale_mapping):
    scaled_df = df.copy()
    for pattern, scale in scale_mapping.items():
        matching_cols = [col for col in df.columns if re.search(pattern, col)]
        for col in matching_cols:
            scaled_df[col] = scaled_df[col] * scale  # ✅ Use multiplication
    return scaled_df


scale_mapping = {
    r"^CDR_Demand": 1e-6,
    r"^Limit": 1e-6,
    r"^Storage_Capacity": 1e-6,
    r"^Cost_": 1e6,              # ✅ Only match columns that *start with* "Cost_"
    r"^Subsidy_": 1e6,
}


def process_scenario_data(file_path, scenario):
    """
    Loads the Excel file, extracts scenario-specific data, 
    cleans it, interpolates missing values, and standardizes column names.
    """
    
    # Load all sheets from the Excel file
    sheets_data = load_excel(file_path)
    
    # Extract relevant scenario-specific data
    scenario_data = select_scenario(sheets_data, scenario)

    # Process each selected sheet: clean, interpolate, and rename columns
    processed_data = {
        sheet: rename_columns_with_sheet(sheet, clean_convert_interpolate(df))
        for sheet, df in scenario_data.items()
    }

    # Concatenate all processed dataframes along columns
    final_df = pd.concat(processed_data.values(), axis=1)
    
    # Remove duplicate columns if any
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # Apply final column name cleaning to ensure consistency
    final_df.columns = [clean_column_name(col) for col in final_df.columns]
    final_df = scale_selected_columns(final_df, scale_mapping) 

    return final_df