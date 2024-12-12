import argparse
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def load_data(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Print diagnostic information
    print(f"Original number of rows: {len(data)}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Data types:\n{data.dtypes}")
    print("\nUnique values in each column:")
    for col in data.columns:
        print(f"{col}: {data[col].nunique()}")
    
    # Check for missing values
    print("\nMissing values:")
    print(data.isnull().sum())
    
    # Print the first few rows
    print("\nFirst few rows of the data:")
    print(data.head())
    
    return data

def check_cell_sizes(data):
    print("\nCell sizes:")
    factors = ['persona', 'model_type', 'model_size', 'trait']
    cell_sizes = data.groupby(factors).size().reset_index(name='count')
    print(cell_sizes)
    
    print("\nEmpty cells:")
    empty_cells = cell_sizes[cell_sizes['count'] == 0]
    print(empty_cells)

def run_anova(data):
    # Prepare the data for ANOVA
    formula = "score ~ C(persona) + C(model_type) + C(model_size) + C(trait)"
    
    try:
        # Fit the OLS model
        model = ols(formula, data=data).fit()
        
        # Perform ANOVA
        anova_table = anova_lm(model, typ=2)
        
        print("\nANOVA Results:")
        print(anova_table)
        
        print("\nModel Summary:")
        print(model.summary())
        
        return model, anova_table
    except Exception as e:
        print(f"Error occurred during ANOVA: {str(e)}")
        return None, None

def post_hoc_tests(data):
    # Tukey's HSD test for each factor
    factors = ['persona', 'model_type', 'model_size', 'trait']
    for factor in factors:
        try:
            tukey_results = pairwise_tukeyhsd(data['score'], data[factor])
            print(f"\nTukey's HSD for {factor}:")
            print(tukey_results)
        except Exception as e:
            print(f"Error occurred during post-hoc test for {factor}: {str(e)}")

def main(file_path):
    data = load_data(file_path)
    check_cell_sizes(data)
    model, anova_table = run_anova(data)
    if model is not None:
        post_hoc_tests(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform ANOVA on survey data")
    parser.add_argument("input_file", help="Path to the input CSV file")
    args = parser.parse_args()
    
    main(args.input_file)
