import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def load_data(file_path):
    # Define dtypes for each column
    dtypes = {
        'persona': 'category',
        'model': 'category',
        'model_type': 'category',
        'model_size': 'category',
        'trait': 'category',
        'score': 'float32'  # Use float32 instead of float64
    }

    # Read the CSV file with specified dtypes
    data = pd.read_csv(file_path, dtype=dtypes)
    data['score'] = data['score'].dropna().round().astype('int8')  # Use int8 for Likert scale

    if 'model_type' or 'model_size' not in data.columns:
        data[['model_type', 'model_size']]=data['model'].str.split('-', expand=True)

    # Check for missing values
    print("\nMissing values before cleaning:")
    print(data.isnull().sum())

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Create a unique identifier for each combination of persona, model_type, and model_size
    #data['id'] = pd.factorize(data['persona'].astype(str) + '_' + data['model_type'].astype(str) + '_' + data['model_size'].astype(str))[0]
    data['id'] = pd.factorize(data['persona'].astype(str) + '_' + data['model_type'].astype(str))[0]
    data['id'] = data['id'].astype('int32')  # Use int32 for id

    # Add trial column
    #data['trial'] = data.groupby(['persona', 'model_type', 'model_size', 'trait'], observed=True).cumcount().astype('int16')
    data['trial'] = data.groupby(['persona', 'model_type', 'trait'], observed=True).cumcount().astype('int16')

    # Ensure 'id' is contiguous from 0 to n-1
    data['id'] = pd.factorize(data['id'])[0]

    # Print diagnostic information
    print(f"Number of unique ids: {data['id'].nunique()}")
    print(f"Number of rows: {len(data)}")
    print(f"Max id value: {data['id'].max()}")
    print(f"Min id value: {data['id'].min()}")
    #print(f"Number of unique combinations: {data.groupby(['persona', 'model_type', 'model_size', 'trait'], observed=True).ngroups}")
    print(f"Number of unique combinations: {data.groupby(['persona', 'model_type', 'trait'], observed=True).ngroups}")


    return data

def check_multicollinearity(data):
    # Create dummy variables for categorical factors
    #dummy_data = pd.get_dummies(data[['persona', 'model_type', 'model_size', 'trait']], drop_first=True)
    dummy_data = pd.get_dummies(data[['persona', 'model_type', 'trait']], drop_first=True)
    dummy_data = dummy_data.astype('float32')
    dummy_data = dummy_data.fillna(0)

    # Add constant term
    dummy_data = add_constant(dummy_data)
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = dummy_data.columns
    vif_data["VIF"] = [variance_inflation_factor(dummy_data.values, i) for i in range(dummy_data.shape[1])]
    
    print("Variance Inflation Factors:")
    print(vif_data)
    
    # Identify highly collinear features
    high_vif = vif_data[vif_data["VIF"] > 5].sort_values("VIF", ascending=False)
    if not high_vif.empty:
        print("\nWarning: The following features show high multicollinearity (VIF > 5):")
        print(high_vif)
        
        # Identify perfect multicollinearity
        perfect_collinearity = vif_data[vif_data["VIF"] > 1000]
        if not perfect_collinearity.empty:
            print("\nPerfect multicollinearity detected. The following factors are linearly dependent:")
            for feature in perfect_collinearity["Feature"]:
                original_feature = feature.split('_')[0]
                print(f"- {original_feature}")
    else:
        print("\nNo high multicollinearity detected among the factors.")

def check_normality(data):
    _, p_value = stats.shapiro(data['score'].sample(min(5000, len(data))))  # Sample to reduce computation
    print(f"Shapiro-Wilk test for normality: p-value = {p_value:.4f}")
    
    plt.figure(figsize=(10, 6))
    qqplot(data['score'].sample(min(5000, len(data))), line='s')
    plt.title("Q-Q Plot of Scores (Sample)")
    plt.savefig('normality_qq_plot.png')
    plt.close()

def check_homoscedasticity(data):
    #groups = data.groupby(['persona', 'model_type', 'model_size', 'trait'], observed=True)
    groups = data.groupby(['persona', 'model_type', 'trait'], observed=True)
    sample_size = min(1000, len(data) // len(groups))
    sampled_groups = [group.sample(sample_size, replace=True) for _, group in groups]
    levene_results = stats.levene(*[group['score'].values for group in sampled_groups])
    print(f"Levene's test for homoscedasticity: p-value = {levene_results.pvalue:.4f}")

def run_anova(data):
    # Prepare the data for mixed linear model
    #data['intercept'] = 1
    factors = ['persona', 'model_type', 'trait', 'trial']

    # Create the formula for fixed effects
    #fixed_effects = ' + '.join(['C(' + f + ')' for f in factors])
    # No intercept
    fixed_effects = ' * '.join([f"({f} - 1)" for f in factors])
    formula = f"score ~ {fixed_effects}"

    # Print diagnostic information
    print(f"Formula: {formula}")
    print(f"Number of rows in data: {len(data)}")
    print(f"Number of unique ids: {data['id'].nunique()}")

    # Ensure 'id' is contiguous from 0 to n-1
    data['id'] = pd.factorize(data['id'])[0]

    # Additional check to ensure 'id' is properly formatted
    assert data['id'].max() == data['id'].nunique() - 1, "ID range is not contiguous from 0 to n-1"
    
    # Fit the mixed linear model
    try:
        model = mixedlm(formula, data, groups='id')
        results = model.fit(method='powell')
        
        print("\nMixed Linear Model Results:")
        print(results.summary())
        
        return results
    except Exception as e:
        print(f"An error occurred during model fitting: {str(e)}")
        print("Data shape:", data.shape)
        print("Unique values in 'id':", data['id'].unique())
        print("Data types:", data.dtypes)
        raise

def post_hoc_tests(data):
    # Tukey's HSD test for each factor
    #factors = ['persona', 'model_type', 'model_size', 'trait']
    factors = ['persona', 'model_type', 'trait']
    for factor in factors:
        unique_groups = data[factor].nunique()
        if unique_groups < 2:
            print(f"\nSkipping Tukey's HSD for {factor}: Insufficient groups (only {unique_groups} group found)")
            continue
        mc = MultiComparison(data['score'], data[factor])
        tukey_results = mc.tukeyhsd()
        print(f"\nTukey's HSD for {factor}:")
        print(tukey_results)

def main(file_path):
    data = load_data(file_path)
    
    print("Data info:")
    print(data.info())
    
    print("\nChecking multicollinearity:")
    check_multicollinearity(data)
    
    print("\nChecking assumptions:")
    check_normality(data)
    check_homoscedasticity(data)
    
    anova_results = run_anova(data)
    
    post_hoc_tests(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Memory-Efficient Repeated Measures ANOVA on survey data")
    parser.add_argument("input_file", help="Path to the input CSV file")
    args = parser.parse_args()
    
    main(args.input_file)
