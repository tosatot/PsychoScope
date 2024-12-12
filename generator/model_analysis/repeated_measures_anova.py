import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.power import FTestAnovaPower
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def load_data(file_path):
    # Define dtypes for each column
    dtypes = {
        'persona': 'category',
        'model_type': 'category',
        'model_size': 'category',
        'trait': 'category',
        'score': 'float64'  # Assumes Likert scale from 1-5 or 1-7
    }
    
    # Read the CSV file with specified dtypes
    data = pd.read_csv(file_path, dtype=dtypes)
    data['score'] = data['score'].dropna().round().astype('int16')

    # Create a unique identifier for each combination of persona, model_type, and model_size
    data['id'] = pd.factorize(data['persona'].astype(str) + '_' + data['model_type'].astype(str) + '_' + data['model_size'].astype(str))[0]

    # Add trial column
    data['trial'] = data.groupby(['persona', 'model_type', 'model_size', 'trait']).cumcount().astype('int16')
    
    return data

def check_multicollinearity(data):
    # Create dummy variables for categorical factors
    dummy_data = pd.get_dummies(data[['persona', 'model_type', 'model_size', 'trait']], drop_first=True)
    dummy_data = dummy_data.astype(float)
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
    _, p_value = stats.shapiro(data['score'])
    print(f"Shapiro-Wilk test for normality: p-value = {p_value:.4f}")
    
    plt.figure(figsize=(10, 6))
    qqplot(data['score'], line='s')
    plt.title("Q-Q Plot of Scores")
    plt.savefig('normality_qq_plot.png')
    plt.close()

def check_homoscedasticity(data):
    groups = data.groupby(['persona', 'model_type', 'model_size', 'trait'])
    levene_results = stats.levene(*[group['score'].values for name, group in groups])
    print(f"Levene's test for homoscedasticity: p-value = {levene_results.pvalue:.4f}")

def run_anova(data):
    # Run ANOVA with intercept
    anova_with_intercept = AnovaRM(data, 'score', 'id', within=['persona', 'model_type', 'model_size', 'trait', 'trial']).fit()
    
    # Run ANOVA without intercept
    formula = 'score ~ C(persona) + C(model_type) + C(model_size) + C(trait) + C(trial) - 1'
    anova_without_intercept = ols(formula, data=data).fit()
    
    print("\nRepeated Measures ANOVA Results (with intercept):")
    print(anova_with_intercept.summary())
    
    print("\nRepeated Measures ANOVA Results (without intercept):")
    print(anova_without_intercept.summary())
    
    # Compare model fit
    print("\nModel Comparison:")
    print(f"R-squared (with intercept): {anova_with_intercept.rsquared:.4f}")
    print(f"R-squared (without intercept): {anova_without_intercept.rsquared:.4f}")
    print(f"AIC (with intercept): {anova_with_intercept.aic:.4f}")
    print(f"AIC (without intercept): {anova_without_intercept.aic:.4f}")
    
    return anova_with_intercept, anova_without_intercept

def post_hoc_tests(data):
    # Tukey's HSD test for each factor
    factors = ['persona', 'model_type', 'model_size', 'trait']
    for factor in factors:
        mc = MultiComparison(data['score'], data[factor])
        tukey_results = mc.tukeyhsd()
        print(f"\nTukey's HSD for {factor}:")
        print(tukey_results)

def calculate_effect_sizes(data, anova_results):
    # Partial Eta Squared
    df_effect = anova_results.df_num
    df_error = anova_results.df_denom
    f_values = anova_results.f_value
    eta_squared = f_values * df_effect / (f_values * df_effect + df_error)
    print("\nPartial Eta Squared (Effect Sizes):")
    for effect, es in zip(anova_results.results['Source'], eta_squared):
        print(f"{effect}: {es:.4f}")

def power_analysis(data, anova_results):
    # Simplified power analysis for main effects
    ftester = FTestAnovaPower()
    for effect, f_value in zip(anova_results.results['Source'], anova_results.f_value):
        if effect in ['persona', 'model_type', 'model_size', 'trait']:
            df_num = anova_results.df_num[anova_results.results['Source'] == effect].iloc[0]
            df_denom = anova_results.df_denom[anova_results.results['Source'] == effect].iloc[0]
            n = len(data) / (len(data[effect].unique()) * 100)  # Divide by 100 for the number of trials
            power = ftester.solve_power(effect_size=f_value, nobs=n, alpha=0.05, df_num=df_num, df_denom=df_denom)
            print(f"Power for {effect}: {power:.4f}")

def main(file_path):
    data = load_data(file_path)
    
    print("Data info:")
    print(data.info())
    
    print("\nChecking multicollinearity:")
    check_multicollinearity(data)
    
    print("\nChecking assumptions:")
    check_normality(data)
    check_homoscedasticity(data)
    
    anova_with_intercept, anova_without_intercept = run_anova(data)
    
    post_hoc_tests(data)
    
    calculate_effect_sizes(data, anova_with_intercept)
    
    print("\nPower Analysis:")
    power_analysis(data, anova_with_intercept)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Repeated Measures ANOVA on survey data")
    parser.add_argument("input_file", help="Path to the input CSV file")
    args = parser.parse_args()
    
    main(args.input_file)
