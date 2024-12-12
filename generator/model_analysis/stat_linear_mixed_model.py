import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pdb 

def load_and_prepare_data(path_to_data_file):
    print("\nLoading and preparing data...")
    df = pd.read_csv(path_to_data_file)
    
    # Create log model size
    df['model_size'] = df['model'].str.extract(r'(\d+)b?$').astype(float)
    df['log_model_size'] = np.log(df['model_size'])
    
    # Ensure categorical variables are treated as such
    df['model'] = df['model_type'].astype('category')
    df['persona'] = df['persona'].astype('category')
    df['trait'] = df['trait'].astype('category')
    
    print("\nData shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())
    
    return df

def check_assumptions(df):
    print("\nChecking model assumptions...")
    
    # Normality test (using a sample if dataset is large)
    sample_size = min(5000, len(df))
    stat, p_value = stats.normaltest(df['score'].sample(sample_size))
    print(f"\nNormality test p-value: {p_value}")
    
    # Homoscedasticity test
    groups = df.groupby(['model', 'persona', 'trait'])
    scores_by_group = [group['score'].values for _, group in groups]
    stat, p_value = stats.levene(*scores_by_group)
    print(f"Homoscedasticity test p-value: {p_value}")
    
    return

def fit_hierarchical_models(df):
    print("\nFitting hierarchical models...")
    
    # List to store model results
    model_results = []
    
    try:
        # 1. Base model with main effects only
        print("\nFitting base model...")
        base_formula = 'score ~ C(model) + C(persona) + C(trait) + log_model_size'
        base_model = smf.mixedlm(base_formula, data=df, groups='id')
        base_results = base_model.fit()
        model_results.append(('Base Model', base_results))
        
        print(" ++++++ START BASE MODEL SUMMARY ++++++")
        print(base_results.summary())
        print(" ++++++ END BASE MODEL SUMMARY ++++++")
        
        # 2. Model with two-way interactions
        print("\nFitting model with two-way interactions...")
        two_way_formula = ('score ~ C(model) + C(persona) + C(trait) + log_model_size + '
                          'C(model):C(persona) + C(model):C(trait) + C(persona):C(trait) + '
                          'log_model_size:C(persona) + log_model_size:C(trait)')
        two_way_model = smf.mixedlm(two_way_formula, data=df, groups='id')
        two_way_results = two_way_model.fit()
        model_results.append(('Two-way Interactions Model', two_way_results))
        
        print(" ++++++ START TWO-WAY INTERACTIONS MODEL SUMMARY ++++++")
        print(two_way_results.summary())
        print(" ++++++ END TWO-WAY INTERACTIONS MODEL SUMMARY ++++++")
        
        # 3. Model with three-way interactions
        print("\nFitting model with three-way interactions...")
        three_way_formula = (two_way_formula + ' + C(model):C(persona):C(trait) + '
                           'log_model_size:C(persona):C(trait)')
        three_way_model = smf.mixedlm(three_way_formula, data=df, groups='id')
        three_way_results = three_way_model.fit()
        model_results.append(('Three-way Interactions Model', three_way_results))
        
        print(" ++++++ START THREE-WAY INTERACTIONS MODEL SUMMARY ++++++")
        print(three_way_results.summary())
        print(" ++++++ END THREE-WAY INTERACTIONS MODEL SUMMARY ++++++")
        
        # Compare models
        print("\nModel Comparisons:")
        print("\nAIC Comparisons:")
        for name, result in model_results:
            print(f"{name}: {result.aic}")
        
        # Likelihood ratio tests
        def lr_test(model1, model2):
            lr_stat = -2 * (model1.llf - model2.llf)
            df_diff = model2.df_resid - model1.df_resid
            p_value = stats.chi2.sf(lr_stat, df_diff)
            return lr_stat, p_value
        
        print("\nLikelihood Ratio Tests:")
        print("Base vs Two-way:", lr_test(base_results, two_way_results))
        print("Two-way vs Three-way:", lr_test(two_way_results, three_way_results))
        
    except Exception as e:
        print(f"\nError in model fitting: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    return model_results

def plot_interaction_effects(df, model_results):
    print("\nCreating interaction plots...")
    
    # Plot model size by persona interaction
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='log_model_size', y='score', hue='persona', alpha=0.5)
    plt.title('Model Size by Persona Interaction')
    plt.savefig('model_size_persona_interaction.png')
    
    # Plot trait by persona interaction
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='trait', y='score', hue='persona')
    plt.xticks(rotation=45)
    plt.title('Trait by Persona Interaction')
    plt.tight_layout()
    plt.savefig('trait_persona_interaction.png')
    
    # Plot model by trait interaction
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df, x='model', y='score', hue='trait')
    plt.xticks(rotation=90)
    plt.title('Model by Trait Interaction')
    plt.tight_layout()
    plt.savefig('model_trait_interaction.png')

def main():
    # Set the path to your data file
    path_to_data_file = "/Users/tommasotosato/Desktop/PsychoBenchPPSP/results_output_figs/bfi_data_formatted.csv"
    
    try:
        # Load and prepare data

        df = load_and_prepare_data(path_to_data_file)
        # pdb.set_trace() 

        # Check model assumptions
        check_assumptions(df)
        
        # Fit hierarchical models
        model_results = fit_hierarchical_models(df)
        
        if model_results:
            # Create interaction plots
            plot_interaction_effects(df, model_results)
            
            print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()