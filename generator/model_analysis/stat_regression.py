import argparse
from sys import exit

import numpy as np
import pandas as pd
import scipy.stats as stats

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.miscmodels.ordinal_model import OrderedModel

def run_regression_analysis(input_file, formula, model_type, distr='Binomial', link='Logit', output_file=None):
    """Run regression analysis with given parameters"""
    df = pd.read_csv(input_file)

    if 'anova' in model_type.lower():
        model = smf.ols(formula, data=df).fit()
        res = anova_lm(model, typ=2, robust='hc3')
    elif 'logit' in model_type.lower():
        specification = OrderedModel.from_formula(formula, df, distr='logit')
        model = specification.fit(method='bfgs', disp=False, maxiter=10)
    elif 'bayes' in model_type.lower():
        left, right = formula.split('~')
        variables = ['persona', 'model_type', 'model_size', 'Category']
        random = {k: f'0 + {v}' for k, v in variables.items()}
        specification = BinomialBayesMixedGLM.from_formula(formula, random, df)
        model = specification.fit_vb()
    elif 'glm' in model_type.lower():
        if distr == 'Binomial' and link == 'Logit':
            family = sm.families.Binomial(sm.families.links.Logit())
        model = smf.glm(formula=formula, data=df, family=family).fit(maxiter=100)
    else:
        exit('Invalid model specification.')

    # Output results
    if output_file:
        with open(output_file, 'w') as f:
            f.write('*** Model Summary ***\n\n')
            f.write(f'Model specification: {formula}\n')
            f.write(f'{model.summary()}\n\n\n')
            if 'res' in locals():
                f.write(f'*** {model_type.upper()} Results ***\n\n')
                f.write(f'{res}')
    else:
        print('*** Model Summary ***\n\n')
        print(f'Model specification: {formula}\n')
        print(f'{model.summary()}\n\n\n')
        if 'res' in locals():
            print(f'*** {model_type.upper()} Results ***\n\n')
            print(f'{res}')

    return model, res if 'res' in locals() else None

def main():
    parser = argparse.ArgumentParser(description='Runs regression analysis.')
    parser.add_argument('input_file', help='Path to data file in csv format.')
    parser.add_argument('--formula', help='Model specification. Variable names must match data columns.', required=True)
    parser.add_argument('--model-type', help='Choices: anova, glm, logit, and Bayes', required=True)
    parser.add_argument('--distribution', help='For GLM, specify the distribution family, options: Gamma, Gaussian, Binomial', default='Binomial')
    parser.add_argument('--link', help='For GLM, specify the link function, options: Log, Logit, Probit', default='Logit')
    parser.add_argument('--output', help='File path to save output.')
    
    args = parser.parse_args()
    
    run_regression_analysis(
        args.input_file,
        args.formula,
        args.model_type,
        args.distribution,
        args.link,
        args.output
    )

if __name__ == "__main__":
    main()
