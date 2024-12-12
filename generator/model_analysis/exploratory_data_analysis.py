# Modify exploratory_data_analysis.py to make it more modular:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(input_file, output_dir):
    """Main function to run exploratory data analysis"""
    df = pd.read_csv(input_file)
    
    # 1. Data Preprocessing
    with open(os.path.join(output_dir, 'data_overview.txt'), 'w') as f:
        f.write("Data Overview:\n")
        f.write(str(df.head()))
        f.write("\n\nData Summary:\n")
        f.write(str(df.describe()))
        f.write("\n\nMissing Values:\n")
        f.write(str(df.isnull().sum()))

    # Convert categorical variables to category type
    df['model'] = df['model'].astype('category')
    df['persona'] = df['persona'].astype('category')
    df['trait'] = df['trait'].astype('category')

    # Create numerical encodings for categorical variables
    df['model_code'] = df['model'].cat.codes
    df['persona_code'] = df['persona'].cat.codes
    df['trait_code'] = df['trait'].cat.codes

    # 2. Correlation matrix
    data = df[['model_code','persona_code','trait_code']]
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.svg'), format='svg')
    plt.close()

    # 3. Distribution of scores
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='score', hue='trait', kde=True, element="step")
    plt.title('Distribution of Scores by Trait')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distributions.svg'), format='svg')
    plt.close()

    # 4. Scores by model and persona
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(x='model', y='score', data=df, ax=ax1)
    ax1.set_title('Scores by Model')
    sns.boxplot(x='persona', y='score', data=df, ax=ax2)
    ax2.set_title('Scores by Persona')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scores_by_factors.svg'), format='svg')
    plt.close()

if __name__ == "__main__":
    from sys import argv
    run_eda(argv[1], argv[2])
