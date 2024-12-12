import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import json
import os




def get_trait_limits(mean_stats, var_stats, trait, version):
    """Calculate y-axis limits for each trait"""
    if version == 'log_log':
        mean_min, mean_max = 1.5, 5.5
        var_min, var_max = 1e-2, 1e0
    elif version == 'linear_log':
        mean_min, mean_max = 1.0, 5.0
        var_min, var_max = 1e-2, 1e0
    else:  # linear_linear
        mean_min, mean_max = 1.0, 5.0
        var_min, var_max = 0.0, 0.8
        
    return mean_min, mean_max, var_min, var_max

def plot_scaling_versions(mean_stats, var_stats, output_path, output_prefix):
    plt.rcParams.update({
        'figure.figsize': [20, 8],
        'figure.dpi': 300,
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.linewidth': 1,
        'axes.grid': True,
        'grid.alpha': 0.15,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
    })
    
    # Define colors and labels for configurations
    config_colors = {
        'shuffle': '#4169E1',    # Royal Blue
        'paraphrase': '#8B0000'      # Dark Red
    }

    config_labels = {
        'shuffle': 'Shuffling (Batch size 10)',
        'paraphrase': 'Paraphrasing (Batch size 10)'
    }
    
    traits = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
    versions = [
        ('linear_log', False, True),
        ('log_log', True, True),
        ('linear_linear', False, False)
    ]
    
    # # Define stagger offsets for x-coordinates
    # stagger_offsets = {
    #     'b1_hist': -15,
    #     'b2_hist': -5,
    #     'b10_hist': 5,
    #     'b10_no_hist': 15
    # }
    minor_ticks = [2,3,4,5,6,7,8,9,20,30,40,50,60,70,80,90,200,300,400]
    for version_name, top_log, bottom_log in versions:
        fig = plt.figure()
        gs = fig.add_gridspec(2, 5, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
        axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(5)] for i in range(2)])
        
        for col, trait in enumerate(traits):
            mean_min, mean_max, var_min, var_max = get_trait_limits(
                mean_stats, var_stats, trait, version_name)
            
            # Plot means (top row)
            ax_mean = axes[0, col]
            for config in sorted(config_colors.keys()):
                config_data = mean_stats[
                    (mean_stats['config'] == config) & 
                    (mean_stats['trait'] == trait)
                ].sort_values('model_size')
                
                if not config_data.empty:
                    # Apply staggering to x-coordinates
                    x_coords = config_data['model_size'] #+ stagger_offsets[config]
                    
                    # Plot variance shading
                    ax_mean.fill_between(
                        x_coords,
                        config_data['mean'] - config_data['std'],
                        config_data['mean'] + config_data['std'],
                        color=config_colors[config], alpha=0.03
                    )
                    
                    ax_mean.plot(x_coords, config_data['mean'],
                               'o-', color=config_colors[config], alpha=0.6,
                               label=config_labels[config] if col == 0 else "",
                               markersize=5, linewidth=1.5)
            
            # Plot variances (bottom row)
            ax_var = axes[1, col]
            for config in sorted(config_colors.keys()):
                config_data = var_stats[
                    (var_stats['config'] == config) & 
                    (var_stats['trait'] == trait)
                ].sort_values('model_size')
                
                if not config_data.empty:
                    # Apply same staggering
                    x_coords = config_data['model_size'] #+ stagger_offsets[config]
                    
                    # Plot confidence interval
                    ax_var.fill_between(
                        x_coords,
                        config_data['var_ci_lower'],
                        config_data['var_ci_upper'],
                        color=config_colors[config], alpha=0.03
                    )
                    
                    ax_var.plot(x_coords, config_data['var'],
                              'o-', color=config_colors[config], alpha=0.6,
                              markersize=5, linewidth=1.5)
            
            # Customize axes
            for ax in [ax_mean, ax_var]:
                ax.set_xscale('log')
                ax.grid(True, which='both', alpha=0.15)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                ax.set_xticks([1, 10, 100])
                ax.set_xticks(minor_ticks, minor=True)
                ax.tick_params(which='minor', length=2)
                ax.tick_params(which='major', length=4, labelsize=11)
                
                if ax == ax_var:
                    ax.set_xlabel('Model Size (B)', fontsize=12)
            
            # Set y-scales and limits
            if top_log:
                ax_mean.set_yscale('log')
            if bottom_log:
                ax_var.set_yscale('log')
            
            ax_mean.set_ylim(mean_min, mean_max)
            ax_var.set_ylim(var_min, var_max)
            
            # Titles and labels
            ax_mean.set_title(trait, pad=10, fontweight='bold', fontsize=13)
            if col == 0:
                ax_mean.set_ylabel("Trait's Score MEAN", fontweight='bold')
                ax_var.set_ylabel("Trait's Score VARIANCE", fontweight='bold')
        
        # Add legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        title_text = 'Configurations'
        legend = fig.legend(handles, labels,
                          loc='upper center',
                          bbox_to_anchor=(0.5, 1.02),
                          ncol=4,
                          frameon=True,
                          fontsize=12,
                          columnspacing=1,
                          handletextpad=0.5,
                          title=title_text,
                          title_fontsize=12,
                          borderpad=0.5,
                          edgecolor='black')
        
        legend.get_title().set_fontweight('bold')
        
        # Save figures
        for fmt in ['pdf', 'png']:
            # fig.savefig(f'{output_prefix}_{version_name}.{fmt}', 
            #            bbox_inches='tight', dpi=300)
            save_path = f'{output_path}/{output_prefix}_{version_name}.{fmt}'
            plt.savefig(save_path, format=fmt, bbox_inches='tight', dpi=300)
            print(f"Saved plot as {save_path}")
        plt.close()


def prepare_data(df):
    """Prepare and clean the data for plotting"""
    # Filter out random and specific models
    df = df[df['model_type'] != 'random']
    
    try:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        # Extract numeric size from model_size column
        df['model_size'] = df['model_size'].str.extract('(\d+)').astype(int)
        df['model_family'] = df['model'].apply(lambda x: next(
            (family for family in ['llama', 'qwen', 'gemma','hist'] if family in x.lower()),
            'other'
        ))
        
        # Create a unique configuration identifier
        df['config'] = df["variability"]
        
    except Exception as e:
        print(f"Error in data preparation: {e}")
        raise
    
    return df


def main():
    # Load and prepare data
    df = pd.read_csv('/Users/tommasotosato/Desktop/outputs/bfi_data.csv')
    assistant_df = prepare_data(df)
    
    # Filter and prepare data
    assistant_df = assistant_df[assistant_df['persona'] == 'assistant'].copy()
    
    # Calculate statistics
    mean_stats = assistant_df.groupby(
        ['model', 'config', 'model_size', 'trait']
    )['score'].agg(['mean', 'std', 'count']).reset_index()
    
    var_stats = assistant_df.groupby(
        ['model', 'config', 'model_size', 'trait']
    )['score'].agg([('var', 'var'), 'count']).reset_index()
    
    var_stats['var_ci_lower'] = var_stats.apply(lambda row: 
        (row['count'] - 1) * row['var'] / stats.chi2.ppf(0.975, row['count'] - 1), axis=1)
    var_stats['var_ci_upper'] = var_stats.apply(lambda row: 
        (row['count'] - 1) * row['var'] / stats.chi2.ppf(0.025, row['count'] - 1), axis=1)
    
    # Create plots
    plot_scaling_versions(mean_stats, var_stats, 'figure2')

if __name__ == "__main__":
    main()