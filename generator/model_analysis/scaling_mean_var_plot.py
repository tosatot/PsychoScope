import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import os



def get_trait_limits(mean_stats, var_stats, trait, version):
    """Calculate y-axis limits for each trait"""
    if version == 'log_log':
        # Extended range for means (top row) in log scale
        mean_min, mean_max = 1.5, 5.5  # Further extended range
        var_min, var_max = 1e-4, 1e1
    elif version == 'linear_log':
        mean_min, mean_max = 1.0, 5.0
        var_min, var_max = 1e-3, 1e0
    else:  # linear_linear
        mean_min, mean_max = 1.0, 5.0
        var_min, var_max = 0.0, 0.35
        
    return mean_min, mean_max, var_min, var_max

def plot_scaling_versions(mean_stats, var_stats, q_traits, output_path, output_prefix):
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
    
    family_colors = {
        'llama': '#4169E1',
        'qwen': '#228B22',
        'gemma': '#8B0000'
    }
    
    traits = q_traits
    versions = [
        ('linear_log', False, True),
        ('log_log', True, True),
        ('linear_linear', False, False)
    ]
    
    minor_ticks = [2,3,4,5,6,7,8,9,20,30,40,50,60,70,80,90,200,300,400]
    
    for version_name, top_log, bottom_log in versions:
        fig = plt.figure()
        gs = fig.add_gridspec(2, len(traits), height_ratios=[1, 1], hspace=0.3, wspace=0.3)
        axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(len(traits))] for i in range(2)])
        
        for col, trait in enumerate(traits):
            mean_min, mean_max, var_min, var_max = get_trait_limits(
                mean_stats, var_stats, trait, version_name)
            
            # Plot means (top row)
            ax_mean = axes[0, col]
            for family in family_colors:
                family_data = mean_stats[
                    (mean_stats['model_family'] == family) & 
                    (mean_stats['trait'] == trait)
                ].sort_values('model_size')
                
                if not family_data.empty:
                    # Plot variance shading with very low alpha
                    ax_mean.fill_between(
                        family_data['model_size'],
                        family_data['mean'] - family_data['std'],  # Full standard deviation
                        family_data['mean'] + family_data['std'],
                        color=family_colors[family], alpha=0.03  # Very low alpha
                    )
                    
                    # # Plot confidence interval
                    # ax_mean.fill_between(
                    #     family_data['model_size'],
                    #     family_data['mean'] - family_data['std'] / np.sqrt(family_data['count']),
                    #     family_data['mean'] + family_data['std'] / np.sqrt(family_data['count']),
                    #     color=family_colors[family], alpha=0.08
                    # )
                    
                    ax_mean.plot(family_data['model_size'], family_data['mean'],
                               'o-', color=family_colors[family], alpha=0.6,
                               label=family.capitalize() if col == 0 else "",
                               markersize=5, linewidth=1.5)
            
            # Plot variances (bottom row)
            ax_var = axes[1, col]
            for family in family_colors:
                family_data = var_stats[
                    (var_stats['model_family'] == family) & 
                    (var_stats['trait'] == trait)
                ].sort_values('model_size')
                
                if not family_data.empty:
                    
                    # Plot confidence interval
                    ax_var.fill_between(
                        family_data['model_size'],
                        family_data['var_ci_lower'],
                        family_data['var_ci_upper'],
                        color=family_colors[family], alpha=0.03
                    )
                    
                    ax_var.plot(family_data['model_size'], family_data['var'],
                              'o-', color=family_colors[family], alpha=0.6,
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
        
        # Add legend and title in a single line
        handles, labels = axes[0, 0].get_legend_handles_labels()
        # Create two-part legend
        title_text = output_prefix + ' Persona'
        legend = fig.legend(handles, labels,
                          loc='upper center',
                          bbox_to_anchor=(0.5, 1.02),
                          ncol=3,
                          frameon=True,  # Add box around legend
                          fontsize=12,
                          columnspacing=1,
                          handletextpad=0.5,
                          title=title_text,
                          title_fontsize=12,
                          borderpad=0.5,
                          edgecolor='black')
        
        legend.get_title().set_fontweight('bold')
        
        for fmt in ['pdf', 'png']:
            # fig.savefig(f'{output_prefix}_{version_name}.{fmt}', 
            #            bbox_inches='tight', dpi=300)
            save_path = f'{output_path}/{output_prefix}_{version_name}.{fmt}'
            plt.savefig(save_path, format=fmt, bbox_inches='tight', dpi=300)
            print(f"Saved plot as {save_path}")
        plt.close()

def main():
    # Load and prepare data
    df = pd.read_csv('bfi_data.csv')
    for persona in df.persona.unique().tolist():
        # Filter and prepare data
        assistant_df = df[df['persona'] == persona].copy()
        assistant_df.loc[:, 'model_size'] = assistant_df['model'].apply(
            lambda x: float(x.split('-')[1].rstrip('b')))
        assistant_df.loc[:, 'model_family'] = assistant_df['model'].apply(
            lambda x: next((family for family in ['llama', 'qwen', 'gemma'] 
                        if family in x.lower()), 'other'))
        
        # Calculate statistics
        mean_stats = assistant_df.groupby(
            ['model', 'model_family', 'model_size', 'trait']
        )['score'].agg(['mean', 'std', 'count']).reset_index()
        
        var_stats = assistant_df.groupby(
            ['model', 'model_family', 'model_size', 'trait']
        )['score'].agg([('var', 'var'), 'count']).reset_index()
        
        var_stats['var_ci_lower'] = var_stats.apply(lambda row: 
            (row['count'] - 1) * row['var'] / stats.chi2.ppf(0.975, row['count'] - 1), axis=1)
        var_stats['var_ci_upper'] = var_stats.apply(lambda row: 
            (row['count'] - 1) * row['var'] / stats.chi2.ppf(0.025, row['count'] - 1), axis=1)
    
        # Create plots
        plot_scaling_versions(mean_stats, var_stats,  df.trait.unique().tolist(), "/path/plots", persona)

if __name__ == "__main__":
    main()