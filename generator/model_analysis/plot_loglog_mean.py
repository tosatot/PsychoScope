import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from generator.config import (
    get_cached_model_color,
    get_size_in_billions,
    order_dict
)
import json

def load_human_baselines(questionnaire_file):
    """Load human baseline data from questionnaire file"""
    with open(questionnaire_file) as f:
        data = json.load(f)
        questionnaires = data['questionnaires']
        
        human_baselines = {}
        # Find both BFI and EPQ-R questionnaires
        for q in questionnaires:
            if q['name'] in ['BFI', 'EPQ-R']:
                # Store baselines for each trait
                human_baselines[q['name']] = {
                    cat['cat_name']: cat['crowd'] 
                    for cat in q['categories']
                }
        return human_baselines

def prepare_data(df):
    """Prepare and clean the data for plotting"""
    # Filter out random and specific models
    df = df[df['model_type'] != 'random']

    
    try:
        # Convert score to numeric, replacing invalid values with NaN
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        
        # Extract model sizes and convert to numeric
        df['model_size'] = df['model'].apply(lambda x: get_size_in_billions(x))
        
        # Add model family column
        df['model_family'] = df['model'].apply(lambda x: next(
            (family for family in ['llama', 'qwen', 'gemma','hist'] if family in x.lower()),
            'other'
        ))
        
    except Exception as e:
        print(f"Error in data preparation: {e}")
        raise
    
    return df

def calculate_ci(mean, std, n, confidence=0.95):
    """Calculate confidence interval"""
    ci = stats.t.ppf((1 + confidence) / 2, n - 1) * std / np.sqrt(n)
    return ci


def plot_scaling_behavior(df, output_path, human_baselines):
    """Create scaling plots with trend lines per model family"""
    # Set up plot style
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.2  # Reduced grid visibility
    plt.rcParams['axes.linewidth'] = 0.5  # Thin spines
    
    # Get unique traits and personas
    traits = sorted(df['trait'].unique())
    all_personas = order_dict['persona']
    # Filter to only include personas that are in the data
    personas = [p for p in all_personas if p in df['persona'].unique()]
    
    # Calculate global y-axis limits for each trait
    y_limits = {}
    for trait in traits:
        trait_data = df[df['trait'] == trait]['score']
        y_min = trait_data.min()
        y_max = trait_data.max()
        # Add padding
        padding = (y_max - y_min) * 0.1
        y_limits[trait] = (y_min - padding, y_max + padding)
    
    # Create figure and axes with adjusted spacing
    fig, axes = plt.subplots(len(personas), len(traits), 
                            figsize=(6*len(traits), 6*len(personas)),
                            squeeze=False)
    
    # Adjust subplot spacing
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9,
                       wspace=0.3, hspace=0.2)
    
    # Color scheme for model families
    family_colors = {
        'llama': '#4169E1',  # Royal Blue
        'qwen': '#228B22',   # Forest Green
        'gemma': '#8B0000',   # Dark Red
        'hist': '#9932CC'    # Dark Orchid - adding hist model
    }
    
    # Minor tick locations for log scale
    minor_ticks = [2,3,4,5,6,7,8,9,20,30,40,50,60,70,80,90,200,300,400]
    
    # Plot each subplot
    for i, persona in enumerate(personas):
        for j, trait in enumerate(traits):
            ax = axes[i, j]
            
            # Add thin border around subplot
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            
            # Filter data for current persona and trait
            mask = (df['persona'] == persona) & (df['trait'] == trait)
            plot_data = df[mask].copy()
            
            try:
                # Plot points and trend lines for each family
                for family, color in family_colors.items():
                    family_data = plot_data[plot_data['model_family'] == family]
                    if not family_data.empty:
                        # Plot individual points with more transparency
                        ax.scatter(family_data['model_size'], family_data['score'],
                                 alpha=0.2, color=color, s=30)
                        
                        # Compute trend line
                        if len(family_data) > 1:
                            # Calculate mean and std for each unique model size
                            stats_data = family_data.groupby('model_size')['score'].agg(['mean', 'std', 'count'])
                            
                            # Calculate confidence intervals
                            stats_data['ci'] = stats_data.apply(
                                lambda row: calculate_ci(row['mean'], row['std'], row['count']), 
                                axis=1
                            )
                            
                            # Plot confidence intervals
                            ax.fill_between(stats_data.index,
                                          stats_data['mean'] - stats_data['ci'],
                                          stats_data['mean'] + stats_data['ci'],
                                          color=color, alpha=0.1)
                            
                            # Plot mean points
                            ax.scatter(stats_data.index, stats_data['mean'],
                                     color=color, s=100, zorder=3,
                                     label=f'{family.capitalize()}')
                            
                            # Plot error bars with increased visibility
                            ax.errorbar(stats_data.index, stats_data['mean'],
                                      yerr=stats_data['std'],
                                      color=color, linewidth=1,
                                      capsize=7, capthick=1,
                                      alpha=0.7)
                            
                            # Connect means with thicker, semi-transparent lines
                            ax.plot(stats_data.index, stats_data['mean'],
                                  color=color, linewidth=3, alpha=0.7)
                
                       
                            # Add human baselines
                            questionnaire_name = 'BFI' if 'Openness' in traits else 'EPQ-R'
                            if questionnaire_name in human_baselines:
                                if trait in human_baselines[questionnaire_name]:
                                    baseline_data = human_baselines[questionnaire_name][trait]
                                    # For EPQ-R, average male and female baselines
                                    means = []
                                    stds = []
                                    for crowd_data in baseline_data:
                                        means.append(crowd_data['mean'])
                                        stds.append(crowd_data['std'])
                                    mean = np.mean(means)
                                    std = np.mean(stds)  # Using mean of standard deviations
                                    
                                    ax.axhline(y=mean, color='gray', linestyle='--', alpha=0.5)
                                    ax.fill_between([ax.get_xlim()[0], ax.get_xlim()[1]],
                                                mean - std, mean + std,
                                                color='lightgray', alpha=0.1)

                
                # Set log scale and ticks for x-axis
                ax.set_xscale('log')
                ax.set_xticks([1, 10, 100])
                ax.set_xticks(minor_ticks, minor=True)
                ax.tick_params(which='minor', length=2)  # Smaller minor ticks
                ax.tick_params(which='major', length=4)  # Consistent major ticks
                
                # Add horizontal gridlines
                ax.grid(True, axis='y', alpha=0.2)
                ax.grid(True, axis='y', which='minor', alpha=0.1)
                
                # Set consistent y-axis limits for each trait
                ax.set_ylim(y_limits[trait])
                
                # Format axis labels
                if i == len(personas)-1:
                    ax.set_xlabel('Model Size (B)', fontsize=12)
                if j == 0:
                    ax.set_ylabel('Score', fontsize=12)
                    # Adjust y-axis label position to prevent overlap
                    ax.yaxis.set_label_coords(-0.1, 0.5)
                
                # Set title for top row only with increased prominence
                if i == 0:
                    ax.set_title(trait, fontsize=20, pad=15, fontweight='bold')
                
                # Customize tick labels
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))  # One decimal place
                
            except Exception as e:
                print(f"Error plotting for persona {persona}, trait {trait}: {e}")
                continue
    
        
    for i, persona in enumerate(personas):
        # Get the leftmost subplot for this row
        ax = axes[i, 0]
        
        # Add persona label using subplot coordinates
        ax.text(-0.3, 0.5, persona,
                transform=ax.transAxes,  # This is key - uses subplot coordinates
                fontsize=20,
                va='center',
                ha='right',
                rotation=90,
                fontweight='bold')

    # And adjust the subplot adjustment parameters:
    plt.subplots_adjust(left=0.1,  
                    right=0.85, 
                    bottom=0.1, 
                    top=0.9,
                    wspace=0.2, 
                    hspace=0.2)
    
    # Add legend with adjusted position and prominence
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
              loc='center right',
              bbox_to_anchor=(0.95, 0.5),  # Moved left
              title="Model Families",
              title_fontsize=14,  # Increased title size
              fontsize=14,
              frameon=True,  # Add frame
              edgecolor='black',
              borderpad=1)
    
    # Save plots
    for fmt in ['svg', 'png']:
        save_path = f'{output_path}/scaling_behavior.{fmt}'
        plt.savefig(save_path, format=fmt, bbox_inches='tight', dpi=300)
        print(f"Saved plot as {save_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Create scaling behavior plots")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("output_dir", help="Path to output directory")
    parser.add_argument("--questionnaire", default="questionnaires.json", help="Path to questionnaire JSON file")
    args = parser.parse_args()
    
    try:
        # Load human baselines
        human_baselines = load_human_baselines(args.questionnaire)
        
        # Read and prepare data
        print("Reading data...")
        df = pd.read_csv(args.input_file)
        
        print("Preparing data...")
        df = prepare_data(df)
        
        # Create plots
        print("Creating plots...")
        plot_scaling_behavior(df, args.output_dir, human_baselines)
        print("Plotting complete")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()




# Run the script with the following command:
# python plot_loglog_mean.py /Users/tommasotosato/Desktop/PsychoBenchPPSP/results_output_figs/bfi_data_formatted.csv /Users/tommasotosato/Desktop/PsychoBenchPPSP/results_output_figs
# python plot_loglog_mean.py /Users/tommasotosato/Desktop/PsychoBenchPPSP/results_output_figs/epq-r_data_formatted.csv /Users/tommasotosato/Desktop/PsychoBenchPPSP/results_output_figs