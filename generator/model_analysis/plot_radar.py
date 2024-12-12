import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import CubicSpline
from generator.config import (
    get_cached_model_color,
    sort_models_by_size
)

# Define persona groupings and their color schemes
PERSONA_GROUPS = {
    'supportive': {
        'personas': ['assistant', 'buddhist', 'teacher'],
        'colors': ['#d73027', '#fc8d59', '#fee090']  # Warm oranges/reds
    },
    'clinical': {
        'personas': ['antisocial', 'anxiety', 'depression', 'schizophrenia'],
        'colors': ['#4575b4', '#74add1', '#abd9e9', '#e0f3f8']  # Cool blues
    }
}

def get_persona_color(persona):
    """Get the color for a specific persona based on its group"""
    for group_info in PERSONA_GROUPS.values():
        if persona in group_info['personas']:
            color_idx = group_info['personas'].index(persona)
            return group_info['colors'][color_idx]
    return '#808080'  # Default gray for any unmatched personas

def interpolate_circular(angles, values, n_points):
    """Helper function to create smooth interpolation for circular data"""
    angles_ext = np.concatenate([angles[-2:] - 2*np.pi, angles, angles[:2] + 2*np.pi])
    values_ext = np.concatenate([values[-2:], values, values[:2]])
    theta_interp = np.linspace(0, 2*np.pi, n_points)
    cs = CubicSpline(angles_ext, values_ext)
    return theta_interp, cs(theta_interp)

def plot_radar_subplot(data, ax, value_column, family):
    """Plot function for a single facet showing different personas using only mean values"""
    # Group by persona and trait
    grouped = data.groupby(['persona', 'trait'])[value_column].mean().reset_index()
    
    # Get unique traits
    traits = sorted(data['trait'].unique())
    
    # Set up angles for the radar plot
    angles = np.linspace(0, 2*np.pi, len(traits), endpoint=False)
    
    # Get all personas and sort them by group and alphabetically within groups
    all_personas = []
    for group_info in PERSONA_GROUPS.values():
        group_personas = sorted(set(group_info['personas']).intersection(set(data['persona'].unique())))
        all_personas.extend(group_personas)
    
    handles = []
    labels = []
    
    for persona in all_personas:
        persona_data = grouped[grouped['persona'] == persona]
        
        # Get means for the current persona
        means = persona_data.set_index('trait')[value_column].reindex(traits).values
        
        # Create smooth interpolation
        theta_interp, smooth_values = interpolate_circular(angles, means, 200)
        
        # Plot smooth curve with persona-specific color
        color = get_persona_color(persona)
        line = ax.plot(theta_interp, smooth_values, linewidth=2, color=color, 
                      label=persona, zorder=2, alpha=0.5)
        
        # Add points for the means
        ax.scatter(angles, means, color=color, s=50, zorder=3,
                  edgecolor='white', linewidth=0.5, alpha=0.4)
        
        handles.append(line[0])
        labels.append(persona)
    
    # Customize the subplot
    ax.set_xticks(angles)
    ax.set_xticklabels(traits, fontsize=8)
    
    # Add gridlines
    ax.grid(True, alpha=0.3)
    
    # Customize grid circles - adjust based on your score range
    max_score = max(grouped[value_column].max(), 1.0)  # Ensure at least 1.0 for percentage
    grid_values = np.arange(0, max_score + 1, 1)  # Integer steps
    ax.set_ylim(0, max_score * 1.1)  # Add 10% padding
    ax.set_rticks(grid_values)
    ax.set_rgrids(grid_values, labels=[f'{v:.2f}' for v in grid_values], fontsize=8)
    
    return handles, labels

def create_faceted_radar_plot(data, value_column, title, output_file):
    # Get unique families
    families = ['llama', 'qwen', 'gemma']
    family_data = {
        family: data[data['model'].str.lower().str.contains(family)]
        for family in families
        if any(data['model'].str.lower().str.contains(family))
    }
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Create subplots grid
    legend_handles = None
    legend_labels = None
    
    for i, (family, fam_data) in enumerate(family_data.items()):
        # Create polar subplot
        ax = fig.add_subplot(1, 3, i + 1, projection='polar')
        
        # Plot radar and get legend info
        handles, labels = plot_radar_subplot(fam_data, ax, value_column, family)
        
        # Store legend info from first subplot
        if i == 0:
            legend_handles = handles
            legend_labels = labels
        
        # Set subplot title
        ax.set_title(f'{family.capitalize()} Models', pad=20)
    
    # Add overall title
    fig.suptitle(title, size=16, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add single legend for personas
    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels,
                  loc='center right', bbox_to_anchor=(1.15, 0.5),
                  title="Personas")
    
    # Save the plot
    plt.savefig(f'{output_file}.png', format='png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_file}.svg', format='svg', bbox_inches='tight', dpi=300)
    plt.close()

def main(input_file, output_path):
    # Read the CSV file
    data = pd.read_csv(input_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create radar plot
    output_file = os.path.join(output_path, 'faceted_radar_plot_by_family_persona')
    create_faceted_radar_plot(
        data, 
        'score', 
        'Model Family Performance Analysis', 
        output_file
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create faceted radar plots from survey data")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("-o", "--output", default=".", 
                       help="Path to the output directory (default: current directory)")
    args = parser.parse_args()
    
    main(args.input_file, args.output)
