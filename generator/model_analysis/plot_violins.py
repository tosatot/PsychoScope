import argparse
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
from generator.utils import get_questionnaire, convert_data
import math
import matplotlib.colors as mcolors
import colorsys
from generator.config import order_dict, get_model_color
from generator.model_analysis.prepare_data import prepare_data 

logger = logging.getLogger(__name__)

def plot_data(df, questionnaire_name, out):

    logger.info(f"Shape of DataFrame at start of plot_data: {df.shape}")
    logger.info(f"Unique models: {df['model'].unique()}")
    logger.info(f"Unique personas: {df['persona'].unique()}")

    questionnaire = get_questionnaire(questionnaire_name)
    categories = [cat['cat_name'] for cat in questionnaire['categories']]
    
    def get_ordered_uniques(column, order_list):
        uniques = df[column].unique()
        return sorted(uniques, key=lambda x: order_list.index(x.rstrip('b')) if x.rstrip('b') in order_list else len(order_list))

    all_personas = get_ordered_uniques('persona', order_dict['persona'])
    all_models = get_ordered_uniques('model', order_dict['model'])
    
    logger.info(f"Length of all_models: {len(all_models)}")
    logger.info(f"Length of all_personas: {len(all_personas)}")
    
    # Calculate height based on number of models
    base_height = 16
    height_per_model = 2  # Adjust this value to increase/decrease height per model
    total_height = max(base_height, len(all_models) * height_per_model)

    # Calculate figure size
    n_cols = len(categories)
    n_rows = len(all_personas)
    fig_width = total_height * 1.4 * n_cols  # aspect ratio is 1.5
    fig_height = total_height * n_rows

    plt.figure(figsize=(fig_width, fig_height))

    # Calculate font sizes
    base_font_size = 72
    font_scale = max(0.5, 1 - (len(all_models) - 10) * 0.05)  # Reduce font size as models increase
    adjusted_font_size = int(base_font_size * font_scale)

    models_abbr = []
    for x in all_models:
        if x.startswith('llama'):
            abbr = f"L-{x.split('-')[1].rstrip('b')}"
        elif x.startswith('qwen'):
            abbr = f"Q-{x.split('-')[1].rstrip('bh')}"
        elif x.startswith('gemma'):
            abbr = f"G-{x.split('-')[1].rstrip('bh')}"
        elif x.startswith('hist'):
            abbr = f"H-{x.split('-')[1].rstrip('bh')}"
        else:
            abbr = x
        models_abbr.append(abbr)

    if "EPQ" in questionnaire_name:
        xlim = (0, 30)
        xs = np.arange(xlim[0], xlim[1], 6)
    else:
        xlim = (1, 6)
        xs = np.arange(xlim[0], xlim[1])

    g = sns.FacetGrid(df, row='persona', col='trait', height=total_height, aspect=1.5, 
                      col_order=categories, row_order=all_personas, xlim=xlim, margin_titles=True)

    # Generate colors for each model
    model_colors = {}
    for model in all_models:
        model_name = model.split('-')[0]
        model_size = float(model.split('-')[1].rstrip('b'))
        model_colors[model] = get_model_color(model_name, model_size)

    # Use the custom color palette in the plot
    g = g.map_dataframe(pt.RainCloud, x="model", y="score", data=df, order=all_models,
                        orient="h", width_viol=1.0, width_box=0.3, scale="area", alpha=0.75,
                        palette=model_colors, dodge=True, bw="silverman")


    # Add human performance data
    humans = {}
    for trait in questionnaire['categories']:
        crowd_list = [(c["crowd_name"], c["n"]) for c in trait["crowd"]]
        mean = sum(trait['crowd'][index]['mean'] for index, _ in enumerate(crowd_list)) / len(crowd_list)
        std = sum(trait['crowd'][index]['std'] for index, _ in enumerate(crowd_list)) / len(crowd_list)
        humans[trait['cat_name']] = {'mean': mean, 'std': std}

    # Customize each subplot
    for idx, ax in enumerate(g.axes.flat):
        def add_mean_and_stdev(trait, **kwargs):
            ax.axvline(humans[trait]['mean'], color='gray', linestyle='dashed')
            ax.axvspan(xmax=humans[trait]['mean']+humans[trait]['std'], 
                       xmin=humans[trait]['mean']-humans[trait]['std'], 
                       color='lightgray', linestyle='dotted', alpha=0.5, zorder=0)
        
        trait_name = categories[idx % len(categories)] if categories else ""
        
        if idx < len(categories):
            ax.set_title(trait_name, fontsize=adjusted_font_size, fontweight='bold', fontfamily='sans-serif')
        else:
            ax.set_title('')
        
        ax.set_xticks(ticks=xs, labels=xs, fontsize=adjusted_font_size*0.7)
        if trait_name in humans:
            add_mean_and_stdev(trait_name)
        ax.spines['left'].set_linewidth(4)
        ax.spines['bottom'].set_linewidth(4)
        
        num_cols = len(categories)
        if idx % num_cols == 0:
            ax.set_yticks(ticks=np.arange(len(all_models)), labels=models_abbr, fontsize=adjusted_font_size*0.8, va='center')
            persona_name = all_personas[idx // num_cols].capitalize()
            ax.text(-0.15, 0.5, persona_name, transform=ax.transAxes, fontsize=adjusted_font_size, 
                    va='center', ha='right', rotation=90, fontweight='bold')

    # g.fig.subplots_adjust(top=0.95, bottom=0.05, left=0.15, right=0.95, hspace=0.3, wspace=0.1)
    g.fig.tight_layout(rect=[0.05, 0.03, 1, 0.95])
    g.fig.subplots_adjust(top=0.95, bottom=0.05, left=0.15, right=0.98, hspace=0.3, wspace=0.2)
    plt.suptitle(f'{questionnaire_name} Scores by Persona and Model', fontsize=adjusted_font_size*1.4, y=0.98)

   # Save as SVG
    svg_path = os.path.join(out, f'combined_raincloud_plot_{questionnaire_name}.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
    logger.info(f"Saved plot as SVG: {svg_path}")

    # Save as PDF
    pdf_path = os.path.join(out, f'combined_raincloud_plot_{questionnaire_name}.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    logger.info(f"Saved plot as PDF: {pdf_path}")
    

    plt.close(g.fig)

    logger.info("Plotting completed.")
    
def main(args):
    # Set up logging
    logging.basicConfig(filename='plot_violins.log', encoding='utf-8', level=args.verbosity)

    directory = os.path.abspath(args.data)
    outdir = os.path.abspath(args.out)
    questionnaire_name = args.questionnaire

    logger.info(f"Input directory: {directory}")
    logger.info(f"Output directory: {outdir}")
    logger.info(f"Processing questionnaire: {questionnaire_name}")

    csv_file = os.path.join(outdir, f'{questionnaire_name.lower()}_data_formatted.csv')
    
    if args.prepare:
        df = prepare_data(directory, questionnaire_name)
        if df.empty:
            logger.error("Prepared DataFrame is empty. Check your input data.")
            return
        df.to_csv(csv_file, index=False)
        logger.info(f"Prepared data saved to {csv_file}")
    else:
        if not os.path.exists(csv_file):
            logger.error(f"CSV file not found: {csv_file}. Run with --prepare first.")
            return
        df = pd.read_csv(csv_file)
        if df.empty:
            logger.error(f"DataFrame read from {csv_file} is empty. Check your CSV file.")
            return

    logger.info(f"DataFrame shape after reading: {df.shape}")
    logger.info(f"DataFrame columns: {df.columns}")

    plot_data(df, questionnaire_name, outdir)

    logger.info("Script execution completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to generate a grid of violin plots.')
    parser.add_argument('-p', '--prepare', action='store_true', help='Run the data preparation step')
    parser.add_argument('-d', '--data', required=True, help='Directory containing CSV data files')
    parser.add_argument('-o', '--out', required=True, help='Output directory for results')
    parser.add_argument('-q', '--questionnaire', required=True, help='Questionnaire to process (e.g., BFI, EPQ-R)')
    parser.add_argument('-v', '--verbosity', help='Verbosity levels: DEBUG, INFO, WARNING, ERROR', default='INFO')
    args = parser.parse_args()
    main(args)


# Run the script with the following command: 
# python plot_violins.py -p -d '/Users/tommasotosato/Library/CloudStorage/GoogleDrive-tommie.tosato@gmail.com/My Drive/all_model_outputs/allRuns_BFI_shuffle_noHist' -o  /Users/tommasotosato/Desktop/PsychoBenchPPSP/results_output_figs -q BFI
# python plot_violins.py -p -d '/Users/tommasotosato/Library/CloudStorage/GoogleDrive-tommie.tosato@gmail.com/My Drive/all_model_outputs/allRuns_EPQ-R_shuffle_noHist' -o  /Users/tommasotosato/Desktop/PsychoBenchPPSP/results_output_figs -q EPQ-R
