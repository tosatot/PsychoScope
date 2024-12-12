import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd



def plot_questions_kde(plot_data, output_path, dict_traits=None):

    if not dict_traits:
        dict_traits = {
            1: 'Extraversion',
            2: 'Agreeableness',
            3: 'Conscientiousness',
            4: 'Neuroticism',
            5: 'Openness'
        }

    # Set style parameters
    plt.style.use('default')
    colors = plt.cm.viridis(np.linspace(0, 1, len(grouped)))

    # Create figure and axes
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    plt.subplots_adjust(hspace=0.4)

    # Questions to plot (first 5)
    questions = sorted(plot_data['question_number'].unique())[:5]

    # Configurations to plot
    configs = [
        {'b_size': 1, 'history': 1, 'label': 'Batch 1 with history'},
        {'b_size': 10, 'history': 1, 'label': 'Batch 10 with history'}
    ]

    # Define fixed bins for histogram
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]  # This ensures each score gets its own bin

    for row, config in enumerate(configs):
        for col, q_num in enumerate(questions):
            ax = axes[row, col]
            
            # Filter data
            mask = (
                (plot_data['b_size'] == config['b_size']) & 
                (plot_data['history'] == config['history']) & 
                (plot_data['question_number'] == q_num) & 
                (plot_data.persona == 'assistant')
            )
            data = plot_data[mask]
            
            # Get question text
            q_text = data['question_text'].iloc[0] + f"  (Trait: {dict_traits[q_num]})" if not data.empty else f"Question {q_num}"
            
            # Plot histogram and KDE for each model size
            grouped = data.groupby('model_size')
            
            for (size, group), color in zip(grouped, colors):
                # Histogram with fixed bins
                sns.histplot(
                    data=group, 
                    x='score', 
                    stat='density',
                    alpha=0.2,  # Reduced alpha
                    color=color,
                    label=f'{size}B Hist',
                    bins=bins,  # Use fixed bins
                    ax=ax
                )
                # KDE with increased linewidth
                sns.kdeplot(
                    data=group,
                    x='score',
                    color=color,
                    linewidth=2.5,  # Increased linewidth
                    label=f'{size}B KDE',
                    ax=ax
                )
            
            # Customize subplot
            ax.set_xlabel('Score' if row == 1 else '')
            ax.set_ylabel('Density' if col == 0 else '')
            ax.set_xlim(0.5, 5.5)
            
            # Fix x-ticks to show only integer scores
            ax.set_xticks([1, 2, 3, 4, 5])
            
            # Title
            if row == 0:
                ax.set_title(f"Q{q_num}: {q_text[:50]}...\n" if len(q_text) > 50 else f"Q{q_num}: {q_text}\n",
                            fontsize=10, pad=10)
            
            # Configuration label
            if col == 0:
                ax.text(-0.3, 0.5, config['label'],
                    transform=ax.transAxes,
                    rotation=90,
                    verticalalignment='center')
            
            # Legend
            if col == 4:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            else:
                ax.legend().remove()

    # Overall title
    fig.suptitle('Response Distribution by Question and Configuration',
                fontsize=16, y=1.02)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()




def prepare_data(df):
    """Prepare and clean the data for plotting"""
    # Create config column
    df['config'] = df.apply(lambda x: f"b{x['b_size']}_{'hist' if x['history'] == 1 else 'no_hist'}", axis=1)
    
    # Create readable config labels
    config_labels = {
        'b1_hist': 'Batch 1 (with history)',
        'b2_hist': 'Batch 2 (with history)',
        'b10_hist': 'Batch 10 (with history)',
        'b10_no_hist': 'Batch 10 (no history)'
    }
    df['config_label'] = df['config'].map(config_labels)
    
    return df


def main():
    # Load and prepare data
    parser = argparse.ArgumentParser(description="Create variance scaling behavior plots")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("output_dir", help="Path to output directory")
    parser.add_argument("--questionnaire", default="questionnaires.json",
                       help="Path to questionnaire JSON file")
    args = parser.parse_args()
    
    try:
        # Read and prepare data
        print("Reading data...")
        df = pd.read_csv(args.input_file)
        
        print("Preparing data...")
        question_df = prepare_data(df)
        
        # Create plots
        print("Creating question kde plots...")
        plot_questions_kde(question_df, args.output_dir)
        print("Plotting complete")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()


