import argparse
from generator.utils import get_questionnaire, convert_data
import os
import pandas as pd
import numpy as np
import logging
from generator.config import order_dict

logger = logging.getLogger(__name__)

def parse_filename(filename):
    """
    Parse filenames in both formats:
    - New: "llama3.1-405b_BFI_assistant_shuffle_1014_2147_cHist_2"
    - Legacy: "llama3.1-405b_BFI_assistant_shuffle_0929_1516_[]"
    
    Returns:
        tuple: (model_name, persona, variability, history, b_size)
    """
    try:
        # Split filename and extension
        base_name = filename.rsplit(".", 1)[0]
        parts = base_name.split('_')
        
        # Common elements in both formats
        model_name = parts[0]
        persona = parts[2]
        variability = parts[3]
        
        # Handle the history and batch size differently based on format
        if len(parts) >= 8 and parts[-2] == 'cHist':
            # New format with explicit history and batch size
            history = 1  # If cHist is present, it means history is being used
            b_size = int(parts[-1])
        else:
            # Legacy format
            history_marker = parts[-1] if len(parts) >= 7 else '[]'
            history = 1 if history_marker != '[]' else 0
            b_size = 10  # Default batch size for legacy format
            
        return model_name, persona, variability, history, b_size
        
    except Exception as e:
        logging.error(f"Error parsing filename {filename}: {e}")
        raise ValueError(f"Invalid filename format: {filename}")


def prepare_data(directory, questionnaire_name):
    questionnaire = get_questionnaire(questionnaire_name)
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv') and questionnaire_name.upper() in f.upper()]
    logger.info(f"Number of files to process: {len(all_files)}")
    
    processed_data = []
    for filename in all_files:
        try:
            model_name, persona, variability, history, b_size = parse_filename(filename)
            
            testing_file_path = os.path.join(directory, filename)
            test_data = convert_data(questionnaire, testing_file_path)
            
            for cat in questionnaire["categories"]:
                for data in test_data:
                    trait_scores = [data[key] for key in data if key in cat["cat_questions"]]
                    trait_score = sum(trait_scores) if questionnaire["compute_mode"] == "SUM" else np.mean(trait_scores)
                    
                    processed_data.append({
                        'trait': cat['cat_name'],
                        'score': trait_score,
                        'model': model_name,
                        'persona': persona,
                        'variability': variability,
                        'history': history,
                        'b_size': b_size,
                    })
                    
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            continue
    
    df = pd.DataFrame(processed_data)
    logger.info(f"Shape of DataFrame after processing: {df.shape}")
    
    def get_order(item, order_list):
        try:
            return order_list.index(item)
        except ValueError:
            return len(order_list)

    df['model_order'] = df['model'].apply(lambda x: get_order(x.rstrip('b'), order_dict['model']))
    df['persona_order'] = df['persona'].apply(lambda x: get_order(x, order_dict['persona']))
    df['variability_order'] = df['variability'].apply(lambda x: get_order(x, order_dict['variability']))
    
    df = df.sort_values(['persona_order', 'model_order', 'variability_order', 'trait'])
    
    df[['model_type', 'model_size']] = df['model'].str.split('-', expand=True)

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prepare data for analysis')
    parser.add_argument('-q', '--questionnaire_name', type=str, help='Questionnaire name')
    parser.add_argument('-d', '--directory', type=str, help='Directory containing data files')
    parser.add_argument('-o', '--output', type=str, help='Output directory to save prepared data file')
    parser.add_argument('-v', '--verbosity', help='Verbosity levels: DEBUG, INFO, WARNING, ERROR', default='INFO')

    args = parser.parse_args()

# Set up logging
    logging.basicConfig(filename='prepare_data.log', encoding='utf-8', level=args.verbosity)
    
    df = prepare_data(args.directory, args.questionnaire_name)
    df.to_csv(f"{args.output}/{args.questionnaire_name}_data.csv", index=False)
    logger.info(f"Data saved to{args.output}/{args.questionnaire_name}_data.csv")
