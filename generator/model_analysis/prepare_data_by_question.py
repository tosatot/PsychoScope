import argparse
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
        base_name = filename.rsplit(".", 1)[0]
        parts = base_name.split('_')
        
        model_name = parts[0]
        persona = parts[2]
        variability = parts[3]
        
        if len(parts) >= 8 and parts[-2] == 'cHist':
            history = 1
            b_size = int(parts[-1])
        else:
            history_marker = parts[-1] if len(parts) >= 7 else '[]'
            history = 1 if history_marker != '[]' else 0
            b_size = 10
            
        return model_name, persona, variability, history, b_size
        
    except Exception as e:
        logging.error(f"Error parsing filename {filename}: {e}")
        raise ValueError(f"Invalid filename format: {filename}")

def prepare_data(directory, questionnaire_name):
    """
    Process data files and extract individual question scores.
    
    Args:
        directory (str): Directory containing the data files
        questionnaire_name (str): Name of the questionnaire
        
    Returns:
        pandas.DataFrame: Processed data with individual question scores
    """
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv') and questionnaire_name.upper() in f.upper()]
    logger.info(f"Number of files to process: {len(all_files)}")
    
    processed_data = []
    
    for filename in all_files:
        try:
            model_name, persona, variability, history, b_size = parse_filename(filename)
            
            # Read the CSV file
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            
            # Get the number of shuffle runs from the column names
            shuffle_columns = [col for col in df.columns if col.startswith('shuffle')]
            num_runs = len(shuffle_columns)
            
            # Process each shuffle run
            for run_idx in range(num_runs):
                shuffle_col = f'shuffle{run_idx}-test0'
                order_col = f'order-{run_idx}'
                
                # Get the questions and their corresponding scores for this run
                questions = df.iloc[:, 0].tolist()  # First column contains questions
                scores = pd.to_numeric(df[shuffle_col], errors='coerce')
                orders = pd.to_numeric(df[order_col], errors='coerce')
                
                # Process each question
                for q_idx, (question, score, order) in enumerate(zip(questions, scores, orders), 1):
                    processed_data.append({
                        'question_number': q_idx,
                        'question_text': question,
                        'score': score,
                        'presentation_order': order,
                        'run_number': run_idx,
                        'model': model_name,
                        'persona': persona,
                        'variability': variability,
                        'history': history,
                        'b_size': b_size
                    })
                    
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    
    # Add ordering columns for sorting
    def get_order(item, order_list):
        try:
            return order_list.index(item)
        except ValueError:
            return len(order_list)
    
    df['model_order'] = df['model'].apply(lambda x: get_order(x.rstrip('b'), order_dict['model']))
    df['persona_order'] = df['persona'].apply(lambda x: get_order(x, order_dict['persona']))
    df['variability_order'] = df['variability'].apply(lambda x: get_order(x, order_dict['variability']))
    
    # Sort the DataFrame
    df = df.sort_values(['persona_order', 'model_order', 'variability_order', 'run_number', 'question_number'])
    
    # Extract model type and size
    df[['model_type', 'model_size']] = df['model'].str.split('-', expand=True)
    
    logger.info(f"Shape of processed DataFrame: {df.shape}")
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
    output_file = f"{args.output}/{args.questionnaire_name}_question_level_data.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Data saved to {output_file}")