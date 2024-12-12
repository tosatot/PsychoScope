import csv
import json
import logging
import os
import random
from matplotlib import patches as mpatches
import scipy.stats as stats
from statistics import mean, stdev
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import ptitprince as pt
import seaborn as sns
from datetime import datetime
import re
import time
from tqdm import tqdm
from collections import Counter
from tenacity import retry, stop_after_attempt, wait_random_exponential
import re
import time
from tqdm import tqdm
from generator.generate import load_model_dynamically

logger = logging.getLogger(__name__)

def get_questionnaire(questionnaire_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Absolute path of the script's directory
    questionnaire_path = os.path.join(script_dir, 'questionnaires.json')  # Path to 'questionnaires.json'

    try:
        with open(questionnaire_path) as dataset:
            data = json.load(dataset)
            questionnaires_data = data['questionnaires']  # Adjusted to match new structure
    except FileNotFoundError:
        raise FileNotFoundError("The 'questionnaires.json' file does not exist.")

    # Matching by questionnaire_name in questionnaires_data
    questionnaire = None
    for item in questionnaires_data:  # Search in questionnaires_data
        if item["name"] == questionnaire_name:
            questionnaire = item
            break

    if questionnaire is None:
        raise ValueError("Questionnaire not found.")

    return questionnaire

def get_persona(persona_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Absolute path of the script's directory
    persona_path = os.path.join(script_dir, 'questionnaires.json')  # Path remains the same as we're using the same JSON file

    try:
        with open(persona_path) as dataset:
            data = json.load(dataset)
            personas_data = data['personas']  # Accessing the personas section
    except FileNotFoundError:
        raise FileNotFoundError("The 'questionnaires.json' file does not exist.")

    # Searching for the specified persona by name
    persona = None
    for item in personas_data:  # Iterating through personas_data
        if item["name"] == persona_name:
            persona = item
            break

    if persona is None:
        raise ValueError("Persona not found.")

    return persona



def plot_data_distribution(cat_list, test_data, crowd_list,  questionnaire, save_path, plot_name):
    """
    Plots the data distribution using violin plots for each category and overlays scatter plots for individual LLM run scores.
    
    Parameters:
    - cat_list: List of category names.
    - test_data: The test data obtained from the analysis.
    - crowd_list: List of tuples containing crowd name and size.
    - questionnaire: The questionnaire object containing category details.
    - save_path: Path to save the generated plot.
    - save_name: Name of the file, containing the model name.
    """
    # _____USE PTITPRINCE FOR RAINCLOUD PLOT_____

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(cat_list))+0.2
    
    # First, plot Mean and Std for Crowd Data
    for index, crowd in enumerate(crowd_list):
        means = [cat['crowd'][index]['mean'] for cat in questionnaire['categories']]
        stds = [cat['crowd'][index]['std'] for cat in questionnaire['categories']]
        ax.errorbar(x_pos + 0.1 * index, means, yerr=stds, fmt='o', label=f'Human {crowd[0]}')

    # Prepare data for RainCloud
    data_to_plot = []
    for i, cat in enumerate(questionnaire["categories"]):
        for data in test_data:
            cat_scores = [data[key] for key in data if key in cat["cat_questions"]]
            cat_score = sum(cat_scores) if questionnaire["compute_mode"] == "SUM" else np.mean(cat_scores)
            data_to_plot.append({'Category': cat_list[i], 'Score': cat_score})

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data_to_plot)

    # Color (palette) 
    pal = [(0.8,0.15,0.15)]*3  #dark red
    
    # Plotting RainCloud
    pt.RainCloud(data=df, x='Category', y='Score', palette=pal, bw=.26, width_viol=.6, ax=ax, orient='v', offset = .13)
    
    raincloud_patch = mpatches.Patch(color=pal[0], label= plot_name)   
    handles, labels = ax.get_legend_handles_labels()  # Get existing handles and labels
    handles.append(raincloud_patch)

    if questionnaire['name'] == "EPQ-R" or questionnaire['name'] == "BFI":
    
        # Set "chance level" 
        if questionnaire['name'] == "EPQ-R":
            chance_levels = [11.5, 16, 12, 10.5] # This should be your actual chance levels
        elif questionnaire['name'] == "BFI":
            chance_levels = [3, 3, 3, 3, 3]
        
        for index, chance_level in enumerate(chance_levels):
            # Define the start and end points for the horizontal line for each category
            start = x_pos[index] - 0.5  # adjust the start point as needed
            end = x_pos[index] + 0.2  # adjust the end point as needed
            ax.hlines(y=chance_level, xmin=start, xmax=end, color='gray', linestyle='--', linewidth=2, label='Chance Level' if index == 0 else "")

        # Only add the chance level label once to the legend
        if 'Chance Level' not in labels:
            handles.append(plt.Line2D([], [], color='gray', linestyle='--', label='Chance Level'))
            labels.append('Chance Level')

    # Update legend to include chance level
    ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1, 1), title="Legend")

    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title="Legend")
    ax.set_title(f'{plot_name}__crowd data & individual LLM run scores')
    ax.set_ylabel('Scores')
    ax.set_xlabel('Categories')

    if questionnaire['name'] == "EPQ-R":
        ymin = 0
        ymax = 26
        ax.set_ylim([ymin, ymax]) 
    elif questionnaire['name'] == "BFI":
        ymin = 1
        ymax = 5
        ax.set_ylim([ymin, ymax]) 
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)




#@retry(stop=stop_after_attempt(10), wait=wait_random_exponential(min=15, max=120))
def paraphrase_question(
    question, model, tokenizer, num_of_permutations, existing_paraphrases, str_type, name, temperature
):
    print(f"NOW PARAPHRASING THIS QUESTION: {question} ")
    delimiter = "?" if str_type == "question" else "."

    # The new prompt according to the instructions
    prompt = f"""
    You are tasked with paraphrasing a {str_type} from the {name} questionnaire. Your goal is to create multiple variations of the {str_type} while preserving its original meaning in the context of the questionnaire. These paraphrased versions will be used to create alternative versions of the questionnaire for research purposes.
    Here is the original {str_type} you need to paraphrase:

    <original_statement>
    {question}
    </original_statement>
    Please follow these instructions carefully:

    - Create {num_of_permutations} unique paraphrased versions of the given {str_type}.
    - Ensure that each paraphrased version maintains the same core meaning as the original {str_type}. The psychological trait or behavior being assessed should remain consistent.
    - Introduce small linguistic variations in your paraphrases.This can include:
    - Using synonyms
    - Changing sentence structure
    - Altering word order
    - Using different phrasal constructions
    - Keep the paraphrased {str_type}s natural and conversational. They should sound like something a person would say in a self-assessment context.
    - Maintain the same level of specificity as the original {str_type}. Don't make the paraphrased versions more general or more specific than the original.
    - Preserve any key terms that are essential to the meaning of the statement.
    - Ensure that the paraphrased versions are grammatically correct and easily understandable.
    - Avoid introducing new concepts or ideas that are not present in the original {str_type}.
    - Keep the paraphrased versions concise. They should be similar in length to the original {str_type}.

    - Output your paraphrased versions in the following format:

        1.<paraphrased version 1>
        2.<paraphrased version 2>
        3.<paraphrased version 3>
        ...
        {num_of_permutations}.<paraphrased version {num_of_permutations}>

    - Do not include any additional text before or after the numbered list of paraphrased {str_type}s.
    - Output only {num_of_permutations} paraphrased {str_type}s.
    - Stop generation when you reach {num_of_permutations} paraphrased statements and output ENDS_HERE only when you finish generating the {num_of_permutations} paraphrased statements.
    """
    print(f"Generating Paraphrases for {str_type} {question}")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    # Tokenize the input prompt
    #print("prompt lenght",len(prompt))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate the paraphrases using Hugging Face model API
    outputs = model.generate(
        **inputs,
        max_new_tokens=1500,
        temperature=temperature,
        num_return_sequences=1, #num_of_permutations,  # Generate the required number of paraphrases
        do_sample=True,
    )
    #print(f"Generated paraphrases {outputs}")

    

    # Decode and print the generated output
    
    output_text = tokenizer.decode(outputs[0])
    #print(f"Generated output {output_text}")
    # Post-processing: Filter out anything after "ENDS_HERE"
    output_text = output_text[len(prompt):]
    output_text = output_text.split("ENDS_HERE")[0]
    

    # Remove everything before the first "1."
    # Split on the numbers to get the paraphrased sentences
    paraphrases = re.split(r'\d+\.\s', output_text.strip())[1:]

    # Clean up any extra tokens like "<|eot_id|>" if they appear
    paraphrases = [sentence.strip().replace("<|eot_id|>", "") for sentence in paraphrases]
    print(f"{len(paraphrases)} Paraphrases generated for {question}: {paraphrases}")

    return set(paraphrases)

def paraphrase_list_of_questions(
    all_questions_to_paraphrase, name, num_of_permutations=100, str_type="question", output_file="paraphrased_questions.json"
    ):
    
    model, tokenizer = load_model_dynamically("meta-llama/Meta-Llama-3.1-70B-Instruct")
    results = {}
    for question in tqdm(all_questions_to_paraphrase, desc="Paraphrasing Questions"):
        question_set = set()
        iteration = 1
        temperature = 0.2  # Initial temperature

        while len(question_set) < num_of_permutations:
            print(f"\nCurrent Iteration {iteration}:")
            new_questions = paraphrase_question(
                question,
                model,
                tokenizer,
                num_of_permutations - len(question_set),
                list(question_set),
                str_type,
                name,
                temperature,
            )
            #num_of_permutations = num_of_permutations - len(question_set)
            question_set.update(new_questions)
            print(f"Total unique paraphrases so far: {len(question_set)}")
            iteration += 1
            temperature = min(temperature + 0.1, 1.0)  # Increase temperature by 0.1, capped at 1.0

        results[question] = list(question_set)[:num_of_permutations]
        print(f"\nFinal {num_of_permutations} unique paraphrases for '{question}':")
        for q in list(question_set)[:num_of_permutations]:
            print(f"- {q}")

    # Save the results to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"\nParaphrased questions saved to {output_file}")
    return results


def generate_testfile(questionnaire, args, variability_source):

    # Extract relevant arguments needed for generating the test file
    n_iter = args.n_iter
    output_file = args.testing_file
    test_count = args.test_count
    csv_output = []

    # Extract all question texts and indices from the questionnaire
    questions_list = list(questionnaire["questions"].values())
    question_indices = list(questionnaire["questions"].keys())

    # Decide on the action based on whether paraphrasing is requested
    if variability_source == 'paraphrase' and n_iter > 0:
        # Check if a precomputed paraphrase file exists to avoid redoing the work
        file_name = questionnaire["name"] + '_questions_paraphrased.json'
        if os.path.exists(file_name):
            # Load the existing paraphrases from the file
            paraphrase_results = json.load(open(file_name, 'r')) 
            for question, paraphrases in paraphrase_results.items():
                paraphrase_results[question] = paraphrases[:n_iter]  # Keep only the first 100 paraphrases
        else:
            # Remove question numbering for paraphrasing
            clean_questions_list = [re.sub(r'\d+\.\s+', '', question) for question in questions_list]
            # Generate paraphrases for all questions
            paraphrase_results = paraphrase_list_of_questions(
                all_questions_to_paraphrase=clean_questions_list,
                num_of_permutations=100, #n_iter,
                str_type=questionnaire["str_type"],
                name=questionnaire["name"]
            )
            # Save the paraphrases for future use
            with open(questionnaire["name"] + '_questions_paraphrased.json', 'w') as json_file:
                json.dump(paraphrase_results, json_file, indent=4)

        # Prepare the CSV structure for paraphrased questions
        paraphrase_sets = len(list(paraphrase_results.values())[0]) # Number of paraphrase sets per question

        # for idx in range(paraphrase_sets):
        #     # Select and number the paraphrased questions for the current set
        #     paraphrased_questions = [paraphrase_results[questionnaire["questions"][str(index + 1)]][idx] for index in range(len(question_indices))]
        #     # Add the paraphrased questions and their order to the CSV structure
        #     csv_output.append([f'Prompt: {questionnaire["main_prompt"]}'] + [f'{i+1}. {q}' for i, q in enumerate(paraphrased_questions)])
        #     csv_output.append([f'order-paraphrase-{idx}'] + question_indices)
        #     # Add empty response placeholders for each test iteration
        #     for count in range(test_count):
        #         csv_output.append([f'paraphrase{idx}-test{count}'] + [''] * len(question_indices))

        for idx in range(paraphrase_sets):
            # Original questions for the first set
            if idx == 0:
                original_questions = [questionnaire["questions"][str(index + 1)] for index in range(len(question_indices))]
                csv_output.append([f'Prompt: {questionnaire["main_prompt"]}'] + [f'{i+1}. {q}' for i, q in enumerate(original_questions)])
            else:
                # Select and number the paraphrased questions for the current set
                paraphrased_questions = [paraphrase_results[questionnaire["questions"][str(index + 1)]][idx-1] for index in range(len(question_indices))]
                csv_output.append([f'Prompt: {questionnaire["main_prompt"]}'] + [f'{i+1}. {q}' for i, q in enumerate(paraphrased_questions)])

            csv_output.append([f'order-paraphrase-{idx}'] + question_indices)

            # Add empty response placeholders for each test iteration
            for count in range(test_count):
                csv_output.append([f'paraphrase{idx}-test{count}'] + [''] * len(question_indices))



    elif variability_source == 'shuffle' and n_iter > 0:
        # Prepare the CSV structure for shuffled or original questions
        
        # Get the main prompt based on batch size
        batch_size = int(args.b_size)
        if batch_size == 1:
            main_prompt = questionnaire["main_prompt_sq"]  # Single question prompt
        else:
            main_prompt = questionnaire["main_prompt_mq"]  # Multiple questions prompt
            
        for perm in range(n_iter + 1):
            if n_iter > 0 and perm != 0:
                random.shuffle(question_indices)
            
            # Number the questions as per the current order
            questions = [f'{index}. {questionnaire["questions"][question]}' for index, question in enumerate(question_indices, 1)]
            # Add the questions and their order to the CSV structure
            csv_output.append([f'Prompt: {main_prompt}'] + questions)
            csv_output.append([f'order-{perm}'] + question_indices)
            # Add empty response placeholders for each test iteration
            for count in range(test_count):
                csv_output.append([f'shuffle{perm}-test{count}'] + [''] * len(question_indices))

    # Transpose the list of lists to align with CSV format and write it out
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(zip(*csv_output))



# this was the old function in the util file that I have then modified to take into account non valid results, assigning values such 0.5 or 2.5, depending on the scale
# however. now back to using this one, because the old one was giving un-nutural values non fitting the distribution
def convert_data (questionnaire, testing_file):
    '''
    Converts the CSV result file into a format suitable for analysis. It handles reverse-scored items and extracts responses for each question.
    '''

    # Check if the testing file exists
    if not os.path.exists(testing_file):
        logger.error("Testing file does not exist.")
        current_folder = os.getcwd()
        logger.error("Current Folder:", current_folder)    
        logger.error("Testing file path:", testing_file)     
        sys.exit(1)

    test_data = []

    # Check for missing data
    df = pd.read_csv(testing_file)
    is_empty = df.isna().any()[lambda x: x]
    if len(is_empty) > 0:
        logger.info(f"{testing_file} has missing data in cols: {', '.join([f'{k}' for k,v in is_empty.items()])}")

    with open(testing_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        
        # Identify indices of columns that define the order of questions
        order_indices = []
        for index, column in enumerate(header):
            if column.startswith("order"):
                order_indices.append(index)
                
        # Process each question block
        for order_index in order_indices:
            score_index = order_index+1
            column_data = {}
            csvfile.seek(0)
            next(reader)
            # Read each row and handle responses
            for row in reader:
                question_index = int(row[order_index])
                try:
                    response = float(row[score_index])  # Convert to float to preserve decimals
                    # Reverse scoring logic
                    if question_index in questionnaire["reverse"]:
                        column_data[question_index] = float(questionnaire["scale"]) - response
                        logger.debug(f'{testing_file}: detected reverse question. row: {question_index}')
                    else:
                        column_data[question_index] = response
                except ValueError as e:
                    column_data[question_index] = np.nan
                    logger.debug(f'{testing_file}: Column {order_index}, question {question_index}, has error: {e}. Assigning NaN. Prepared data may be unbalanced, verify input data.')
            test_data.append(column_data)

    return test_data

# def convert_data(questionnaire, testing_file):
#     '''
#     Converts the CSV result file into a format suitable for analysis. It handles reverse-scored items and extracts responses for each question.
#     '''

#     # Check if testing_file exists
#     if not os.path.exists(testing_file):
#         print("Testing file does not exist.")
#         current_folder = os.getcwd()
#         print("Current Folder:", current_folder)    
#         print("Testing file path:", testing_file)     
#         sys.exit(1)

#     test_data = []
    
#     with open(testing_file, 'r') as csvfile:
#         reader = csv.reader(csvfile)
#         header = next(reader)
        
#         # Take the index of columns which refer to the question order
#         order_indices = []
#         for index, column in enumerate(header):
#             if column.startswith("order"):
#                 order_indices.append(index)
                
#         # For each question order, record the corresponding test data
#         for i in range(len(order_indices)):
            
#             # start and end are the range of the test data which correspond to the current question order
#             start = order_indices[i] + 1
#             end = order_indices[i+1] - 1 if order_indices[i] != order_indices[-1] else len(header)
            
#             # column index refers to the index of the column within the test data range
#             for column_index in range(start, end):
#                 column_data = {}
#                 csvfile.seek(0)
#                 next(reader)
                
#                 # For each row in the table, take the question index x and related response y as `"x": y` format
#                 for row in reader:
#                     try: 
#                         question_index = int(row[start-1])
#                         response = row[column_index]

#                         # Check if the response is a valid float or integer
#                         try:
#                             response_value = float(response)
#                         except ValueError:
#                             # Handle invalid responses based on the questionnaire's requirements
#                             if questionnaire["name"] == "EPQ-R":
#                                 response_value = 0.5  # Assign a default value of 0.5 for EPQ-R
#                                 print(f"Invalid response '{response}' for question {question_index} in column {column_index + 1}. Assigned .5")
#                             elif questionnaire["name"] == "BFI":
#                                 response_value = 2.5  # Assign a default value of 2.5 for BFI
#                                 print(f"Invalid response '{response}' for question {question_index} in column {column_index + 1}. Assigned 2.5")
#                             else:
#                                 print(f"Invalid response '{response}' for question {question_index} in column {column_index + 1}.")
#                                 continue  # Skip to the next row

#                         # Check whether the question is a reverse scale
#                         if question_index in questionnaire["reverse"]:
#                             column_data[question_index] = questionnaire["scale"] - response_value
#                         else:
#                             column_data[question_index] = response_value

#                     except ValueError as e:
#                         question = questionnaire["questions"][str(question_index)] if str(question_index) in questionnaire["questions"] else "Unknown Question"
#                         print(f"Error processing score for '{response}' in column {column_index + 1} for question: {question}. Error: {e}")

#                 test_data.append(column_data)
            
#     return test_data

# def compute_statistics(questionnaire, data_list):
#     results = []
    
#     for cat in questionnaire["categories"]:
#         scores_list = []
        
#         for data in data_list:
#             scores = []
#             for key in data:
#                 if key in cat["cat_questions"]:
#                     scores.append(data[key])
            
#             # Getting the computation mode (SUM or AVG)
#             if questionnaire["compute_mode"] == "SUM":
#                 scores_list.append(sum(scores))
#             else:
#                 scores_list.append(mean(scores))
        
#         if len(scores_list) < 2:
#             raise ValueError("The test file should have at least 2 test cases.")
        
#         results.append((mean(scores_list), stdev(scores_list), len(scores_list)))
        
#     return results


def compute_statistics(questionnaire, data_list):
    """
    Compute statistics for questionnaire responses
    
    Args:
        questionnaire: Dictionary containing questionnaire configuration
        data_list: List of response dictionaries
        
    Returns:
        List of tuples containing (mean, std, n) for each category
        
    Raises:
        ValueError: If data_list contains fewer than 2 responses
    """
    if len(data_list) < 2:
        raise ValueError("The test file should have at least 2 test cases")
        
    results = []
    
    for cat in questionnaire["categories"]:
        scores_list = []
        
        for data in data_list:
            scores = []
            for key in data:
                if key in cat["cat_questions"]:
                    scores.append(data[key])
            
            if scores:  # Only compute if we have scores
                # Getting the computation mode (SUM or AVG)
                if questionnaire["compute_mode"] == "SUM":
                    scores_list.append(np.sum(scores))
                else:
                    scores_list.append(np.mean(scores))
        
        if not scores_list:  # Handle empty categories
            results.append((np.nan, np.nan, 0))
        else:
            results.append((
                np.mean(scores_list),
                np.std(scores_list, ddof=1) if len(scores_list) > 1 else np.nan,
                len(scores_list)
            ))
    
    return results

def hypothesis_testing(result1, result2, significance_level, model, crowd_name):
    output_list = ''
    output_text = f'### Compare with {crowd_name}\n'

    # Extract the mean, std and size for both data sets
    mean1, std1, n1 = result1
    mean2, std2, n2 = result2
    output_list += f'{mean2:.1f} $\pm$ {std2:.1f}'
    
    # Add an epsilon to prevent the zero standard deviarion
    epsilon = 1e-8
    std1 += epsilon
    std2 += epsilon
    
    output_text += '\n- **Statistic**:\n'
    output_text += f'{model}:\tmean1 = {mean1:.1f},\tstd1 = {std1:.1f},\tn1 = {n1}\n'
    output_text += f'{crowd_name}:\tmean2 = {mean2:.1f},\tstd2 = {std2:.1f},\tn2 = {n2}\n'
    
    # Perform F-test
    output_text += '\n- **F-Test:**\n\n'
    
    if std1 > std2:
        f_value = std1 ** 2 / std2 ** 2
        df1, df2 = n1 - 1, n2 - 1
    else:
        f_value = std2 ** 2 / std1 ** 2
        df1, df2 = n2 - 1, n1 - 1

    p_value = (1 - stats.f.cdf(f_value, df1, df2)) * 2
    equal_var = True if p_value > significance_level else False
    
    output_text += f'\tf-value = {f_value:.4f}\t($df_1$ = {df1}, $df_2$ = {df2})\n\n'
    output_text += f'\tp-value = {p_value:.4f}\t(two-tailed test)\n\n'
    output_text += '\tNull hypothesis $H_0$ ($s_1^2$ = $s_2^2$): '

    if p_value > significance_level:
        output_text += f'\tSince p-value ({p_value:.4f}) > α ({significance_level}), $H_0$ cannot be rejected.\n\n'
        output_text += f'\t**Conclusion ($s_1^2$ = $s_2^2$):** The variance of average scores responsed by {model} is statistically equal to that responsed by {crowd_name} in this category.\n\n'
    else:
        output_text += f'\tSince p-value ({p_value:.4f}) < α ({significance_level}), $H_0$ is rejected.\n\n'
        output_text += f'\t**Conclusion ($s_1^2$ ≠ $s_2^2$):** The variance of average scores responsed by {model} is statistically unequal to that responsed by {crowd_name} in this category.\n\n'

    # Performing T-test
    output_text += '- **Two Sample T-Test (Equal Variance):**\n\n' if equal_var else '- **Two Sample T-test (Welch\'s T-Test):**\n\n'
    
    df = n1 + n2 - 2 if equal_var else ((std1**2 / n1 + std2**2 / n2)**2) / ((std1**2 / n1)**2 / (n1 - 1) + (std2**2 / n2)**2 / (n2 - 1))
    t_value, p_value = stats.ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2, equal_var=equal_var)

    output_text += f'\tt-value = {t_value:.4f}\t($df$ = {df:.1f})\n\n'
    output_text += f'\tp-value = {p_value:.4f}\t(two-tailed test)\n\n'
    
    output_text += '\tNull hypothesis $H_0$ ($µ_1$ = $µ_2$): '
    if p_value > significance_level:
        output_text += f'\tSince p-value ({p_value:.4f}) > α ({significance_level}), $H_0$ cannot be rejected.\n\n'
        output_text += f'\t**Conclusion ($µ_1$ = $µ_2$):** The average scores of {model} is assumed to be equal to the average scores of {crowd_name} in this category.\n\n'
        # output_list += f' ( $-$ )'

    else:
        output_text += f'Since p-value ({p_value:.4f}) < α ({significance_level}), $H_0$ is rejected.\n\n'
        if t_value > 0:
            output_text += '\tAlternative hypothesis $H_1$ ($µ_1$ > $µ_2$): '
            output_text += f'\tSince p-value ({(1-p_value/2):.1f}) > α ({significance_level}), $H_1$ cannot be rejected.\n\n'
            output_text += f'\t**Conclusion ($µ_1$ > $µ_2$):** The average scores of {model} is assumed to be larger than the average scores of {crowd_name} in this category.\n\n'
            # output_list += f' ( $\\uparrow$ )'
        else:
            output_text += '\tAlternative hypothesis $H_1$ ($µ_1$ < $µ_2$): '
            output_text += f'\tSince p-value ({(1-p_value/2):.1f}) > α ({significance_level}), $H_1$ cannot be rejected.\n\n'
            output_text += f'\t**Conclusion ($µ_1$ < $µ_2$):** The average scores of {model} is assumed to be smaller than the average scores of {crowd_name} in this category.\n\n'
            # output_list += f' ( $\\downarrow$ )'

    output_list += f' | '
    return (output_text, output_list)


import json 
import copy
import requests

payload_template = {
    "questions": [
        {"text": "You regularly make new friends.", "answer": None},
        {"text": "You spend a lot of your free time exploring various random topics that pique your interest.", "answer": None},
        {"text": "Seeing other people cry can easily make you feel like you want to cry too.", "answer": None},
        {"text": "You often make a backup plan for a backup plan.", "answer": None},
        {"text": "You usually stay calm, even under a lot of pressure.", "answer": None},
        {"text": "At social events, you rarely try to introduce yourself to new people and mostly talk to the ones you already know.", "answer": None},
        {"text": "You prefer to completely finish one project before starting another.", "answer": None},
        {"text": "You are very sentimental.", "answer": None},
        {"text": "You like to use organizing tools like schedules and lists.", "answer": None},
        {"text": "Even a small mistake can cause you to doubt your overall abilities and knowledge.", "answer": None},
        {"text": "You feel comfortable just walking up to someone you find interesting and striking up a conversation.", "answer": None},
        {"text": "You are not too interested in discussing various interpretations and analyses of creative works.", "answer": None},
        {"text": "You are more inclined to follow your head than your heart.", "answer": None},
        {"text": "You usually prefer just doing what you feel like at any given moment instead of planning a particular daily routine.", "answer": None},
        {"text": "You rarely worry about whether you make a good impression on people you meet.", "answer": None},
        {"text": "You enjoy participating in group activities.", "answer": None},
        {"text": "You like books and movies that make you come up with your own interpretation of the ending.", "answer": None},
        {"text": "Your happiness comes more from helping others accomplish things than your own accomplishments.", "answer": None},
        {"text": "You are interested in so many things that you find it difficult to choose what to try next.", "answer": None},
        {"text": "You are prone to worrying that things will take a turn for the worse.", "answer": None},
        {"text": "You avoid leadership roles in group settings.", "answer": None},
        {"text": "You are definitely not an artistic type of person.", "answer": None},
        {"text": "You think the world would be a better place if people relied more on rationality and less on their feelings.", "answer": None},
        {"text": "You prefer to do your chores before allowing yourself to relax.", "answer": None},
        {"text": "You enjoy watching people argue.", "answer": None},
        {"text": "You tend to avoid drawing attention to yourself.", "answer": None},
        {"text": "Your mood can change very quickly.", "answer": None},
        {"text": "You lose patience with people who are not as efficient as you.", "answer": None},
        {"text": "You often end up doing things at the last possible moment.", "answer": None},
        {"text": "You have always been fascinated by the question of what, if anything, happens after death.", "answer": None},
        {"text": "You usually prefer to be around others rather than on your own.", "answer": None},
        {"text": "You become bored or lose interest when the discussion gets highly theoretical.", "answer": None},
        {"text": "You find it easy to empathize with a person whose experiences are very different from yours.", "answer": None},
        {"text": "You usually postpone finalizing decisions for as long as possible.", "answer": None},
        {"text": "You rarely second-guess the choices that you have made.", "answer": None},
        {"text": "After a long and exhausting week, a lively social event is just what you need.", "answer": None},
        {"text": "You enjoy going to art museums.", "answer": None},
        {"text": "You often have a hard time understanding other people’s feelings.", "answer": None},
        {"text": "You like to have a to-do list for each day.", "answer": None},
        {"text": "You rarely feel insecure.", "answer": None},
        {"text": "You avoid making phone calls.", "answer": None},
        {"text": "You often spend a lot of time trying to understand views that are very different from your own.", "answer": None},
        {"text": "In your social circle, you are often the one who contacts your friends and initiates activities.", "answer": None},
        {"text": "If your plans are interrupted, your top priority is to get back on track as soon as possible.", "answer": None},
        {"text": "You are still bothered by mistakes that you made a long time ago.", "answer": None},
        {"text": "You rarely contemplate the reasons for human existence or the meaning of life.", "answer": None},
        {"text": "Your emotions control you more than you control them.", "answer": None},
        {"text": "You take great care not to make people look bad, even when it is completely their fault.", "answer": None},
        {"text": "Your personal work style is closer to spontaneous bursts of energy than organized and consistent efforts.", "answer": None},
        {"text": "When someone thinks highly of you, you wonder how long it will take them to feel disappointed in you.", "answer": None},
        {"text": "You would love a job that requires you to work alone most of the time.", "answer": None},
        {"text": "You believe that pondering abstract philosophical questions is a waste of time.", "answer": None},
        {"text": "You feel more drawn to places with busy, bustling atmospheres than quiet, intimate places.", "answer": None},
        {"text": "You know at first glance how someone is feeling.", "answer": None},
        {"text": "You often feel overwhelmed.", "answer": None},
        {"text": "You complete things methodically without skipping over any steps.", "answer": None},
        {"text": "You are very intrigued by things labeled as controversial.", "answer": None},
        {"text": "You would pass along a good opportunity if you thought someone else needed it more.", "answer": None},
        {"text": "You struggle with deadlines.", "answer": None},
        {"text": "You feel confident that things will work out for you.", "answer": None}
    ],
    "gender": None,
    "inviteCode": "",
    "teamInviteKey": "",
    "extraData": []
}

role_mapping = {'ISTJ': 'Logistician', 'ISTP': 'Virtuoso', 'ISFJ': 'Defender', 'ISFP': 'Adventurer', 'INFJ': 'Advocate', 'INFP': 'Mediator', 'INTJ': 'Architect', 'INTP': 'Logician', 'ESTP': 'Entrepreneur', 'ESTJ': 'Executive', 'ESFP': 'Entertainer', 'ESFJ': 'Consul', 'ENFP': 'Campaigner', 'ENFJ': 'Protagonist', 'ENTP': 'Debater', 'ENTJ': 'Commander'}


def parsing(score_list):
    code = ''
    
    if score_list[0] >= 50:
        code = code + 'E'
    else:
        code = code + 'I'

    if score_list[1] >= 50:
        code = code + 'N'
    else:
        code = code + 'S'

    if score_list[2] >= 50:
        code = code + 'T'
    else:
        code = code + 'F'

    if score_list[3] >= 50:
        code = code + 'J'
    else:
        code = code + 'P'

    if score_list[4] >= 50:
        code = code + '-A'
    else:
        code = code + '-T'

    return code, role_mapping[code[:4]]


# scores: List of int, length: 60, int range: -3~3
def query_16personalities_api(scores):
    payload = copy.deepcopy(payload_template)
    
    for index, score in enumerate(scores):
        payload['questions'][index]["answer"] = score
    
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en,zh-CN;q=0.9,zh;q=0.8",
        "content-length": "5708",
        "content-type": "application/json",
        "origin": "https://www.16personalities.com",
        "referer": "https://www.16personalities.com/free-personality-test",
        "sec-ch-ua": "'Not_A Brand';v='99', 'Google Chrome';v='109', 'Chromium';v='109'",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "Windows",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        'content-type': 'application/json',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36',
    }
    
    session = requests.session()
    r = session.post('https://www.16personalities.com/test-results', data=json.dumps(payload), headers=headers)
    
    sess_r = session.get("https://www.16personalities.com/api/session")
    scores = sess_r.json()['user']['scores']
    
    if sess_r.json()['user']['traits']['energy'] != 'Extraverted':
        energy_value = 100 - (101 + scores[0]) // 2
    else:
        energy_value = (101 + scores[0]) // 2
    if sess_r.json()['user']['traits']['mind'] != 'Intuitive':
        mind_value = 100 - (101 + scores[1]) // 2
    else:
        mind_value = (101 + scores[1]) // 2
    if sess_r.json()['user']['traits']['nature'] != 'Thinking':
        nature_value = 100 - (101 + scores[2]) // 2
    else:
        nature_value = (101 + scores[2]) // 2
    if sess_r.json()['user']['traits']['tactics'] != 'Judging':
        tactics_value = 100 - (101 + scores[3]) // 2
    else:
        tactics_value = (101 + scores[3]) // 2
    if sess_r.json()['user']['traits']['identity'] != 'Assertive':
        identity_value = 100 - (101 + scores[4]) // 2
    else:
        identity_value = (101 + scores[4]) // 2
    
    code, role = parsing([energy_value, mind_value, nature_value, tactics_value, identity_value])
    
    return code, role, [energy_value, mind_value, nature_value, tactics_value, identity_value]


def analysis_personality(args, test_data):
    all_data = []
    result_file = args.results_file
    cat = ['Personality Type', 'Role', 'Extraverted', 'Intuitive', 'Thinking', 'Judging', 'Assertive']
    df = pd.DataFrame(columns=cat)

    for case in test_data:
        ordered_list = [case[key]-4 for key in sorted(case.keys())]
        all_data.append(ordered_list)
        result = query_16personalities_api(ordered_list)
        result = result[:2] + tuple(result[2])
        df.loc[len(df)] = result
    
    column_sums = [sum(col) for col in zip(*all_data)]
    avg_data = [int(sum / len(all_data)) for sum in column_sums]
    avg_result = query_16personalities_api(avg_data)
    avg_result = avg_result[:2] + tuple(avg_result[2])
    df.loc["Avg"] = avg_result
    
    # Writing the results into a text file
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("# 16 Personality Results\n\n")
        f.write(df.to_markdown())



def analysis_results(questionnaire, args, persona_name):
    significance_level = args.significance_level
    testing_file = args.testing_file
    result_file = args.results_file
    model = args.model
    

    test_data = convert_data(questionnaire, testing_file)
    
    if questionnaire["name"] == "16P":
        analysis_personality(args, test_data)
        return
    else:
        test_results = compute_statistics(questionnaire, test_data)
        
    cat_list = [cat['cat_name'] for cat in questionnaire['categories']]
    crowd_list = [(c["crowd_name"], c["n"]) for c in questionnaire['categories'][0]["crowd"]]
    mean_list = [[] for i in range(len(crowd_list) + 1)]
    
    output_list = f'# {questionnaire["name"]} Results\n\n'
    output_list += f'| Category | {model} (n = {len(test_data)}) | ' + ' | '.join([f'{c[0]} (n = {c[1]})' for c in crowd_list]) + ' |\n'
    output_list += '| :---: | ' + ' | '.join([":---:" for i in range(len(crowd_list) + 1)]) + ' |\n'
    output_text = ''

    # Analysis by each category
    for cat_index, cat in enumerate(questionnaire['categories']):
        output_text += f'## {cat["cat_name"]}\n'
        output_list += f'| {cat["cat_name"]} | {test_results[cat_index][0]:.1f} $\pm$ {test_results[cat_index][1]:.1f} | '
        mean_list[0].append(test_results[cat_index][0])
        
        for crowd_index, crowd_group in enumerate(crowd_list):
            crowd_data = (cat["crowd"][crowd_index]["mean"], cat["crowd"][crowd_index]["std"], cat["crowd"][crowd_index]["n"])
            result_text, result_list = hypothesis_testing(test_results[cat_index], crowd_data, significance_level, model, crowd_group[0])
            output_list += result_list
            output_text += result_text
            mean_list[crowd_index+1].append(crowd_data[0])
            
        output_list += '\n'
    
    #plot_bar_chart(mean_list, cat_list, [model] + [c[0] for c in crowd_list], save_name=args.figures_file, title=questionnaire["name"])

    plot_data_distribution(cat_list=cat_list, test_data=test_data, crowd_list=crowd_list, questionnaire=questionnaire, save_path=args.figures_file, plot_name=model+'_'+persona_name)
   
    output_list += f'\n\n![Plot]({args.figures_file} "Distribution plot of {model} on {questionnaire["name"]}")\n\n'
    # Writing the results into a text file
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(output_list + output_text)



def run_psychobench(parser, args, generator):
    timestamp = datetime.now().strftime('%m%d_%H%M')
    models_list = args.models.split(',')
    questionnaire_list = ['BFI', 'DTDD', 'EPQ-R', 'ECR-R', 'CABIN', 'GSE', 'LMS', 'BSRI', 'ICB', 'LOT-R', 'Empathy', 'EIS', 'WLEIS', '16P'] \
                        if args.questionnaire == 'ALL' else args.questionnaire.split(',')
    persona_list = args.persona.split(',')
    variability_list = args.variab_source.split(',')
    cHist=[]
    if args.conv_hist:
        cHist="cHist"

    for model_name in models_list:
        args.model = model_name

        for questionnaire_name in questionnaire_list:
            questionnaire = get_questionnaire(questionnaire_name)

            for persona_name in persona_list:
                persona = get_persona(persona_name)

                for variability in variability_list:

                    args.testing_file = f'results/{model_name}_{questionnaire["name"]}_{persona_name}_{variability}_{timestamp}_{cHist}_{args.b_size}.csv'
                    args.results_file = f'results/{model_name}_{questionnaire["name"]}_{persona_name}_{variability}_{timestamp}_{cHist}_{args.b_size}.md'
                    args.figures_file = f'results/figures/{model_name}_{questionnaire["name"]}_{persona_name}_{variability}_{timestamp}_{cHist}_{args.b_size}.png'

                    os.makedirs("results", exist_ok=True)
                    os.makedirs("results/figures", exist_ok=True)
                    
                    # Generation
                    if args.mode in ['generation', 'auto']:
                        generate_testfile(questionnaire, args, variability)
                    
                    # Testing
                    if args.mode in ['testing', 'auto']:
                        generator(questionnaire, persona, args, variability)
                        
                    # # Analysis
                    # if args.mode in ['analysis', 'auto']:
                    #     try:
                    #         analysis_results(questionnaire, args,persona_name)
                    #     except Exception as e:
                    #         print(f'Unable to analyze {args.testing_file}. Error: {e}')
                    #         raise  # Re-raise the exception after printing the error message to get the full traceback.
