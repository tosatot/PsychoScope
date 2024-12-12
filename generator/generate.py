import pandas as pd
from tqdm import tqdm
import os
import datetime
import re
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import random
import time
import re



def load_model_dynamically(model_path):
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        device = torch.device("cpu")
        print("No GPUs detected. Using CPU.")
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float32)
        model.to(device)
    elif num_gpus == 1:
        print("Single GPU detected. Using GPU 0.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        print(f"{num_gpus} GPUs detected. Using all available GPUs.")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                max_memory={i: f"{int(torch.cuda.get_device_properties(i).total_memory / 1024**3 * 0.9)}GiB" for i in range(num_gpus)}
            )
        except Exception as e:
            print(f"Error loading model onto GPU(s): {e}")
            print("Attempting to load with CPU offloading...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                device_map="auto",
                offload_folder="offload",
                torch_dtype=torch.float32
            )
    
    return model, tokenizer


def generate_response_linux(inputs, model, tokenizer):
    """
    Generate a response using open models locally, using the HF/TRANSFORMERS library, with a given prompt, and question string.
    """
    inputs_id = tokenizer.encode(inputs, add_special_tokens=False, return_tensors="pt")
    
    # Move input to the same device as the model
    if hasattr(model, 'device'):
        device = model.device
    else:
        # Fallback to CPU if model.device is not set (shouldn't happen with our setup)
        device = torch.device('cpu')
    
    inputs_id = inputs_id.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs_id,
            max_new_tokens=120,
            do_sample=False,
            num_beams=4,
            num_return_sequences=1,
        )

 #   return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(inputs):]
    return tokenizer.decode(outputs[0])[len(inputs):] 


def generate_response_macOS(inputs, model, tokenizer, generate):
    """
    Generate a response using open models locally, using the MLX library, with a given prompt, and question string.
    """
    response = generate(model, tokenizer, inputs, temp=0, max_tokens=120)  # MAX_TOKENS is a critical parameters to avoid annecessary additional text
    return response.strip()


def generate_response_gemini(inputs, model, retry_count=0):
    """
    Generate a response using the Gemini model API,  with a given prompt, and question string.
    """
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    import google.generativeai as genai
    
    try:
        response = model.generate_content(
            inputs,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                # stop_sequences=['space'],
                max_output_tokens=130,
                temperature=0),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
        )
    except Exception as e:
        print(f"retrying Gemini due to an error: {e}")
        time.sleep(5)
        retry_count+=1
        if retry_count>10:
            raise Exception("Maximum retries exceeded for Gemini") from e
        return generate_response_gemini(inputs, model, retry_count)


    return response.parts[0].text #befor: response.text.strip()


def generate_response_claude(inputs, model, api_key, retry_count=0):
    """
    Generate a response using the Claude API, with a given prompt and question string.
    """
    import anthropic
    if not retry_count:
        retry_count = 0

    # Initialize the anthropic client with the provided API key
    client = anthropic.Client(api_key=api_key)

    try:
        # Call the Claude API to generate text
        response = client.messages.create(
            model=model,
            messages=[
                {"role": "user", "content": inputs}  # Use the input as user prompt
            ],
            max_tokens=130,
            temperature=0
        )
        message = response.content[0].text
    except Exception as e:
        print(f"retrying Claude due to an error: {e}")
        time.sleep(5)
        retry_count += 1
        if retry_count > 10:
            raise Exception("Maximum retries exceeded for Claude") from e
        return generate_response_claude(inputs, model, api_key, retry_count)

    if not message:
        return ""
    else:
        return message.strip()

def generate_response_groq(inputs, model, api_key, retry_count=0):
    """
    Generate a response using the Groq API, with a given model, prompt and question string.
    """
    from groq import Groq

    client = Groq(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": inputs
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
    except Exception as e:
        print(f"retrying Groq due to an error: {e}")
        time.sleep(5)
        retry_count += 1
        if retry_count > 10:
            raise Exception("Maximum retries exceeded for Groq") from e
        return generate_response_claude(inputs, model, api_key, retry_count)

    out = ""
    for chunk in completion:
        chunk_out = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
        out+=chunk_out
    
    return out

def generate_response_together(inputs, model, client, retry_count=0):
    """
    Generate a response using the Together AI API with conversation history.
    Args:
        inputs (str): The user's input text
        model (str): The model identifier (e.g., 'togethercomputer/llama-2-70b-chat')
        client: Together AI client instance
        retry_count (int): Number of retry attempts made
    Returns:
        tuple: (response_text, updated_history)
    """
    try:
        # Together AI API request
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": inputs}],
            max_tokens=130,
            temperature=0,
            # repetition_penalty=1.1,
            # top_p=0.7,
            # frequency_penalty=0
        )

    except Exception as e:
        print(f"Retrying due to an error occurred: {e}")
        retry_count += 1
        time.sleep(10)
        if retry_count > 10:
            raise Exception(f"Maximum retries exceeded for {model}") from e
        return generate_response_together(inputs, model, client, retry_count)

    if not response or not isinstance(response.choices[0].message.content, str):
        retry_count += 1
        time.sleep(10)
        return generate_response_together(inputs, model, client, retry_count)
    
    # Return both the response and the updated history
    return response.choices[0].message.content

def generate_response_together_with_history(inputs, model, client, history=None, retry_count=0):
    """
    Generate a response using the Together AI API with conversation history.
    Args:
        inputs (str): The user's input text
        model (str): The model identifier (e.g., 'togethercomputer/llama-2-70b-chat')
        client: Together AI client instance
        history (list, optional): List of previous conversation messages
        retry_count (int): Number of retry attempts made
    Returns:
        tuple: (response_text, updated_history)
    """
    if history is None:
        history = []  # Initialize conversation history if it's not provided

    # Add the user's current question to the history
    history.append({"role": "user", "content": inputs})

    try:
        # Together AI API request
        response = client.chat.completions.create(
            model=model,
            messages=history,
            max_tokens=130,
            temperature=0,
            # repetition_penalty=1.1,
            # top_p=0.7,
            # frequency_penalty=0
        )

    except Exception as e:
        print(f"Retrying due to an error occurred: {e}")
        retry_count += 1
        time.sleep(10)
        if retry_count > 10:
            raise Exception(f"Maximum retries exceeded for {model}") from e
        return generate_response_together_with_history(inputs, model, client, history, retry_count)

    if not response or not isinstance(response.choices[0].message.content, str):
        retry_count += 1
        time.sleep(10)
        return generate_response_together_with_history(inputs, model, client, history, retry_count)
    
    # Return both the response and the updated history
    return response.choices[0].message.content, history

def get_vertex_client(PROJECT_ID, SERVICE_ACCOUNT_FILE):
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
    import openai
    from google.auth import default, transport
    
    MODEL_LOCATION = "us-central1"

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_request = Request()
    credentials.refresh(auth_request)
    
    vertex_client = openai.OpenAI(
        base_url=f"https://{MODEL_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{MODEL_LOCATION}/endpoints/openapi/chat/completions?",
        api_key=credentials.token,
    )
    return vertex_client


def generate_response_vertex_with_history(inputs, model, client, history=None, retry_count=0):
    """
    Generate a response using the Vertex AI API with conversation history.
    """

    if history is None:
        history = []  # Initialize conversation history if it's not provided

    # Add the user's current question to the history
    history.append({"role": "user", "content": inputs})

    response = ""
    try:
        # Use the conversation history in the request
        response = client.chat.completions.create(
            model=model,
            messages=history,
            max_tokens=130,
            temperature=0,
            extra_body={
                "extra_body": {
                    "google": {
                        "model_safety_settings": {"enabled": False, "llama_guard_settings": {}}
                    }
                }
            },
        )
    except Exception as e:
        print(f"Retrying due to an error occurred: {e}")
        retry_count += 1
        time.sleep(10)
        if retry_count > 10:
            raise Exception("Maximum retries exceeded for LLama 400B") from e
        return generate_response_vertex_with_history(inputs, model, client, history, retry_count)
  
    if not response or not isinstance(response.choices[0].message.content, str):
        retry_count += 1
        time.sleep(10)
        return generate_response_vertex_with_history(inputs, model, client, history, retry_count)
    
    # Return both the response and the updated history
    return response.choices[0].message.content, history

def generate_response_vertex(inputs, model, client, args, retry_count=0):
    """
    Generate a response using the Groq API, with a given model, prompt and question string.
    """    
    response = ""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": inputs}],
            max_tokens=130,
            temperature=0,
            extra_body={
                "extra_body": {
                    "google": {
                        "model_safety_settings": {"enabled": False, "llama_guard_settings": {}}
                    }
                }
            },
        )
    except Exception as e:
        print(f"Retrying due to an error occurred: {e}")
        retry_count+=1
        time.sleep(10)
        if retry_count > 10:
            raise Exception("Maximum retries exceeded for LLama 400B") from e
        return generate_response_vertex(inputs, model, client, args, retry_count)
  
    if not response or not isinstance(response.choices[0].message.content,str):
        retry_count+=1
        time.sleep(10)
        if isinstance(response.choices[0].message.content, int) or isinstance(client,int):
            refresh_client = get_vertex_client(PROJECT_ID=args.PROJECT_ID,  SERVICE_ACCOUNT_FILE=args.SERVICE_ACCOUNT_FILE)
            return generate_response_vertex(inputs, model, refresh_client, args, retry_count)
        else:
            return generate_response_vertex(inputs, model, client, args, retry_count)

    return response.choices[0].message.content


def generate_response_random(num_responses, responses):
    """
    Generates random responses either 1 or 0 for a given number of inputs.

    Parameters:
    - num_responses (int): The number of responses to generate.

    Returns:
    - list: A list of strings formatted as '{index}:{response}', where response is either 0 or 1.
    """
    return [f"{i+1}:{random.choice(responses)}" for i in range(num_responses)]


#_______________________________________________________________________________________

def convert_results(result, column_header, questions_string, questionnare_name):
    """
    Converts model response into a list of integer scores.
    Parameters:
    - result (str): The string containing the model's response.
    - column_header (str): Column header used for error messages.
    - questions_string (str): The string containing the questions.
    Returns:
    - list: A list of integer scores extracted from the result.
    """
    result = result.strip()  # Remove leading and trailing whitespace
    result_list = []
    questions = questions_string.split('\n')  # Split the questions string into a list of questions

    questID=0
    for i, element in enumerate(result.split('\n')):
        if element.strip():  # Check if the line is not just whitespace
            try:
                match_na = re.search(r'[\:\.]\s*[\*\s]*(?:n/?a)[\*\s]*', element, re.IGNORECASE)
                if match_na:  # Check if the element contains "N/A" or "n/a"
                    if questionnare_name == "EPQ-R":
                        score = 0.5 # Transform "N/A" answers to 0.5
                        question = questions[questID] if i < len(questions) else "Unknown Question"
                        print(f"++++____++++ Score set to 0.5 for '{element}' in column {column_header} for question: {question}. ++++____++++")
                    elif questionnare_name == "BFI":
                        score = 2.5 # Transform "N/A" answers to 0.5     
                        question = questions[questID] if i < len(questions) else "Unknown Question"
                        print(f"++++____++++ Score set to 2.5 for '{element}' in column {column_header} for question: {question}. ++++____++++")
                    else:
                        print("I don't know how to treat NANs for this questionnaire. Please, check the questionnaire name.")
                    result_list.append(score)
                    questID=questID+1
                else: # If no "N/A" is found, try to extract a numeric score
                    match_numeric = re.search(r'[\:\.]\s*[\*\s]*([0-9]+)[\*\s]*', element)
                    if match_numeric:
                        clean_score = re.sub(r'[^\d]', '', match_numeric.group(1))
                        score = int(clean_score)
                        result_list.append(score)
                        questID=questID+1
                    else:
                        # Handle cases where no numeric score or "N/A" is found
                        question = questions[questID] if i < len(questions) else "Unknown Question"
                        print(f"No valid score found for '{element}' in column {column_header} just before element '{questID}'.")
                    
            except ValueError as e:
                # Handle ValueError specifically if conversion to integer fails
                question = questions[i] if i < len(questions) else "Unknown Question"
                print(f"Error processing score for '{element}' in column {column_header} for question: {question}. Error: {e}")
            except Exception as e:
                # Catch-all for any other unexpected errors
                question = questions[i] if i < len(questions) else "Unknown Question"
                print(f"Unexpected error processing '{element}' in column {column_header} for question: {question}. Error: {e}")
    
    return result_list


#_______________________________________________________________________________________

def generator(questionnaire, persona, args, variability):

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    testing_file = args.testing_file
    use_conversation_history = args.conv_hist

    df = pd.read_csv(testing_file) # Read the existing CSV file into a pandas DataFrame

    records_file = f"{args.model}_{questionnaire['name']}_{persona['name']}_{variability}_{'cHist' if args.conv_hist else ''}_{args.b_size}"
    # Check if our variability source is paraphrasing or shuffling 
    paraphrase_mode = False
    if variability=="paraphrase":
        paraphrase_mode = True

    # Check if the model is random, if not, load the model based on the environment
    if args.model == 'random':  
        pass
    elif args.environment == 'macOS':
        from mlx_lm import load, generate
        if args.model == 'mixtral_4bit':
            model, tokenizer = load("mlx-community/Mixtral-8x7B-Instruct-v0.1-hf-4bit-mlx") 
        elif args.model == 'gemma2B':
            model, tokenizer = load("mlx-community/gemma-1.1-2b-it-4bit") #now, new version of Gemma!
        elif args.model == 'gemma7B':
            model, tokenizer = load("mlx-community/gemma-1.1-7b-it-4bit") #now, new version of Gemma!
        elif args.model == "llama3.2-3b":
            model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct")
        elif args.model == "llama70B":
            model, tokenizer = load("mlx-community/Meta-Llama-3-70B-Instruct-4bit")
        elif args.model == 'hermes':
            model, tokenizer = load("mlx-community/Nous-Hermes-2-Mixtral-8x7B-DPO-4bit")

    elif args.environment == 'linux': 
        if args.model == 'gemma2-2b':
            model_path = "google/gemma-2-2b-it"
        elif args.model == "gemma2-9b":
            model_path = "google/gemma-2-9b-it"
        elif args.model == "gemma2-27b":
            model_path = "google/gemma-2-27b-it"
        elif args.model == 'gemma-rec-9b':
            model_path = "google/recurrentgemma-9b-it"
        elif args.model == 'gemma-rec-2b':
            model_path = "google/recurrentgemma-2b-it"
        elif args.model == "llama3.1-8b":
            model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif args.model == "llama3.1-70b":
            model_path = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        elif args.model == "llama3.1-405b":
            model_path = "meta-llama/Meta-Llama-3.1-405B-Instruct"
        elif args.model == "llama3.2-11b":
            model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        elif args.model == "llama3.2-90b":
            model_path = "meta-llama/Llama-3.2-90B-Vision-Instruct"
        elif args.model == "llama3.2-1b":
            model_path = "meta-llama/Llama-3.2-1B-Instruct"
        elif args.model == "llama3.2-3b":
            model_path = "meta-llama/Llama-3.2-3B-Instruct"

        elif args.model == "qwen2.5-1.5b":
            model_path = "Qwen/Qwen2.5-1.5B-Instruct"
        elif args.model == "qwen2.5-3b":
            model_path = "Qwen/Qwen2.5-3B-Instruct"
        elif args.model == "qwen2.5-7b":
            model_path = "Qwen/Qwen2.5-7B-Instruct"
        elif args.model == "qwen2.5-14b":
            model_path = "Qwen/Qwen2.5-14B-Instruct"
        elif args.model == "qwen2.5-32b":
            model_path = "Qwen/Qwen2.5-32B-Instruct"
        elif args.model == "qwen2.5-72b":
            model_path = "Qwen/Qwen2.5-72B-Instruct"

        model, tokenizer = load_model_dynamically(model_path)

    elif args.environment == "API":
        if args.model == 'gemini-pro':
            import google.generativeai as genai
            genai.configure(api_key=args.API_KEY)    
            model = genai.GenerativeModel('gemini-pro')
        elif args.model == 'haiku':
            model = 'claude-3-haiku-20240307'
        elif args.model == "llama70B":
            model = "llama3-70b-8192"
        elif args.model == "llama8B":
            model = "llama3-8b-8192"
        elif args.model == "llama3.1-405b":
            goog_vertex_client = get_vertex_client(PROJECT_ID=args.PROJECT_ID, SERVICE_ACCOUNT_FILE=args.SERVICE_ACCOUNT_FILE) 
            model = "meta/llama3-405b-instruct-maas"
        elif args.model == "llama3.1-70b":
            goog_vertex_client = get_vertex_client(PROJECT_ID=args.PROJECT_ID, SERVICE_ACCOUNT_FILE=args.SERVICE_ACCOUNT_FILE) 
            model = "meta/llama3-70b-instruct-maas"
        elif args.model == "llama3.1-8b":
            goog_vertex_client = get_vertex_client(PROJECT_ID=args.PROJECT_ID, SERVICE_ACCOUNT_FILE=args.SERVICE_ACCOUNT_FILE) 
            model = "meta/llama3-8b-instruct-maas"
        elif args.model == "llama3.1-405b-together":
            import together 
            tog_client = together.Client()
            model = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        elif args.model == "llama3.1-70b-together":
            import together 
            tog_client = together.Client()
            model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        elif args.model == "llama3.1-8b-together":
            import together 
            tog_client = together.Client()
            model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        elif args.model == "qwen2.5-72b":
            import together 
            tog_client = together.Client()
            model = "Qwen/Qwen2-72B-Instruct"
    else:
        raise ValueError(f"Unsupported environment: {args.environment}")

    order_columns = [col for col in df.columns if col.startswith("order")]
    count = 0 #shuffle/praphrase count
    insert_count = 0
    total_iterations = len(order_columns) * args.test_count

    persona_description = persona["description"]
    rememb_persona = persona["remember"]
    rememb_task = questionnaire["prompt_conclusion"]
    batch_size = int(args.b_size)
    if batch_size == 1:
        main_prompt = questionnaire["main_prompt_sq"]
    else:
        main_prompt = questionnaire["main_prompt_mq"]
    
    
    
    with tqdm(total=total_iterations) as pbar:
        # For each iteration of a certain shuffling-order/ paraphrasing-version
        for i, header in enumerate(df.columns):
            if header in order_columns:
                questions_column_index = i - 1
                count += 1 #shuffle-order/praphrase-version count

                questions_list = df.iloc[:, questions_column_index].astype(str)
                
                # Adjusted to use the batch_size parameter
                separated_questions = [questions_list[i:i+batch_size] for i in range(0, len(questions_list), batch_size)]
                questions_list = ['\n'.join([f"{i+1}.{q.split('.')[1]}" for i, q in enumerate(questions)]) for questions in separated_questions]
                
                # For each test-count (repetition of exactly the same questions order/paraphrasing - usually set to 0)
                for k in range(args.test_count):
                    
                    if paraphrase_mode:
                        column_header = f'paraphrase{count - 1}-test{k}'
                    else:
                        column_header = f'shuffle{count - 1}-test{k}'
                    
                    retry_count = 0 
                    max_retries = 1 
                    while True:
                        retry_count+=1

                        result_string_list = []
                        conversation_history = []
                        conv_hist_api = None 
                        
                        is_first_batch = True
                        for questions_string in questions_list: # For each batch of questions
                            

                            # Combine the selected persona description, main prompt, reminder, and the questions into one string
                            if not use_conversation_history or is_first_batch:
                                combined_content = f"{persona_description}\n{main_prompt}\n\n{questions_string}\n\n{rememb_persona}{rememb_task}"
                                is_first_batch = False  # Set to False after processing the first batch
                            elif use_conversation_history and args.environment == 'API':
                                if batch_size == 1:
                                    combined_content = f"{questionnaire['prompt_conv-hist_sq']}\n{questions_string}\n{rememb_persona} {rememb_task}"
                                else:
                                    combined_content = f"{questionnaire['prompt_conv-hist']}\n{questions_string}\n{rememb_persona} {rememb_task}"
                                
                            else:
                                #combined_content = '\n'.join(conversation_history) + f"\n{questionnaire['prompt_conv-hist']}\n\n{questions_string}\n\n{rememb_persona}{rememb_task}" 
                                if batch_size == 1:
                                    combined_content = f"{conversation_history}\n{questionnaire['prompt_conv-hist_sq']}\n{questions_string}\n{rememb_persona} {rememb_task}"
                                    
                                else:
                                    combined_content = f"{conversation_history}\n{questionnaire['prompt_conv-hist']}\n{questions_string}\n{rememb_persona} {rememb_task}"
                                
                            # Generate responses based on the enviroment  
                            if args.model == 'random':  
                                responses=questionnaire["response"]
                                # Dynamically set batch_size to the number of questions in questions_string
                                actual_batch_size = len(questions_string.split('\n'))
                                result_list=generate_response_random(actual_batch_size, responses)
                                result = '\n'.join(result_list)
                                inputs = 'random'
                                #result = '\n'.join(generate_response_random(len(questions_string.split('\n'))))
                            
                            elif args.environment == 'linux':
                                messages = [{"role": "user", "content": combined_content}] # Use the combined string as content for a single "user" role message
                                inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # Apply the chat template 
                                # Remove the "Cutting Knowledge Date" and "Today Date" part
                                
                                inputs = re.sub(r"Cutting Knowledge Date: .*?\nToday Date: .*?\n|<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>[\s\S]*?<\|eot_id\|>", "", inputs)


                                
                                result = generate_response_linux(inputs, model, tokenizer) # Generate a response  

                            elif args.environment == 'macOS':
                                if args.model == 'mixtral':
                                    inputs = f"<s>[INST] {combined_content } [/INST]"
                                elif args.model == 'gemma2B' or args.model == 'gemma7B':
                                    inputs = f"<start_of_turn>user\n{combined_content}<end_of_turn>\n<start_of_turn>model"
                                elif args.model == 'hermes':
                                    inputs = f"<|im_start|>user\n{combined_content}<|im_end|>\n<|im_start|>system"
                                elif args.model == 'llama8B' or args.model == 'llama70B' or args.model == 'llama3.2-3b':
                                    inputs = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{combined_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"    
                                else:
                                    raise ValueError(f"Model {args.model} not recognized.")
                                result = generate_response_macOS(inputs, model, tokenizer, generate)
                            
                            elif args.environment == 'API':
                                if args.model == 'gemini-pro':
                                    inputs = f"<start_of_turn>user\n{combined_content}<end_of_turn>\n<start_of_turn>model"
                                    result = generate_response_gemini(inputs=inputs, model=model)
                                elif args.model == 'haiku':
                                    inputs = f"<start_of_turn>user\n{combined_content}<end_of_turn>\n<start_of_turn>model"
                                    result = generate_response_claude(inputs=inputs, model=model, api_key=args.API_KEY)
                                elif args.model == 'llama8B' or args.model == 'llama70B':
                                    inputs = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{combined_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nSure, I can help with that. Here is the information you requested:"    
                                    result = generate_response_groq(inputs=inputs, model=model, api_key=args.API_KEY)
                                elif args.model == 'llama3.1-405b' or args.model == 'llama3.1-70b' or args.model == 'llama3.1-8b' :
                                    inputs = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{combined_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                                    if not use_conversation_history:
                                        result = generate_response_vertex(inputs=inputs, model=model, client=goog_vertex_client, args=args)
                                    else:
                                        result, conv_hist_api = generate_response_vertex_with_history(inputs=inputs, model=model, client=goog_vertex_client, history=conv_hist_api)
                                        conv_hist_api.append({"role": "assistant", "content": result})
                                elif args.model == 'llama3.1-405b-together' or args.model == "llama3.1-70b-together" or args.model == "llama3.1-8b-together" or args.model == "qwen2.5-72b":
                                    inputs = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{combined_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                                    if not use_conversation_history:
                                        result = generate_response_together(inputs=inputs, model=model, client=tog_client)
                                    else:
                                        result, conv_hist_api = generate_response_together_with_history(inputs=inputs, model=model, client=tog_client, history=conv_hist_api)
                                        conv_hist_api.append({"role": "assistant", "content": result})

                                    
                            if result is not None:
                                result_string_list.append(result.strip())

                                # Update conversation history
                                clean_input = re.sub(r'^<bos><start_of_turn>user\n?', '', inputs.strip())
                                conversation_history = f"{clean_input}\n{result.strip()}"
                            else:
                                print(f"Warning: generate_response returned None for inputs: {inputs}")

                            os.makedirs(f"results/prompts_{records_file}_{timestamp}", exist_ok=True)
                            os.makedirs(f"results/responses_{records_file}_{timestamp}", exist_ok=True)

                            with open(f'results/prompts_{records_file}_{timestamp}/{records_file}-{questionnaire["name"]}-{variability}{count - 1}.txt', "a") as file:
                                file.write(f'{inputs}\n====\n')
                            with open(f'results/responses_{records_file}_{timestamp}/{records_file}-{questionnaire["name"]}-{variability}{count - 1}.txt', "a") as file:
                                file.write(f'{result}\n====\n')

                        result_string = '\n'.join(result_string_list)
                        result_list = convert_results(result_string, column_header, '\n'.join(questions_list), questionnaire["name"])
                        
                        # Attempt to insert/update the DataFrame with result_list
                        try:
                            if column_header in df.columns:
                                df[column_header] = result_list
                            else:
                                df.insert(i + insert_count + 1, column_header, result_list)
                                insert_count += 1
                            break # Exit the loop on success
                        except Exception as e:
                            print(f"Unable to capture the responses on {column_header} due to {e}. Attempt {retry_count} of {max_retries}.")  # Often the problem here is the number of responses is less than the number of questions
                            retry_count += 1
                            if retry_count >= max_retries:
                                print("Max retries reached. Moving to next set of questions.")
                                break  # Exit the loop after max retries

                    df.to_csv(testing_file, index=False)
                    pbar.update(1)
