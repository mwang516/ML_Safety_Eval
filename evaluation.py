import os
import json
import random
from tqdm import tqdm
from openai import OpenAI
from transformers import pipeline
from huggingface_hub import login
from collections import defaultdict

# Need to figure out a way to store safely (using .env)
os.environ["GITHUB_TOKEN"] = "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"
os.environ["OPENAI_API_KEY"] = "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"
login("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")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Requirements for Llama Guard which cannot be run due to hardware limitations
# ----------------------------------------------------------------------------
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# model_id = "meta-llama/LlamaGuard-7b"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

# Wrapper for OpenAI Moderation API
def openai_moderation_wrapper(input_text):
    try:
        moderation_result = client.moderations.create(
            model="omni-moderation-latest", input=input_text
        )
        return moderation_result.results[0]
    except Exception as e:
        print(f"Error calling OpenAI Moderation API: {e}")
        return None

# Wrapper for OpenAI Response Generation
def openai_response_generation_wrapper(model_name, prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Wrapper for HuggingFace Model Response Generation
def huggingface_response_generation_wrapper(model_name, prompt):
    hf_pipeline = pipeline("text-generation", model=model_name)
    result = hf_pipeline(prompt, max_length=100, truncation=True)
    return result[0]["generated_text"]

# Configed model: type, name and filepath
MODEL_CONFIGS = [
    {"type": "openai", "name": "gpt-3.5-turbo", "filepath": "output/gpt-3.5-turbo-responses.json"},
    {"type": "openai", "name": "gpt-4o-mini", "filepath": "output/gpt-4o-mini-responses.json"},
    {"type": "huggingface", "name": "distilbert/distilgpt2", "filepath": "output/distilbert_distilgpt2-responses.json"},
]

PROMPT_FILE = "alert.jsonl"  # Input prompts in JSONL
EVAL_OUTPUT_FILE = "evaluation_results.json" #file_path to evaluation results
OUTPUT_DIR = "output"  # Directory to store results
PROMPT_AMOUNT = 100
# LLAMAGUARD_ID = "meta-llama/LlamaGuard-7b"

# Generate responses based on model_type, model_name and given prompts
def generate_response(model_type, model_name, prompt):
    if model_type == "openai":
      return openai_response_generation_wrapper(model_name, prompt)
    
    elif model_type == "huggingface":
      return huggingface_response_generation_wrapper(model_name, prompt)

    else:
      print(f"Model {model_name} is currently unable to be processed")
      return "Unable to process"

# Load prompts from file_path
def load_prompts(file_path, limit):
    """Load the first 'limit' prompts from a JSONL file."""
    prompts = []
    with open(file_path, "r") as f:
      for line in f:
        prompts.append(json.loads(line))
    
    random.shuffle(prompts)
    return prompts[:limit]

# Save data in JSON format to file_path
def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

# Helper method to save evaluation results 
def save_evaluation_results(output_file, result_summary):
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # Convert to a dictionary for quick lookup by model_name
    existing_dict = {entry["model_name"]: entry for entry in existing_data}

    # Overwrite or create new result_summary entry for model_name
    existing_dict[result_summary["model_name"]] = result_summary

    # Convert back to a list and save the updated results
    updated_data = list(existing_dict.values())
    with open(output_file, "w") as f:
        json.dump(updated_data, f, indent=4)
    
    print(f"\nSaved evaluation results for {result_summary['model_name']} to {output_file}")

# Evaluate prompt responses of given model using OpenAI Moderation API
def evaluate_responses_with_moderation(responses, model_config):
    """Evaluate responses stored in a JSON file using OpenAI's moderation endpoint."""

    model_name = model_config["name"]

    total_responses = len(responses)
    flagged_count = 0
    category_counts = defaultdict(int) 
    flagged_scores = []

    print(f"Evaluating responses for {model_name}...")
    
    for item in tqdm(responses, desc="Evaluating"):
        try:
            response_text = item["response"]
            result = openai_moderation_wrapper(response_text)
            if result is None:
              continue

            # print(result) (debugging)
            categories = dict(result.categories)

            if result.flagged:
                flagged_count += 1
                scores = vars(result.category_scores)
                flagged_scores.append(sum(score for score in scores.values()) / len(scores))
                # print(f"\n {flagged_scores}") (debugging)
                for category, is_flagged in categories.items():
                    if is_flagged:
                        category_counts[category] += 1

        except Exception as e:
            print(f"Error moderating response: {e}")
            continue

    # Percentage of responses that were flagged
    flagged_percentage = flagged_count / total_responses * 100 

    # Average score of flagged
    average_flagged_score = sum(flagged_scores) / len(flagged_scores) if flagged_scores else 0.0

    # Evaluation result format
    result_summary = {
        "model_name": model_name,
        "safe_responses_percentage": round(100.0 - flagged_percentage, 2),
        "average_flagged_score": round(average_flagged_score, 3),
        "categories": dict(category_counts)
    }

    save_evaluation_results(EVAL_OUTPUT_FILE, result_summary)
    
    # Printing summary results
    print(f"\nEvaluation Results for {model_type} ({model_name}):")
    print(f"Flagged Responses: {flagged_count} out of {total_responses}")
    print(f"Safe Responses: {total_responses - flagged_count} out of {total_responses}")

    # Printing category counts
    print("\nCounts of Flagged Categories:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")

# Main Pipeline
if __name__ == "__main__":
    # Load the prompts dataset
    prompts = load_prompts(PROMPT_FILE, limit=PROMPT_AMOUNT)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Iterate over each model
    for model_config in MODEL_CONFIGS:
        model_type = model_config["type"]
        model_name = model_config["name"]

        print(f"Generating responses with {model_type}'s {model_name}...")

        responses = []

        # Generate responses
        for prompt in tqdm(prompts, desc=f"Processing {model_name}"):
            try:
                user_prompt = prompt["prompt"]
                response = generate_response(model_type, model_name, user_prompt)
                responses.append({"prompt": user_prompt, "response": response})
            except Exception as e:
                print(f"Error processing prompt: {e}")
                continue  

        # Save generated responses
        output_file = os.path.join(
          OUTPUT_DIR, f"{model_name.replace('/', '_')}-responses.json"
        )
        save_json(responses, output_file)
        model_config["filepath"] = output_file

        print(f"Saved responses for {model_name} to {output_file}")

        # Evaluate responses
        evaluate_responses_with_moderation(responses, model_config)


