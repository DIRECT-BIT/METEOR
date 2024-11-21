import os
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate guidelines using a base model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model.")
    parser.add_argument("--data_folder", type=str, required=True, help="Folder containing raw data files.")
    parser.add_argument("--out_folder", type=str, required=True, help="Folder to save the generated guidelines.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing.")
    return parser.parse_args()

def load_data(data_folder):
    data_files = ["cv_data.json", "dl_data.json", "ml_data.json", "nlp_data.json"]
    data = []
    for file_name in data_files:
        file_path = os.path.join(data_folder, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist.")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            file_data = json.load(f)
            data.extend(file_data)
    return data

def generate_prompts(data):
    prompts = []
    for item in data:
        title = item.get("title", "").strip()
        question_content = item.get("question_content", "").strip()
        if title and question_content:
            prompt = f"Title: {title}\nContent: {question_content}\n\nBased on the above, please provide guidelines for solving this problem."
            prompts.append(prompt)
    return prompts

def generate_guidelines(prompts, model, tokenizer, batch_size, device):
    guidelines = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                do_sample=True,
                top_p=0.95,
                top_k=50,
            )
        batch_guidelines = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        guidelines.extend(batch_guidelines)
    return guidelines

def save_guidelines(guidelines, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    output_file = os.path.join(out_folder, "guidelines.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for guideline in guidelines:
            json_line = json.dumps({"guideline": guideline}, ensure_ascii=False)
            f.write(json_line + "\n")
    print(f"Guidelines saved to {output_file}")

def main():
    args = parse_arguments()
    
    # Load base model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    model.eval()
    
    # Load data
    print("Loading data...")
    data = load_data(args.data_folder)
    print(f"Total data items: {len(data)}")
    
    # Generate prompts
    print("Generating prompts...")
    prompts = generate_prompts(data)
    print(f"Total prompts: {len(prompts)}")
    
    # Generate guidelines
    print("Generating guidelines...")
    guidelines = generate_guidelines(prompts, model, tokenizer, args.batch_size, device)
    
    # Save guidelines
    print("Saving guidelines...")
    save_guidelines(guidelines, args.out_folder)

if __name__ == "__main__":
    main()
