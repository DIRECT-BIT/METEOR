import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import openai
from tqdm import tqdm
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser(description="Iterative training data generation with model self-evaluation and GPT-4 feedback.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the distilled data used for iterative training.")
    parser.add_argument("--out_folder", type=str, required=True, help="Folder to save the iterative training dataset.")
    parser.add_argument("--api_key", type=str, default=None, help="Your OpenAI API key.")
    parser.add_argument("--max_iterations", type=int, default=3, help="Maximum number of iterations for refinement.")
    parser.add_argument("--max_retries", type=int, default=5, help="Maximum number of retries for API calls.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing.")
    parser.add_argument("--fine_tune_script", type=str, default="finetune.py", help="Path to the fine-tuning script.")
    parser.add_argument("--fine_tune_args", type=str, default="", help="Additional arguments for fine-tuning.")
    parser.add_argument("--fine_tune_interval", type=int, default=10, help="Number of samples to accumulate before fine-tuning.")
    parser.add_argument("--gpu_devices", type=str, default="0", help="Comma-separated list of GPU device IDs to use for fine-tuning.")
    return parser.parse_args()

def load_data(data_path):
    data_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            data_list.append(data)
    return data_list

def generate_cot(model, tokenizer, prompts, device):
    cot_outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the output to get the model's answer
        answer = output_text.replace(prompt, "").strip()
        cot_outputs.append(answer)
    return cot_outputs

def evaluate_answer_with_gpt4(question, answer, max_retries):
    feedback_prompt = f"""You are provided with a question and an answer with reasoning steps (Chain-of-Thought). Please evaluate the correctness of the answer and reasoning. If it is correct, respond with "Correct". If not, provide feedback on where it is incorrect and give hints for improvement.

### Question:
{question}

### Answer with Reasoning:
{answer}

Please provide your evaluation:"""

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for educational content."},
                    {"role": "user", "content": feedback_prompt}
                ],
                temperature=0,
                max_tokens=512,
            )
            evaluation = response['choices'][0]['message']['content']
            return evaluation.strip()
        except openai.error.OpenAIError as e:
            print(f"Error: {e}")
            print(f"Retrying evaluation... ({attempt + 1}/{max_retries})")
    print("Maximum retries exceeded for evaluation.")
    return None

def fine_tune_model(model_path, training_data_path, output_dir, fine_tune_script, fine_tune_args, gpu_devices):
    # Build the fine-tuning command
    command = [
        "deepspeed",
        "--include",
        f"localhost:{gpu_devices}",
        fine_tune_script,
        "--model_name_or_path", model_path,
        "--train_data_path", training_data_path,
        "--output_dir", output_dir,
    ]
    # Add additional fine-tuning arguments
    if fine_tune_args:
        additional_args = fine_tune_args.strip().split()
        command.extend(additional_args)
    # Run the fine-tuning command
    print(f"Starting fine-tuning with command: {' '.join(command)}")
    subprocess.run(command, check=True)

def main():
    args = parse_arguments()

    # Set OpenAI API key
    if args.api_key:
        openai.api_key = args.api_key
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key must be provided via --api_key argument or OPENAI_API_KEY environment variable")

    # Load the fine-tuned model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    model.eval()

    # Load the data
    print("Loading data...")
    data_list = load_data(args.data_path)
    print(f"Total data items: {len(data_list)}")

    iterative_data = []
    accumulated_data = []

    output_dir = args.out_folder
    os.makedirs(output_dir, exist_ok=True)
    iterative_data_file = os.path.join(output_dir, "iterative_training_data.jsonl")

    print("Starting iterative data generation and fine-tuning...")
    for idx, item in enumerate(tqdm(data_list)):
        question = item.get("question_content", "")
        if not question:
            continue

        prompt = f"Question: {question}\nAnswer with reasoning:"

        # Iterative refinement
        for iteration in range(args.max_iterations):
            # Model generates answer with reasoning (CoT)
            answer = generate_cot(model, tokenizer, [prompt], device)[0]

            # GPT-4 evaluates the answer
            evaluation = evaluate_answer_with_gpt4(question, answer, args.max_retries)

            if not evaluation:
                print("Failed to get evaluation from GPT-4.")
                break  # Skip to next item

            if "Correct" in evaluation:
                print(f"Iteration {iteration + 1}: Answer is correct.")
                iterative_sample = {
                    "question": question,
                    "answer": answer,
                    "evaluation": evaluation
                }
                iterative_data.append(iterative_sample)
                accumulated_data.append(iterative_sample)
                break  # Exit the iteration loop
            else:
                print(f"Iteration {iteration + 1}: Answer is incorrect. GPT-4 feedback: {evaluation}")
                # Create a new prompt including GPT-4's feedback
                prompt = f"Question: {question}\nFeedback: {evaluation}\nPlease provide a revised answer with reasoning:"
        else:
            # Maximum iterations reached without correct answer
            print(f"Maximum iterations reached for question: {question}")
            iterative_sample = {
                "question": question,
                "answer": answer,
                "evaluation": evaluation
            }
            iterative_data.append(iterative_sample)
            accumulated_data.append(iterative_sample)

        # Save the iterative data incrementally
        with open(iterative_data_file, 'a', encoding='utf-8') as f:
            json_line = json.dumps(iterative_sample, ensure_ascii=False)
            f.write(json_line + '\n')

        # Check if accumulated data reaches the fine-tuning interval
        if len(accumulated_data) >= args.fine_tune_interval:
            # Save the accumulated data to a temporary training data file
            temp_training_data_path = os.path.join(output_dir, "temp_training_data.jsonl")
            with open(temp_training_data_path, 'w', encoding='utf-8') as f:
                for data in accumulated_data:
                    json_line = json.dumps(data, ensure_ascii=False)
                    f.write(json_line + '\n')

            # Fine-tune the model with the accumulated data
            fine_tune_model(
                model_path=args.model_path,
                training_data_path=temp_training_data_path,
                output_dir=args.model_path,  # Overwrite the model
                fine_tune_script=args.fine_tune_script,
                fine_tune_args=args.fine_tune_args,
                gpu_devices=args.gpu_devices,
            )

            # Reload the fine-tuned model
            print("Reloading the fine-tuned model...")
            model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
            model.eval()

            # Clear the accumulated data
            accumulated_data = []
            # Remove the temporary training data file
            os.remove(temp_training_data_path)

    print(f"Iterative training data saved to {iterative_data_file}")

if __name__ == "__main__":
    main()
