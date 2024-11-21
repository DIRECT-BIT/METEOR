import os
import json
import argparse
import openai
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Distill high-quality domain data using GPT-4.")
    parser.add_argument("--data_folder", type=str, required=True, help="Folder containing the domain questions and guidelines.")
    parser.add_argument("--out_folder", type=str, required=True, help="Folder to save the distilled data.")
    parser.add_argument("--api_key", type=str, default=None, help="Your OpenAI API key.")
    parser.add_argument("--max_retries", type=int, default=5, help="Maximum number of retries for API calls.")
    return parser.parse_args()

def load_guidelines(data_folder):
    guidelines_file = os.path.join(data_folder, "guidelines.jsonl")
    if not os.path.exists(guidelines_file):
        raise FileNotFoundError(f"Guidelines file not found at {guidelines_file}")
    guidelines = []
    with open(guidelines_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            guidelines.append(data)
    return guidelines

def generate_prompt(question_title, question_content, guideline):
    prompt = f"""You are provided with an educational question and a guideline for solving it. Please provide a detailed, accurate answer that follows the guideline and includes necessary reasoning steps.

### Question Title:
{question_title}

### Question Content:
{question_content}

### Guideline:
{guideline}

Please generate a comprehensive answer based on the above."""
    return prompt

def call_gpt4(prompt, max_retries):
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional educational assistant skilled at providing detailed and accurate answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            answer = response['choices'][0]['message']['content']
            return answer
        except openai.error.OpenAIError as e:
            print(f"Error: {e}")
            print(f"Retrying... ({attempt + 1}/{max_retries})")
    print("Maximum retries exceeded.")
    return None

def save_distilled_data(distilled_data, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    output_file = os.path.join(out_folder, "distilled_data.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for data in distilled_data:
            json_line = json.dumps(data, ensure_ascii=False)
            f.write(json_line + "\n")
    print(f"Distilled data saved to {output_file}")

def main():
    args = parse_arguments()

    # Set OpenAI API key
    if args.api_key:
        openai.api_key = args.api_key
    else:
        # Alternatively, you can set the OPENAI_API_KEY environment variable
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key must be provided via --api_key argument or OPENAI_API_KEY environment variable")

    # Load questions and guidelines
    print("Loading guidelines...")
    guidelines = load_guidelines(args.data_folder)
    print(f"Total guidelines: {len(guidelines)}")

    distilled_data = []

    print("Starting data distillation...")
    for item in tqdm(guidelines):
        question_title = item.get("question_title", "")
        question_content = item.get("question_content", "")
        guideline = item.get("guideline", "")

        if not question_title or not question_content or not guideline:
            continue

        # Generate prompt
        prompt = generate_prompt(question_title, question_content, guideline)

        # Call GPT-4 API
        answer = call_gpt4(prompt, args.max_retries)

        if answer:
            distilled_item = {
                "question_title": question_title,
                "question_content": question_content,
                "guideline": guideline,
                "answer": answer.strip()
            }
            distilled_data.append(distilled_item)

    # Save distilled data
    print("Saving distilled data...")
    save_distilled_data(distilled_data, args.out_folder)

if __name__ == "__main__":
    main()
