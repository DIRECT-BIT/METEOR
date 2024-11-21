import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Self-training with contrastive learning based on reasoning complexity.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model after Stage 2 training.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data used for self-training.")
    parser.add_argument("--out_folder", type=str, required=True, help="Folder to save the self-trained model.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of update steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--fp16", action='store_true', help="Whether to use fp16 mixed precision training.")
    parser.add_argument("--bf16", action='store_true', help="Whether to use bf16 mixed precision training.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config file.")
    parser.add_argument("--overwrite_output_dir", action='store_true', help="Overwrite the content of the output directory.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling for text generation.")
    return parser.parse_args()

def load_data(data_path):
    # Load data from JSONL file
    data_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            data_list.append(data)
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_list(data_list)
    return dataset

def main():
    args = parse_arguments()

    # Load the tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Load the dataset
    print("Loading dataset for self-training...")
    dataset = load_data(args.data_path)

    # Generate different reasoning outputs
    def generate_outputs(batch):
        questions = batch['question'] if 'question' in batch else batch['question_content']
        simple_outputs = []
        complex_outputs = []
        
        for question in questions:
            input_text = f"Question: {question}\nAnswer with reasoning:"
            inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=args.max_seq_length)
            inputs = inputs.to(device)

            # Simple reasoning (shorter output)
            with torch.no_grad():
                simple_output = model.generate(
                    input_ids=inputs,
                    max_length=args.max_seq_length,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            simple_text = tokenizer.decode(simple_output[0], skip_special_tokens=True)
            simple_answer = simple_text.replace(input_text, "").strip()
            simple_outputs.append(simple_answer)

            # Complex reasoning (longer output)
            with torch.no_grad():
                complex_output = model.generate(
                    input_ids=inputs,
                    max_length=args.max_seq_length,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            complex_text = tokenizer.decode(complex_output[0], skip_special_tokens=True)
            complex_answer = complex_text.replace(input_text, "").strip()
            complex_outputs.append(complex_answer)

        return {'simple_answer': simple_outputs, 'complex_answer': complex_outputs, 'question': questions}

    # Apply generation
    print("Generating reasoning outputs...")
    generated_dataset = dataset.map(generate_outputs, batched=True, batch_size=args.per_device_train_batch_size)

    # Prepare data for contrastive learning
    def preprocess_function(examples):
        inputs = []
        labels = []
        for question, simple_ans, complex_ans in zip(examples['question'], examples['simple_answer'], examples['complex_answer']):
            # Compute the length of the answers as a proxy for complexity
            simple_length = len(tokenizer.encode(simple_ans))
            complex_length = len(tokenizer.encode(complex_ans))

            if complex_length > simple_length:
                # Prefer complex answer
                preferred_answer = complex_ans
                dispreferred_answer = simple_ans
            else:
                # Prefer simple answer
                preferred_answer = simple_ans
                dispreferred_answer = complex_ans

            # Create input-output pairs for contrastive learning
            # Positive example
            pos_input = f"Question: {question}\nAnswer with reasoning:{preferred_answer}"
            inputs.append(pos_input)
            labels.append(1)  # Label 1 for preferred

            # Negative example
            neg_input = f"Question: {question}\nAnswer with reasoning:{dispreferred_answer}"
            inputs.append(neg_input)
            labels.append(0)  # Label 0 for dispreferred

        # Tokenize inputs
        model_inputs = tokenizer(inputs, max_length=args.max_seq_length, truncation=True, padding=True)
        model_inputs["labels"] = labels
        return model_inputs

    print("Preparing data for contrastive learning...")
    tokenized_dataset = generated_dataset.map(preprocess_function, batched=True, remove_columns=generated_dataset.column_names)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.out_folder,
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        deepspeed=args.deepspeed,
        report_to="none",
        dataloader_drop_last=True,
    )

    # Define custom loss function for contrastive learning
    from transformers import Trainer

    class ContrastiveTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits  # Shape: [batch_size, seq_length, vocab_size]

            # Compute the log-likelihood of the sequences
            shift_logits = logits[..., :-1, :].contiguous()  # Shift so that tokens align with labels
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            
            # Obtain the average per-token loss for each sequence
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
            per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            per_seq_loss = per_token_loss.view(shift_labels.size()).mean(dim=1)  # Shape: [batch_size]

            # Reshape labels to match loss
            labels = labels.to(per_seq_loss.device).float()

            # Compute contrastive loss (e.g., margin ranking loss)
            pos_loss = per_seq_loss[::2]  # Even indices are positive examples
            neg_loss = per_seq_loss[1::2]  # Odd indices are negative examples

            margin = 0.0  # You can adjust the margin
            contrastive_loss = torch.mean(torch.relu(pos_loss - neg_loss + margin))

            return (contrastive_loss, outputs) if return_outputs else contrastive_loss

    # Initialize trainer
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Start training
    print("Starting self-training...")
    trainer.train()

    # Save the self-trained model
    print("Saving the self-trained model...")
    trainer.save_model(args.out_folder)
    tokenizer.save_pretrained(args.out_folder)

if __name__ == "__main__":
    main()
