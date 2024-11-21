# finetune.py

import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import json
from datasets import load_dataset, Dataset
import deepspeed

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune the base model with distilled data.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the base model.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data (distilled data).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size per device during evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of update steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--fp16", action='store_true', help="Whether to use fp16 mixed precision training.")
    parser.add_argument("--bf16", action='store_true', help="Whether to use bf16 mixed precision training.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config file.")
    parser.add_argument("--overwrite_output_dir", action='store_true', help="Overwrite the content of the output directory.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length.")
    return parser.parse_args()

def load_data(train_data_path):
    # Load the distilled data from the JSONL file
    data_list = []
    with open(train_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            data_list.append(data)
    # Convert to a Hugging Face Dataset
    dataset = Dataset.from_list(data_list)
    return dataset

def main():
    args = parse_arguments()

    # Load the tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # Load the dataset
    print("Loading dataset...")
    dataset = load_data(args.train_data_path)

    # Preprocess the data
    def preprocess_function(examples):
        inputs = []
        for question, answer in zip(examples['question_content'], examples['answer']):
            # Construct the input-output pair as a single string
            text = f"Question: {question}\nAnswer: {answer}"
            inputs.append(text)
        # Tokenize the inputs
        model_inputs = tokenizer(inputs, max_length=args.max_seq_length, truncation=True)
        # Use the same inputs as labels for language modeling
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        deepspeed=args.deepspeed,
        report_to="none",
        prediction_loss_only=True,
        dataloader_drop_last=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save the final model
    print("Saving the model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
