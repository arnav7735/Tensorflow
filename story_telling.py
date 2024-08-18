from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
from datasets import load_dataset, Dataset

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

#The purpose of a data collator is for creation of attention masks and padding

def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

#dataset = load_and_preprocess_dataset(file_path, tokenizer)
data_collator = create_data_collator(tokenizer)

# Tokenize the dataset
'''
dataset = load_and_preprocess_dataset(file_path, tokenizer)

# Save the tokenized dataset to a .pt file
torch.save(dataset, "tokenized_dataset.pt")

print("Tokenized dataset saved successfully.")
'''

import torch

# Load the tokenized dataset from the .pt file
import torch
import os

# Specify the directory where the tokenized dataset is saved
dir_location = r"C:\Users\Arnav\Desktop\Gagan_tokenized"

# Specify the full path to the dataset
load_path = os.path.join(dir_location, "tokenized_dataset.pt")

# Load the tokenized dataset from the specified directory
dataset = torch.load(load_path)

print(f"Tokenized dataset loaded successfully from {load_path}.")


print("Tokenized dataset loaded successfully.")


training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=20,  # Use more epochs for better fine-tuning
    per_device_train_batch_size=10,
    save_steps=10_000,
    save_total_limit=2,
)

# Fine-tune the model using Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

#trainer.train()

from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Specify the directory where the model and tokenizer are saved
load_dir_location = r"C:\Users\Arnav\Desktop\gagan_finetuned_model"

# Load the fine-tuned model and tokenizer from the specified directory
model = GPT2LMHeadModel.from_pretrained(load_dir_location)


print(f"Fine-tuned model and tokenizer loaded successfully from {load_dir_location}.")

load_tokenizer_dir_location = r"C:\Users\Arnav\Desktop\gagan_tokenizer"

# Load the tokenizer from the specified directory
tokenizer = GPT2Tokenizer.from_pretrained(load_tokenizer_dir_location)

print(f"Tokenizer loaded successfully from {load_tokenizer_dir_location}.")

print("Fine-tuned model and tokenizer loaded successfully.")

# Initialize the text generation pipeline
from transformers import pipeline
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Starting text
input_text = "Once upon a time"

# Number of iterations for story generation
num_iterations = 5

# Base max_length
base_max_length = 80

# Generate the story
for i in range(num_iterations):
    # Adjust max_length for each iteration
    current_max_length = base_max_length + (i * 15)
    
    # Generate text with added parameters
    outputs = generator(
        input_text, 
        max_length=current_max_length, 
        num_return_sequences=3, 
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.7,  # Lower temperature for more focused generations
        top_k=50,  # Restrict to top 50 words
        top_p=0.9,  # Nucleus sampling
        no_repeat_ngram_size=2,  # Avoid repeating phrases
        pad_token_id=tokenizer.pad_token_id  # Prevent abrupt padding
    )
    
    # Print and select outputs
    print(outputs[0]['generated_text'])
    print("\n")
    print(outputs[1]['generated_text'])
    print("\n")
    print(outputs[2]['generated_text'])
    print("\n")
    print("Which one do you like the most 0, 1 or 2?")
    print("\n")
    choice = int(input())
    input_text = outputs[choice]['generated_text']  # Update input_text with the selected generated text
print("The final story is:")
print("\n")
print(input_text)

