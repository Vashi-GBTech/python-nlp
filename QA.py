from datasets import load_dataset, load_metric
from transformers import DistilBertTokenizerFast, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification

# Define the checkpoint and load the dataset
checkpoint = "distilbert-base-uncased"
dataset = load_dataset("boolq")

# Tokenizer initialization
tokenizer = DistilBertTokenizerFast.from_pretrained(checkpoint)

# Tokenization function
def tokenize_function(example):
    encoded = tokenizer(example["question"], example["passage"], truncation=True)
    encoded["labels"] = [int(a) for a in example["answer"]]
    return encoded

# Tokenize the datasets and collate data
tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Training setup
args = TrainingArguments(
    # Define training arguments
    output_dir="./results",
    per_device_train_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=500,  # Set as needed
    save_steps=500,  # Set as needed
    num_train_epochs=3,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
