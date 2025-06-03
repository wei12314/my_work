from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
import evaluate
import numpy as np

raw_datasets = load_from_disk("/home/bdhapp/ft/my_work/datasets/1k_train_DISC")
checkpoint = "/home/bdhapp/ft/my_work/models/chinese-roberta-wwm-ext-large"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

def compute_metrics(eval_preds):
    metric = evaluate.combine(["f1", "accuracy", "precision"])
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

tokenized_datasets = raw_datasets.map(tokenize_function)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(output_dir="/home/bdhapp/ft/my_work/models/1k_bert_3",
                                  eval_strategy="epoch")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()