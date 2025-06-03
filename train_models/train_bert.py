from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate



raw_datasets = load_from_disk("/home/bdhapp/ft/my_work/datasets/test_raw_DISC")
checkpoint = "/home/bdhapp/ft/my_work/models/chinese-roberta-wwm-ext-large"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["raw"], truncation=True, max_length=512)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(['_id', 'source', 'conversation', 'is_diabetes', 'reasoning', 'raw'])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

tokenized_datasets.set_format("torch")


train_dataloader = DataLoader(
    tokenized_datasets, shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets[:24], batch_size=8, collate_fn=data_collator
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

for batch in train_dataloader:
    break

print({k: v.shape for k, v in batch.items()})

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)


device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)

progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

metric = evaluate.load("accuracy")
model.eval()
for batch in train_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

results = metric.compute()
print(f"Multi-label Accuracy: {results['accuracy']:.4f}")

model.save_pretrained("/home/bdhapp/ft/my_work/models/test")