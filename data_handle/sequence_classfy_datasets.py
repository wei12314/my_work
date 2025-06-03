from datasets import load_from_disk

raw_dataset = load_from_disk("/home/bdhapp/ft/my_work/datasets/1k_DISC")

def get_classfy_labels(example):
    dialogue = example["conversation"]
    raw_text = ""
    for d in dialogue:
        raw_text += d["content"]
    if example["is_diabetes"] == "true":
        label = 1
    else:
        label = 0
    return {"text": raw_text, "label": label}

label_dataset = raw_dataset.map(get_classfy_labels)

train_test_dataset = label_dataset.train_test_split(train_size=0.9, seed=42)
# print(train_test_dataset)
train_test_dataset.save_to_disk("/home/bdhapp/ft/my_work/datasets/1k_train_DISC")