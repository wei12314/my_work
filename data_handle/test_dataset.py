import datasets

datas = datasets.load_from_disk("/home/bdhapp/ft/my_work/datasets/filter_fine_DISC")
print(datas[:5])
print(len(datas))

def get_raw_text(sample):
    dialogue = sample["conversation"]
    raw_text = ""
    for d in dialogue:
        raw_text += d["content"]

    if sample["is_diabetes"] == "true":
        label = 1
    else:
        label = 0
    return {"raw": raw_text, "label": label}

def filter_by_turn(example):
    dialogue = example["conversation"]
    turn = []
    for d in dialogue:
        turn.append(d['role'])

    if len(turn) % 2 != 0:
        return False
    
    for i, t in enumerate(turn):
        if (i % 2) == 0:
            if t != 'user':
                return False
        else:
            if t != 'assistant':
                return False
    return True

# filter_datas = datas.filter(lambda x : x["label"] == 1)
# print(len(filter_datas))
# print(filter_datas["conversation"][:10])
# datas = datas.map(get_raw_text)
# print(datas.column_names)
# datas.save_to_disk("/home/bdhapp/ft/my_work/datasets/test_raw_DISC")

filter_datas = datas.filter(filter_by_turn)
filter_datas.save_to_disk("/home/bdhapp/ft/my_work/datasets/fine_end_DISC")
# print(len(filter_datas))
# print(filter_datas[:5])