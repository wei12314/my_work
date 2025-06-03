import datasets

datas = datasets.load_from_disk("/home/bdhapp/ft/my_work/datasets/Bert_check_DISC")
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

    


filter_datas = datas.filter(lambda x : x["label"] == 1)
print(len(filter_datas))
print(filter_datas["conversation"][:10])
# datas = datas.map(get_raw_text)
print(datas.column_names)
# datas.save_to_disk("/home/bdhapp/ft/my_work/datasets/test_raw_DISC")