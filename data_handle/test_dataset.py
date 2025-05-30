import datasets

datas = datasets.load_from_disk("/home/bdhapp/ft/my_work/datasets/test_DISC")
print(len(datas))

def get_raw_text(sample):
    dialogue = sample["conversation"]
    raw_text = ""
    for d in dialogue:
        raw_text += d["content"]
    return {"raw": raw_text}


# filter_datas = datas.filter(lambda x : x["is_diabetes"] == "true")
# print(len(filter_datas))
datas = datas.map(get_raw_text)
print(datas[:3])
# datas.save_to_disk("/home/bdhapp/ft/my_work/datasets/test_raw_DISC")