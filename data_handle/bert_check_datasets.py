from datasets import load_from_disk
from transformers import pipeline

raw_datasets = load_from_disk("/home/bdhapp/ft/my_work/datasets/clean_DISC-Med-SFT")
print("raw length: ", len(raw_datasets))
checkpoint = "/home/bdhapp/ft/my_work/models/1k_bert_3/checkpoint-339"
classifier = pipeline(task="text-classification", model=checkpoint, max_length=512, truncation=True)

def check_is_diabetes(example):
    dialogues = example["conversation"]
    content = ""
    for dialogue in dialogues:
        content += dialogue["content"]
    res = classifier(content)
    if res[0]['label'] == "LABEL_0":
        label = 0
    else:
        label = 1
    return {"label": label}

bert_check_datasets = raw_datasets.map(check_is_diabetes)
print("bert check length: ", len(bert_check_datasets))

bert_check_datasets.save_to_disk("/home/bdhapp/ft/my_work/datasets/Bert_check_DISC")


# res = pipeline('举而不坚，坚而不久。有时无法完成性生活。今年上半年，感觉很优秀。后来因为咳凑诊所给配的中药（12付其中有川贝粉30克）然后就出现举而不坚，坚而不久。不知道是巧合还是吃中药吃的。后来也在江阴中医院和江阴人民医院看过，医生没有做任何检查，只是配了些药。中医院配的还少胶囊和金水宝胶囊。吃了没有效果。我该怎么办？我这还有希望能治好吗？自我感觉身体挺好的，可以一口气跑8公里。身高1.64米，体重126斤。不吸烟、不喝酒。没有手淫史。平时爱锻炼自我觉得很结实。没有糖尿病和其它毛病。今年上半年还勃起特别有力，勃起后如没有性生活会翘好久')
# print(res)