import datasets

# 定义糖尿病相关关键词列表（可根据需要扩展）
diabetes_keywords = [
    "糖尿病", "血糖", "胰岛素", "降糖药", "糖化血红蛋白",
    "空腹血糖", "餐后血糖", "1型糖尿病", "2型糖尿病",
    "糖尿病肾病", "糖尿病足", "高血糖", "低血糖",
    "葡萄糖", "糖耐量", "口服葡萄糖耐量试验", "OGTT",
    "HbA1c", "二甲双胍", "磺脲类", "糖尿病并发症"
]

datas = datasets.load_dataset("/home/bdhapp/ft/my_work/datasets/DISC-Med-SFT", split='train')
print(len(datas))
print(type(datas[0]["conversation"]))
# 定义关键词筛选函数
def contains_diabetes_keywords(sample):
    dialogue = str(sample["conversation"])
    
    # 检查是否包含任意关键词
    return any(keyword in dialogue for keyword in diabetes_keywords) and len(sample) > 2

def compute_dialogue_length(example):
    return {"length": len(example)}

diabetica_datas = datas.filter(contains_diabetes_keywords)
# diabetica_datas = diabetica_datas.map()

diabetica_datas.save_to_disk("/home/bdhapp/ft/my_work/datasets/clean_DISC-Med-SFT")
# print(len(diabetica_datas))

# print(datas[:10])