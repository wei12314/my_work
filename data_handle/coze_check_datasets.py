import datasets
from coze import CozeWorkFlow
import json

datas = datasets.load_from_disk("/home/bdhapp/ft/my_work/datasets/clean_DISC-Med-SFT")
datas_sample = datas.shuffle(seed=42).select(range(1000))

def coze_request(sample):
    coze = CozeWorkFlow("pat_nPuQ8A1quO2oXV3xgyhkUbnmrzm215lrGTGjSp5QPpNxVGehkwAzZNWGhDp76aE8", "7510065004298403878")
    dialogue = str(sample["conversation"])
    parameters = {"dialogue": f"{dialogue}"}
    resp = coze.requestWorkFlow(parameters)
    return json.loads(resp['data'])

datas_sample = datas_sample.map(coze_request, num_proc=2)
print(datas_sample[:3])
datas_sample.save_to_disk("/home/bdhapp/ft/my_work/datasets/1k_DISC")
