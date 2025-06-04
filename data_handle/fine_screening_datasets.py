from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from datasets import load_from_disk
from utils.filter_datasets import coze_request

raw_datasets = load_from_disk("/home/bdhapp/ft/my_work/datasets/Bert_check_DISC")
print("len: ", len(raw_datasets))
filter_datasets = raw_datasets.filter(lambda x : x["label"] == 1)
print("len: ", len(filter_datasets))

fine_datasets = filter_datasets.shuffle(seed=42)
fine_datasets = fine_datasets.map(coze_request, num_proc=2)
fine_datasets.save_to_disk("/home/bdhapp/ft/my_work/datasets/fine_DISC")
print(fine_datasets[:5])
print("len: ", len(fine_datasets))

filter_fine_datasets = fine_datasets.filter(lambda x : x["is_diabetes"] == "true")
filter_fine_datasets.save_to_disk("/home/bdhapp/ft/my_work/datasets/filter_fine_DISC")
print(filter_fine_datasets[:5])
print("len: ", len(filter_fine_datasets))

# fine_datasets.save_to_disk("")