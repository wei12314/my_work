from datasets import load_from_disk

def generate_id(digits, id):
    padded_id = str(id).zfill(digits)
    return {'id': padded_id}



if __name__ == "__main__":

    raw_datasets = load_from_disk("/home/bdhapp/ft/my_work/datasets/distill/100distill_test")
    print(raw_datasets[:1]["result_dialogues"])