from datasets import load_from_disk
from unsloth import FastLanguageModel
import argparse


def get_multiturn(example):
    dialogues = example["conversation"] # 得到对话
    history = []
    user_dialogues = [] # 用户说的话
    for d in dialogues:
        if d['role'] == 'user':
            user_dialogues.append(d['content'])
    
    for user_dialogue in user_dialogues:
        history.append({"role":"user", "content": user_dialogue})
        response = model_chat(history)
        history.append({"role":"assistant", "content": response})
    return {"model_generation": history}


def model_chat(history):

    inputs = tokenizer.apply_chat_template(
        history,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    gen_idx = len(inputs[0])
    outputs = model.generate(input_ids = inputs, max_new_tokens = 1024, use_cache = True)
    response = tokenizer.batch_decode(outputs[:, gen_idx:], skip_special_tokens = True)[0]
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default ='qwen2.5', type = str)
    parser.add_argument("--path", default = "/home/bdhapp/ft/models/qwen2.5-7B-Instruct", type = str)
    args = parser.parse_args()
    model_name = args.path
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True


    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
    FastLanguageModel.for_inference(model)

    raw_dataset = load_from_disk("/home/bdhapp/ft/my_work/datasets/fine_end_DISC")
    some_dataset = raw_dataset.shuffle(seed=24).select(range(10))
    print(len(some_dataset))
    dataset_one = some_dataset.map(get_multiturn)
    print(dataset_one[:2])