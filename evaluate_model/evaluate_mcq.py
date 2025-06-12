import pandas as pd
import torch
import re
from tqdm.auto import tqdm
import argparse
from unsloth import FastLanguageModel

seed = 55
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def model_output(user_prompt, model, tokenizer):
    messages = [
        {"role": "user", "content": user_prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    gen_idx = len(inputs[0])

    outputs = model.generate(input_ids = inputs, max_new_tokens = 1024, use_cache = True)
    response = tokenizer.batch_decode(outputs[:, gen_idx:], skip_special_tokens = True)[0]
    return response

def main(args):
    model_name = args.path
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False

    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
    FastLanguageModel.for_inference(model)

    scores = {}
    mcq_list = ['/home/bdhapp/ft/my_work/evaluate_model/data/ZhiCheng_MCQ_A1.csv','/home/bdhapp/ft/my_work/evaluate_model/data/ZhiCheng_MCQ_A2.csv'] # for Chinese
    # mcq_list = ['./data/ZhiCheng_MCQ_A1_translated.csv','./data/ZhiCheng_MCQ_A2_translated.csv'] # for English
    for i, file_path in enumerate(mcq_list):
        file_name = file_path[-6:-4]
        df = pd.read_csv(file_path)
        merged_df = pd.DataFrame(columns=['question', 'response','answer'])
        addPrompt = '\n请根据题干和选项，给出唯一的最佳答案。输出内容仅为选项大写英文字母，不要输出任何其他内容。不要输出汉字。'

        for index, row in tqdm(df.iterrows(), total=len(df)):
            output = {}
            response = model_output(row['question'] + addPrompt, model, tokenizer)
            output["question"] = [row['question']]
            output["response"] = [response]
            output['answer'] = row['answer']
            df_output = pd.DataFrame(output)
            merged_df = pd.concat([merged_df, df_output])

        # calculate MCQ score
        merged_df['responseTrimmed'] = merged_df['response'].apply(lambda x: re.search(r'[ABCDE]', x).group() if re.search(r'[ABCDE]', x) else None)
        merged_df['check'] = merged_df.apply(lambda row: 1 if row['responseTrimmed'] == row['answer'][0] else 0, axis=1)
        score = merged_df['check'].mean()
        name = f"{args.model}{file_name}"
        scores[name] = score

    print(scores)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default ='qwen2.5', type = str)
    parser.add_argument("--path", default = "/home/bdhapp/ft/models/qwen2.5-7B-Instruct", type = str)
    args = parser.parse_args()
    if not args.path:
        print("model path is None please repeate try!!")
    main(args)