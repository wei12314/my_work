import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from datasets import load_from_disk, Dataset
from unsloth import FastLanguageModel
import argparse
import torch
import gc
from tqdm.auto import tqdm
from data_handle.coze import CozeWorkFlow
import json

def prepare_dialogue_history(example):
    """准备对话历史（不包含模型生成）"""
    user_dialogues = [
        {"role": "user", "content": d["content"]} 
        for d in example["conversation"] 
        if d["role"] == "user"
    ]
    return {"user_history": user_dialogues}

def model_chat(history, model, tokenizer):
    """生成单轮对话回复"""
    inputs = tokenizer.apply_chat_template(
        history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    gen_idx = len(inputs[0])
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=1024,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id  # 防止pad token警告
        )
    return tokenizer.batch_decode(outputs[:, gen_idx:], skip_special_tokens=True)[0]

def process_dataset(dataset, model, tokenizer):
    """逐个样本处理对话生成"""
    processed_data = []
    for example in tqdm(dataset, total=len(dataset)):
        full_history = []
        for user_msg in example["user_history"]:
            # 添加用户消息
            full_history.append(user_msg)
            
            # 生成助手回复
            response = model_chat(full_history, model, tokenizer)
            
            # 添加助手回复
            full_history.append({"role": "assistant", "content": response})
            
            # 清理显存
            torch.cuda.empty_cache()
        
        processed_data.append({
            "full_conversation": full_history
        })
        
        # 定期清理内存
        if len(processed_data) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    return Dataset.from_list(processed_data)

def coze_judge(example):
    """coze评分"""
    dialogues = str(example["full_conversation"])
    parameters = {"dialogues": f"{dialogues}"}
    resp = coze.requestWorkFlow(parameters)
    return {"socre": json.loads(resp['data'])}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='qwen2.5', type=str)
    parser.add_argument("--path", default="/home/bdhapp/ft/models/qwen2.5-7B-Instruct", type=str)
    args = parser.parse_args()
    
    # 1. 先加载数据集（不加载模型）
    raw_dataset = load_from_disk("/home/bdhapp/ft/my_work/datasets/fine_end_DISC")
    some_dataset = raw_dataset.shuffle(seed=24).select(range(10))
    prepared_dataset = some_dataset.map(prepare_dialogue_history)
    
    # 2. 单独加载模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,  # 使用4bit量化
        load_in_8bit=True,
    )
    model = FastLanguageModel.for_inference(model)
    
    # 3. 处理数据集
    result_dataset = process_dataset(prepared_dataset, model, tokenizer)
    
    # 4. 初始化coze工作流
    coze = CozeWorkFlow("pat_nPuQ8A1quO2oXV3xgyhkUbnmrzm215lrGTGjSp5QPpNxVGehkwAzZNWGhDp76aE8", "7514935183008645159")
    
    # 5. coze 评分
    score_dataset = result_dataset.map(coze_judge)
    score_dataset.save_to_disk("/home/bdhapp/ft/my_work/evaluate_model/score_data/test2")
    
    # 5. 查看结果
    print(score_dataset[0])