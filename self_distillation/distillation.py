from datasets import load_from_disk
from distilabel.models.llms import TransformersLLM, OpenAILLM
from prompt import SYSTEM_PROMPT, SUGGEST_PROMPT, USER_PROMPT, USER_PROMPT_V2
import json
from openai import OpenAI
from typing import Dict, List, Any
from dotenv import load_dotenv
import os


# raw_dataset = load_from_disk("/home/bdhapp/ft/my_work/datasets/fine_end_DISC")

# for d in raw_dataset:
#     print(d)
#     break

# 1. 提取出user的话，到数组中
# 2. 按照轮次，将history+user对话数组中的话，传给原始qwen2.5 7B，让其生成响应，需要一个history数组，将新的user话和模型生成对话，加入其中
# 3. 将history，传给豆包等大模型，让其参考原始对话，并对模型多轮问答，最后一轮进行建议
# 4. 将建议+最后一轮问答对，传给qwen2.5 7B 2，让其改善输出，将该输出存入到结果数组、
# 5. 保存结果数组、history多轮对话

def self_distillation(example: Dict[str, Any]):
    dialogues = example["conversation"] # 获得一条对话
    user_dialogues = [] # 用户说的话
    real_doctor = [] # 真实医生说的话
    history = [] # 历史记录
    result = [] # 优化结果
    result_dialogues = [] # 优化结果合并
    suggest = [] # 传给模型提建议
    suggest_responses = [] # 模型输出的建议

    for d in dialogues:
        if d['role'] == 'user':
            user_dialogues.append(d['content'])
        else:
            real_doctor.append(d['content'])
    
    for user_d, real_d in zip(user_dialogues, real_doctor):
        history.append({'role': 'user', 'content':user_d})
        suggest.append({'role': 'user', 'content':user_d})

        response = llm_chat(history)
        history.append(response)
        suggest.append(response)

        suggest.append({'role': 'real_doctor', 'content':real_d})
    
    # print(history)
    # return

        suggest_response = gpt_suggest(suggest)
        suggest_response_json = json.loads(suggest_response)
        suggest_responses.append(suggest_response_json)
        # optimize_response = llm_chat_optimize(history, suggest_response_json)
        qa_optimize_response = llm_qa_optimize(history, suggest_response_json)
        result.append(qa_optimize_response)
    print("-------history--------")
    print(history)
    print("-------suggestion--------")
    print(suggest_responses)
    print("-------optimize--------")
    print(result)
    # print(json.loads(result[0]))
    return

    #     optimize_response = llm_chat_optimize(history, coze_response)
    #     result.append(optimize_response)
    
    # return {'history': history, 'optimize_chat': result}

# 使用slm生成对话
def llm_chat(messages: List[Dict[str, str]]) -> Dict[str, str]:
    system_message = system_message_for(SYSTEM_PROMPT)
    messages = [system_message] + messages

    output = llm.generate_outputs(inputs=[messages], max_new_tokens=1024, temperature=0.3)
    content = output[0]['generations'][0]
    return {"role": "assistant", "content": content}
    

# TODO: 看情况完成coze建议
def coze_suggest(history):
    pass

# 使用大模型给出相关建议和优化后的对话
def gpt_suggest(suggest: List[Dict[str, str]]) -> str:
    suggest_str = json.dumps(suggest, ensure_ascii=False)
    user_prompt = SUGGEST_PROMPT.format(dialogue = suggest_str)

    response = client.chat.completions.create(
        model = remote_model,
        messages= [{'role':'user','content':user_prompt}],
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content

# 使用原始slm根据大模型建议，对自己生成的对话进行优化
def llm_chat_optimize(history: List[Dict[str, str]], suggest: Dict[str, str]) -> Dict[str, str]:
    user_chat = history[-2]["content"]
    assistant_chat = history[-1]["content"]
    dialogue = json.dumps([{"role":"user", "content":user_chat}, {"role":"assistant", "content":assistant_chat}],ensure_ascii=False)
    suggestion = suggest["suggestion"]
    user_prompt = USER_PROMPT.format(suggestion = suggestion, dialogue = dialogue)
    output = llm.generate_outputs(inputs=[[{"role":"user", "content":user_prompt}]], max_new_tokens=1024, temperature=0.3)
    content = output[0]['generations'][0]
    return {"role": "assistant", "content": content}

# 使用原始slm根据大模型建议，对自己生成的对话进行优化
def llm_qa_optimize(history: List[Dict[str, str]], suggest: Dict[str, str]) -> Dict[str, str]:
    user_chat = history[-2]["content"]
    dialogue = suggest["fine_assistant_dialogue"]
    user_prompt = USER_PROMPT_V2.format(question = user_chat, dialogue = dialogue)
    output = llm.generate_outputs(inputs=[[{"role":"user", "content":user_prompt}]], max_new_tokens=1024, temperature=0.3)
    content = output[0]['generations'][0]
    return {"role": "assistant", "content": content}

def system_message_for(system_prompt):
    return {'role':'system', 'content':system_prompt}

if __name__ == "__main__":
    load_dotenv()
    # api_key = os.getenv("GM_SECRET_KEY")
    # base_url = os.getenv("GM_BASE_URL")
    # remote_model = os.getenv("GM_MODEL")
    api_key = os.getenv("DS_SECRET_KEY")
    base_url = os.getenv("DS_BASE_URL")
    remote_model = os.getenv("DS_MODEL")

    raw_dataset = load_from_disk("/home/bdhapp/ft/my_work/datasets/fine_end_DISC")
    llm = TransformersLLM(model="/home/bdhapp/ft/models/qwen2.5-7B-Instruct")
    llm.load()
    client = OpenAI(api_key=api_key, base_url=base_url)

    for d in raw_dataset:
        self_distillation(d)
        break