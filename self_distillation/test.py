import os
from openai import OpenAI

# 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
# 初始化Openai客户端，从环境变量中读取您的API Key
client = OpenAI(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 从环境变量中获取您的 API Key
    api_key="f9ee98b7-5658-4a78-b0ef-4b035bd0dacf",
)

# Non-streaming:
print("----- standard request -----")
completion = client.chat.completions.create(
    # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
    model="deepseek-v3-250324",
    messages=[
        {"role": "system", "content": "你是人工智能助手"},
        {"role": "user", "content": "你好"},
    ],
)
print(completion.choices[0].message.content)

# Streaming:
print("----- streaming request -----")
response = client.chat.completions.create(
    # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
    model="deepseek-v3-250324",
    messages=[
        {"role": "system", "content": "你是人工智能助手"},
        {"role": "user", "content": "你好"},
    ],
    # 响应内容是否流式返回
)
print(response.choices[0].message.content)