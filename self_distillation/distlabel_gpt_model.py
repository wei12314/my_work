from distilabel.models.llms import OpenAILLM

# llm = OpenAILLM(model="deepseek-chat", api_key="sk-b4078ae59856471ebb5b088c8db0d464", base_url="https://api.deepseek.com")
llm = OpenAILLM(model="google/gemini-2.5-flash-preview-05-20", api_key="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", base_url="https://openrouter.ai/api/v1")
llm.load()

output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"}]])
print(output)