from distilabel.models.llms import TransformersLLM

llm = TransformersLLM(model="/home/bdhapp/ft/models/qwen2.5-7B-Instruct")

llm.load()

# Call the model
output = llm.generate_outputs(inputs=[[{"role": "user", "content": "Hello world!"},
                                       {"role": "assistant", "content":'Hello! Nice to meet you. How can I assist you today?'},
                                       {"role": "user", "content": "Today is good day!"},]])
print(output)
# [{'generations': ["That's great to hear! A good day can make all the difference. Is there anything specific you'd like to do or achieve today? Whether it's working on a project, spending time with friends and family, or trying something new, I'm here to help or just chat about it with you."], 
# 'statistics': {'input_tokens': [22], 
# 'output_tokens': [61]}}]