from distilabel.steps.tasks import ChatGeneration
from distilabel.models.llms import TransformersLLM

# Consider this as a placeholder for your actual LLM.
chat = ChatGeneration(
    llm = TransformersLLM(model="/home/bdhapp/ft/models/qwen2.5-7B-Instruct")
)

chat.load()

result = next(
    chat.process(
        [
            {
                "messages": [
                    {"role": "user", "content": "2+2等于多少"},
                    {"role": "assistant", "content": "2+2等于4"},
                    {"role": "user", "content": "为什么等于4，请你以小学生能懂的方式给我讲解一下"}
                ]
            }
        ]
    )
)
print(result)
# result
# [
#     {
#         'messages': [{'role': 'user', 'content': 'How much is 2+2?'}],
#         'model_name': 'mistralai/Mistral-7B-Instruct-v0.2',
#         'generation': '4',
#     }
# ]