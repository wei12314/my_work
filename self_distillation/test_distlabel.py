from distilabel.steps.tasks import ChatGeneration
from distilabel.models import InferenceEndpointsLLM

# Consider this as a placeholder for your actual LLM.
chat = ChatGeneration(
    llm=InferenceEndpointsLLM(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
    )
)

chat.load()

result = next(
    chat.process(
        [
            {
                "messages": [
                    {"role": "user", "content": "How much is 2+2?"},
                ]
            }
        ]
    )
)