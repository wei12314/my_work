from unsloth import FastLanguageModel
import gradio as gr


def inference_model(user_prompt, history):

    messages = history + [
        {"role": "user", "content": user_prompt},
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

if __name__ == "__main__":
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True

    # model_epoch3, tokenizer = FastLanguageModel.from_pretrained(
    #         model_name = "/home/bdhapp/ft/my_work/finetune_models/epoch_3/checkpoint-36", # YOUR MODEL YOU USED FOR TRAINING
    #         max_seq_length = max_seq_length,
    #         dtype = dtype,
    #         load_in_4bit = load_in_4bit,
    #     )
    # FastLanguageModel.for_inference(model_epoch3) # Enable native 2x faster inference

    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "/home/bdhapp/ft/models/qwen2.5-7B-Instruct", # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
    FastLanguageModel.for_inference(model)

    demo = gr.ChatInterface(inference_model, type="messages",examples=["空腹血糖6.8，正常吗？", "hola", "merhaba"], title="糖尿病健康管理对话助手")

    demo.launch("server_name=0.0.0.0")