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
    load_in_4bit = False
    model_name = "/home/bdhapp/ft/my_work/finetune_models/1k_epoch_3/checkpoint-375"

    # model_epoch3, tokenizer = FastLanguageModel.from_pretrained(
    #         model_name = "/home/bdhapp/ft/my_work/finetune_models/epoch_3/checkpoint-36", # YOUR MODEL YOU USED FOR TRAINING
    #         max_seq_length = max_seq_length,
    #         dtype = dtype,
    #         load_in_4bit = load_in_4bit,
    #     )
    # FastLanguageModel.for_inference(model_epoch3) # Enable native 2x faster inference

    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
    FastLanguageModel.for_inference(model)

    demo = gr.ChatInterface(inference_model, type="messages",examples=["空腹血糖6.8，正常吗？", "糖尿病患者，神经系统会不会受到影响，比如记忆力大不如前，脾气暴躁，正常对话有点困难", "饮食怎么注意呀"], title="糖尿病健康管理对话助手", save_history=True)

    demo.launch(server_name="0.0.0.0", server_port=7861)