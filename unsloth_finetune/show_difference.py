import gradio as gr
from datasets import load_from_disk
import json

# 配置路径与列名
DATASET_PATH = "/home/bdhapp/ft/my_work/datasets/distill/1000_distill_test"  # 例如 "./my_dataset"
# COLUMN_1 = "conversation"   # 替换为你实际的数据列名
COLUMN_1 = "suggest"
COLUMN_2 = "result_dialogues"  # 替换为你实际的数据列名

# 加载 Dataset
dataset = load_from_disk(DATASET_PATH)
total_samples = len(dataset)

def format_json(data):
    """格式化JSON数据"""
    try:
        if isinstance(data, str):
            data = json.loads(data)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        return str(data)

def navigate_data(current_index, direction):
    """导航数据集"""
    # 处理边界情况
    if direction == "next":
        new_index = current_index + 1
    elif direction == "prev":
        new_index = current_index - 1
    else:  # reset
        new_index = 0
    
    # 检查索引范围
    if new_index < 0:
        return 0, "// 已在第一条数据", "", ""
    elif new_index >= total_samples:
        return total_samples - 1, "// 已在最后一条数据", "", ""
    
    # 获取并格式化数据
    col1 = format_json(dataset[new_index][COLUMN_1])
    col2 = format_json(dataset[new_index][COLUMN_2])
    
    # 更新状态消息
    status = f"显示 {new_index + 1}/{total_samples}"
    return new_index, status, col1, col2

# === Gradio 界面 ===
with gr.Blocks() as demo:
    gr.Markdown("## 浏览 🤗 HuggingFace Dataset 的两列 JSON")
    
    # 状态显示
    index_state = gr.State(0)  # 存储当前索引
    status_display = gr.Markdown(f"准备显示数据 (共 {total_samples} 条)")
    
    with gr.Row():
        with gr.Column():
            col1_view = gr.Code(label=f"{COLUMN_1}", language="json")
        with gr.Column():
            col2_view = gr.Code(label=f"{COLUMN_2}", language="json")
    
    with gr.Row():
        prev_btn = gr.Button("Previous")
        next_btn = gr.Button("Next")
        reset_btn = gr.Button("Reset")
    
    # 按钮事件处理
    def handle_next(current_index):
        return navigate_data(current_index, "next")
    
    def handle_prev(current_index):
        return navigate_data(current_index, "prev")
    
    def handle_reset(current_index):
        return navigate_data(current_index, "reset")
    
    # 绑定事件
    next_btn.click(
        handle_next,
        inputs=[index_state],
        outputs=[index_state, status_display, col1_view, col2_view]
    )
    
    prev_btn.click(
        handle_prev,
        inputs=[index_state],
        outputs=[index_state, status_display, col1_view, col2_view]
    )
    
    reset_btn.click(
        handle_reset,
        inputs=[index_state],
        outputs=[index_state, status_display, col1_view, col2_view]
    )
    
    # 初始加载第一条数据
    def load_first():
        return navigate_data(-1, "next")
    
    demo.load(
        load_first,
        inputs=[],
        outputs=[index_state, status_display, col1_view, col2_view]
    )

demo.launch(server_name="0.0.0.0", server_port=7862)