import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Generating example data
# /home/bdhapp/ft/my_work/finetune_models/epoch_3/checkpoint-36/trainer_state.json
# 得到微调过程中，训练状态对象
def get_train_loss_history(location):
    with open(location, "r", encoding="utf-8") as f:
        train_state = json.load(f)
    return train_state["log_history"] if train_state["log_history"] else None

# 得到训练状态中的step和loss并转换为np.array
def get_step_nploss(distill_history, raw_history):
    distill_loss = []
    raw_loss = []
    step = []
    for d_obj, raw_obj in zip(distill_history, raw_history):
        step.append(d_obj["step"])
        distill_loss.append(d_obj["loss"])
        raw_loss.append(raw_obj["loss"])
    return np.array(step), np.array(distill_loss), np.array(raw_loss)

# 绘制损失函数图片
def draw_img(step, distill_loss, raw_loss):
    # Create a DataFrame
    df = pd.DataFrame({
        'Step': step,
        'Distill': distill_loss,
        'Raw': raw_loss,
    })

    # Set the seaborn style
    sns.set_theme(style="whitegrid")

    # Create the line plot with shaded uncertainty
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Step', y='Distill', data=df, color="blue", label="Distill", errorbar="sd")
    sns.lineplot(x='Step', y='Raw', data=df, color="orange", label="Raw", errorbar="sd")

    # Adding the legend
    plt.legend(title="Legend")

    # Adding titles and labels
    plt.title('Raw vs Distill', fontsize=16)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)


    # 保存图片到本地（可修改路径）
    save_path = save_img_path
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至：{save_path}")

    # 显示图表（若不需要弹出显示，可注释掉）
    plt.show()

if __name__ == "__main__":
    dir_img = Path("/home/bdhapp/ft/my_work/draw_img/loss_img")
    img_name = Path("test_1k.png")
    save_img_path = dir_img / img_name

    path_distill = Path("/home/bdhapp/ft/my_work/finetune_models/1k_epoch_3/checkpoint-375") / "trainer_state.json"
    path_raw = Path("/home/bdhapp/ft/my_work/finetune_models/1k_raw_epoch_3/checkpoint-375") / "trainer_state.json"
    distill_history = get_train_loss_history(path_distill)
    raw_history = get_train_loss_history(path_raw)
    
    step, distill_loss, raw_loss =get_step_nploss(distill_history, raw_history)
    draw_img(step, distill_loss, raw_loss)