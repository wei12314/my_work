import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generating example data
np.random.seed(42)
time = np.arange(0, 100, 1)
attacker = np.cumsum(np.random.randn(100)) + 5  # random walk for attacker
target = np.cumsum(np.random.randn(100)) + 10  # random walk for target

# Create a DataFrame
df = pd.DataFrame({
    'Time': time,
    'Attacker': attacker,
    'Target': target
})

# Set the seaborn style
sns.set_theme(style="whitegrid")

# Create the line plot with shaded uncertainty
plt.figure(figsize=(10, 6))
sns.lineplot(x='Time', y='Attacker', data=df, color="blue", label="Attacker", errorbar="sd")
sns.lineplot(x='Time', y='Target', data=df, color="orange", label="Target", errorbar="sd")

# Adding the legend
plt.legend(title="Legend")

# Adding titles and labels
plt.title('Attacker vs Target', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Value', fontsize=12)


# 保存图片到本地（可修改路径）
save_path = "double_line_plot.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图表已保存至：{save_path}")

# 显示图表（若不需要弹出显示，可注释掉）
plt.show()