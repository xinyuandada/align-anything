import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 参数配置
#plt.rcParams['font.family'] = 'Arial'  # 设置字体与示例图一致

# 分布参数
higher_params = (0.4426, 2.2771)  # (mean, std)
lower_params = (0.6497, 1.7529)
diff_acc = 0.7435  # 分类准确率

# 生成坐标数据
x = np.linspace(-5, 12.5, 1000)

# 计算概率密度函数
higher_pdf = norm.pdf(x, *higher_params)
lower_pdf = norm.pdf(x, *lower_params)

# 创建画布
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制分布曲线
ax.plot(x, higher_pdf, color='#1f77b4', lw=2, 
        label=f'Higher rewards (μ={higher_params[0]:.4f}, σ={higher_params[1]:.4f})')
ax.plot(x, lower_pdf, color='#ff7f0e', lw=2,
        label=f'Lower rewards (μ={lower_params[0]:.4f}, σ={lower_params[1]:.4f})')

# 绘制准确率线（示例中的红色曲线可能需要替换为实际数据）
ax.axvline(diff_acc, color='#d62728', linestyle='--', lw=2,
          label=f'Reward diff threshold (acc={diff_acc:.4f})')

# 图表装饰
ax.set_title("Distribution of higher/lower rewards and reward diff", pad=20)
ax.set_xlabel("Reward Value")
ax.set_ylabel("Density")
ax.set_xlim(-5, 12.5)
ax.set_ylim(0, 0.30)
ax.grid(False)

# 添加图例
legend = ax.legend(loc='upper right', frameon=False, 
                  bbox_to_anchor=(1.0, 1.0), handlelength=1.5)

# 添加刻度标签
ax.set_xticks(np.arange(-5, 15, 2.5))
ax.set_yticks(np.arange(0, 0.31, 0.05))

plt.tight_layout()
plt.show()