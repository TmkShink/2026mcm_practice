import numpy as np
import matplotlib.pyplot as plt
##cos余弦函数


# 1. 准备数据
# 模拟一年 365 天
t = np.linspace(0, 365, 365)

# 定义一个余弦函数来模拟游客自然分布
# 使用 -cos 让波峰出现在中间（夏天），波谷在两边（冬天）
# A=10000 (振幅), B=5000 (基准线)
natural_distribution = -10000 * np.cos(2 * np.pi * t / 365) + 8000
# 把小于0的部分修正为0（不可能有负数个游客）
natural_distribution = np.maximum(natural_distribution, 0)

# 定义政策限制的阈值
c1 = 14000  # 旺季上限 (削峰)
c2 = 2000   # 淡季下限 (填谷)

# 计算受政策影响后的曲线
# 逻辑：先取最大值(填谷)，再取最小值(削峰)
policy_distribution = np.maximum(natural_distribution, c2) # 填谷
policy_distribution = np.minimum(policy_distribution, c1)  # 削峰

# 2. 开始画图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) # 创建左右两张子图

# --- 画左图 (a) ---
# 绘制曲线
ax1.plot(t, natural_distribution, color='blue', alpha=0.6, label='Natural')
# 填充颜色 (模拟那个渐变效果通常比较复杂，这里用半透明蓝色代替)
ax1.fill_between(t, natural_distribution, color='purple', alpha=0.5)
# 画分割虚线 (模拟旺季淡季分界线)
ax1.vlines(x=[100, 265], ymin=0, ymax=18000, colors='white', linestyles='dashed')
# 设置标题和标签
ax1.set_title('(a) Natural distribution', y=-0.15) # 标题放到底部
ax1.set_ylim(0, 20000)
ax1.set_ylabel('N')
ax1.set_xticks([]) # 隐藏x轴刻度
ax1.spines['top'].set_visible(False)   # 去掉上边框
ax1.spines['right'].set_visible(False) # 去掉右边框
# 添加文字标注
ax1.text(50, 10000, 'off-\nseason', ha='center')
ax1.text(182, 10000, 'peak\nseason', ha='center')
ax1.text(315, 10000, 'off-\nseason', ha='center')


# --- 画右图 (b) ---
# 绘制原始虚线（作为背景对比）
ax2.plot(t, natural_distribution, color='gray', alpha=0.3, linestyle='--')
# 绘制政策后的实线
ax2.plot(t, policy_distribution, color='blue', alpha=0.8)
# 填充颜色
ax2.fill_between(t, policy_distribution, color='purple', alpha=0.6)
# 画横向限制线 (c1 和 c2)
ax2.hlines(y=c1, xmin=0, xmax=365, colors='black', linestyles='dashed', label='c1 (Limit)')
ax2.hlines(y=c2, xmin=0, xmax=365, colors='black', linestyles='dashed', label='c2 (Support)')
# 画纵向分割线
ax2.vlines(x=[100, 265], ymin=0, ymax=18000, colors='white', linestyles='dashed')

# 设置标题和标签
ax2.set_title('(b) Distribution affected by policy', y=-0.15)
ax2.set_ylim(0, 20000)
ax2.set_ylabel('N')
ax2.set_xticks([])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.text(50, 10000, 'off-\nseason', ha='center')
ax2.text(182, 10000, 'peak\nseason', ha='center')
ax2.text(315, 10000, 'off-\nseason', ha='center')

plt.tight_layout()
plt.show()