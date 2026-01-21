import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt  # 核心库

# ==========================================
# 1. 模拟数据 (Data Simulation)
# ==========================================
# 我们根据原图中每个国家的分布特征（均值、离散程度）生成假数据
np.random.seed(42)
n_samples = 500  # 模拟 Bootstrap 样本量

# 定义每个国家的分布参数 (根据目测调整)
# loc=均值, scale=标准差 (控制胖瘦/离散度)
data_config = {
    "Burkina Faso":    {"loc": 1.0, "scale": 0.35},
    "Ghana":           {"loc": 1.3, "scale": 0.45},
    "Grenada":         {"loc": 1.25, "scale": 0.3},
    "Kuwait":          {"loc": 1.2, "scale": 0.4},
    "North Macedonia": {"loc": 0.95, "scale": 0.3},
    "San Marino":      {"loc": 1.6, "scale": 0.4}, # 均值较高
    "Saudi Arabia":    {"loc": 1.0, "scale": 0.35},
    "Turkmenistan":    {"loc": 1.0, "scale": 0.38},
}

# 对应的 P 值标注 (从图中抄录)
p_values = [
    "p = 1.2e-26", "p = 1.0e+00", "p = 1.0e+00", "p = 6.8e-01",
    "p = 1.2e-26", "p = 8.5e-01", "p = 1.1e-03", "p = 3.0e-09"
]

# 生成 DataFrame
df_list = []
for country, params in data_config.items():
    # 生成正态分布数据
    values = np.random.normal(params["loc"], params["scale"], n_samples)
    # 截断数据，模拟金牌数 >= 0
    values = np.clip(values, 0, 4) 
    
    temp_df = pd.DataFrame({"Country": country, "Value": values})
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

# ==========================================
# 2. 绘图代码 (Plotting)
# ==========================================
# 设置画布和风格
f, ax = plt.subplots(figsize=(14, 6))
sns.set_style("whitegrid")
sns.set_context("talk") # 调整字体大小适合论文

# 颜色调色板 (使用柔和色系)
pal = sns.color_palette("ch:s=.25,rot=-.25", n_colors=8) # 或者用 "Set2", "Pastel1"

# --- 核心函数: RainCloud Plot ---
pt.RainCloud(
    x = "Country",       # X轴分类
    y = "Value",         # Y轴数值
    data = df, 
    palette = pal,       # 颜色
    bw = .25,            # 核密度平滑度 (Bandwidth)
    width_viol = .5,     # 云(小提琴)的宽度
    ax = ax,             # 画布对象
    orient = "v",        # 垂直方向 (v=vertical, h=horizontal)
    alpha = 0.6,         # 透明度
    dodge = True,        # 避让
    pointplot = True,    # 显示散点(雨)
    move = .2,           # 散点和小提琴的距离
    jitter = 0.8         # 散点的抖动幅度
)

# ==========================================
# 3. 细节修饰 (Annotation)
# ==========================================

# 添加 P-value 文字 (这一步需要手动加，因为 P 值是统计结果，不是原始数据)
countries = list(data_config.keys())
y_limit_top = 3.5 # 文字的高度位置

for i, p_text in enumerate(p_values):
    # 在每个柱子上方写字
    ax.text(
        x = i, 
        y = y_limit_top, 
        s = p_text, 
        ha = 'center',     # 水平居中
        va = 'center', 
        rotation = 90,     # 旋转90度
        fontsize = 12, 
        fontweight = 'bold',
        color = 'black'
    )

# 调整标签
plt.title("Violin-Raincloud Hybrid Plot (Reproduced)", fontsize=16, pad=20)
plt.xlabel("")  # 移除X轴标题
plt.ylabel("Value", fontsize=14, fontweight='bold')

# 旋转 X 轴国家名字，防止重叠
plt.xticks(rotation=45, ha='right', fontsize=12)

# 设置Y轴范围，留出空间给上面的字
plt.ylim(-0.2, 4.2)

plt.tight_layout()
plt.show()