import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings

# 忽略一些警告，保持输出整洁
warnings.filterwarnings("ignore")

# ==========================================
# 1. 准备数据
# ==========================================
# 模拟 12 个月的失业率数据 (假设 d=0，数据已平稳)
data = [3.5, 3.4, 3.6, 3.7, 3.5, 3.4, 3.3, 3.4, 3.5, 3.6, 3.5, 3.4]
dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
df = pd.Series(data, index=dates)
p_values = range(0, 6)
q_values = range(0, 6)
d = 0 # 论文中已确定 d=0

# 用于存储结果的列表
results_list = []

print("开始进行 BIC 网格搜索...")

for p in p_values:
    for q in q_values:
        try:
            # 建立并训练 ARIMA 模型
            model = ARIMA(df, order=(p, d, q))
            result = model.fit()
            
            # 记录 p, q 和对应的 BIC 分数
            results_list.append([p, q, result.bic])
            # print(f"ARIMA({p},0,{q}) - BIC: {result.bic:.2f}")
        except:
            continue

# 整理成 DataFrame
results_df = pd.DataFrame(results_list, columns=['p', 'q', 'BIC'])

# 转换成矩阵形式，方便画热力图
bic_matrix = results_df.pivot(index='p', columns='q', values='BIC')

# 画热力图
plt.figure(figsize=(8, 6))
# cmap="Purples" 是为了模仿论文的紫色风格
sns.heatmap(bic_matrix, annot=True, fmt=".2f", cmap="Purples")
plt.title("Heatmap of BIC Values (Lower is Better)")
plt.xlabel("MA Order (q)")
plt.ylabel("AR Order (p)")
plt.show()

print("图 2 (BIC 热力图) 已生成。")
print("请在图中寻找颜色最浅（数值最小）的格子，其坐标即为最优 (p, q)。")

# 自动找出最小值
best_row = results_df.loc[results_df['BIC'].idxmin()]
print("-" * 30)
print(f"计算结果建议最优参数: p={int(best_row['p'])}, q={int(best_row['q'])}")