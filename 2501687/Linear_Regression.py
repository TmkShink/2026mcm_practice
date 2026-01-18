import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ==========================================
# 1. 准备数据
# ==========================================

# 数据组 (a): Population
data_a = pd.DataFrame({
    'Year': [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
             2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
             2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 
             2020, 2021, 2022, 2023],
    'Population': [26500, 26200, 27000, 27900, 28300, 28500, 28800, 29100, 29600, 30100,
                   30600, 30400, 30300, 30700, 30900, 31100, 31200, 30900, 31000, 31400,
                   32100, 32300, 32600, 32800, 32700, 33100, 32900, 32500, 32300, 32000,
                   31800, 31700, 31500, 31400]
})

# 数据组 (b): Median Personal Income
data_b = pd.DataFrame({
    'Year': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    'Income': [38200, 38700, 37800, 40300, 40800, 40900, 42300, 43400, 42700, 43200, 45000]
})
# 从CSV读取数据
# 确保 csv 文件和你的 py 脚本在同一个文件夹下
# data_a = pd.read_csv('population.csv')
# data_b = pd.read_csv('income.csv')
# ==========================================
# 2. 绘图设置
# ==========================================
# 设置风格，seaborn的 "whitegrid" 最接近原图
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ------------------------------------------
# 绘制图 (a) Population
# ------------------------------------------
sns.regplot(ax=axes[0], x='Year', y='Population', data=data_a,
            color="#5D9BCF",   # 接近原图的浅蓝色
            scatter_kws={'s': 30}, # 散点大小
            line_kws={'color': '#5D9BCF', 'linewidth': 2}, # 回归线样式
            ci=95) # 95% 置信区间（阴影部分）

axes[0].set_title("(a)", y=-0.15, fontsize=14) # 把标题放在底部
axes[0].set_xlim(1988, 2025) # 调整X轴范围
axes[0].set_ylim(26000, 34000) # 调整Y轴范围
# 添加方程注释
axes[0].text(2005, 33500, r'$y = -271606.1 + 150.65x, \ R^2=0.78$', 
             fontsize=11, color='#5D9BCF', fontweight='bold')

# ------------------------------------------
# 绘制图 (b) Median Personal Income
# ------------------------------------------
# 原图 (b) 也是偏绿色/灰绿色的调子
sns.regplot(ax=axes[1], x='Year', y='Income', data=data_b,
            color="#5F9EA0",   # 接近原图的灰绿色/CadetBlue
            scatter_kws={'s': 30},
            line_kws={'color': '#5F9EA0', 'linewidth': 2},
            ci=95)

axes[1].set_title("(b)", y=-0.15, fontsize=14)
axes[1].set_ylabel("Median Personal Income")
axes[1].set_xlim(2010, 2022)
axes[1].set_ylim(37000, 46000)
# 添加方程注释
axes[1].text(2014, 45000, r'$y = -1338185 + 684.24x, \ R^2=0.92$', 
             fontsize=11, color='#5F9EA0', fontweight='bold')

# ==========================================
# 3. 显示结果
# ==========================================
plt.tight_layout()
plt.show()