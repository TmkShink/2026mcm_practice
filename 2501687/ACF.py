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

print("原始数据预览:")
print(df.values)
print("-" * 30)

# ==========================================
# 2. 第一步：肉眼定阶 (ACF 和 PACF)
# ==========================================
# 对应论文 Figure 7(a)
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# 画 ACF (自相关) -> 用来看 q (MA阶数)
# lags=5 因为数据只有12个，滞后阶数不能太大
plot_acf(df, lags=5, ax=axes[0], title="Autocorrelation (Determine q)")

# 画 PACF (偏自相关) -> 用来看 p (AR阶数)
plot_pacf(df, lags=5, ax=axes[1], title="Partial Autocorrelation (Determine p)")

plt.tight_layout()
plt.show()

print("图 1 (ACF/PACF) 已生成。请观察截尾和拖尾现象。")
print("-" * 30)